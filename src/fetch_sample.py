"""
fetch_sample.py — Download a single MeetingBank sample for local testing.

Datasets used:
  - huuuyeah/meetingbank (HF)         → transcript + summary text
  - data/MeetingBank/Metadata/MeetingBank.json → video URL lookup
  - Granicus CDN                       → mp4 download

Usage:
    python fetch_sample.py                          # first record from test split
    python fetch_sample.py --index 3               # fourth record
    python fetch_sample.py --split train           # from training split
    python fetch_sample.py --random                # random record
    python fetch_sample.py --output sample.json    # save JSON + video to disk
    python fetch_sample.py --no-audio              # skip video download
    python fetch_sample.py --token hf_xxx          # HuggingFace access token
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from datasets import load_dataset
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TimeRemainingColumn, TransferSpeedColumn
from rich.rule import Rule

load_dotenv(Path(__file__).parent.parent / ".env")

console = Console()

TEXT_DATASET = "huuuyeah/meetingbank"

DATA_ROOT = Path(__file__).parent.parent / "data" / "MeetingBank"
MEETINGBANK_JSON = DATA_ROOT / "Metadata" / "MeetingBank.json"


# ---------------------------------------------------------------------------
# Text (transcript + summary)
# ---------------------------------------------------------------------------

def fetch_text_sample(split: str, index: int, use_random: bool) -> dict:
    console.print(f"[cyan]Loading MeetingBank text ({split} split)…")
    ds = load_dataset(TEXT_DATASET, split=split)

    if use_random:
        index = random.randint(0, len(ds) - 1)
        console.print(f"[cyan]Random index: {index} / {len(ds) - 1}")
    elif index >= len(ds):
        console.print(f"[red]Index {index} out of range — split has {len(ds)} records.")
        sys.exit(1)

    record = ds[index]
    return {
        "uid": record["uid"],
        "id": record["id"],
        "split": split,
        "index": index,
        "transcript": record["transcript"],
        "summary": record["summary"],
    }


# ---------------------------------------------------------------------------
# Audio/video
# ---------------------------------------------------------------------------

def _meeting_key(uid: str) -> str:
    """Strip the agenda item suffix from a uid to get the MeetingBank.json key.

    e.g. 'DenverCityCouncil_08072017_17-0807' → 'DenverCityCouncil_08072017'
         'AlamedaCC_01212020'                 → 'AlamedaCC_01212020'
    """
    parts = uid.split("_")
    return "_".join(parts[:2])


def _video_url(uid: str) -> str | None:
    """Look up the Granicus mp4 URL for a meeting uid via the local metadata file."""
    if not MEETINGBANK_JSON.exists():
        console.print(f"[yellow]Metadata file not found: {MEETINGBANK_JSON}")
        console.print("[yellow]Run with --no-audio or download MeetingBank.json first.")
        return None

    with MEETINGBANK_JSON.open(encoding="utf-8") as f:
        metadata = json.load(f)

    key = _meeting_key(uid)
    entry = metadata.get(key)
    if entry is None:
        console.print(f"[yellow]No metadata entry for meeting key '{key}'.")
        return None

    url = entry.get("URLs", {}).get("Video")
    if not url:
        console.print(f"[yellow]No video URL in metadata for '{key}'.")
    return url


def fetch_video(uid: str, out_dir: Path) -> Path | None:
    """Download the mp4 for *uid* using the URL from MeetingBank.json."""
    url = _video_url(uid)
    if not url:
        return None

    filename = url.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if out_path.exists():
        console.print(f"[green]Video already cached → {out_path}")
        return out_path

    if not url.startswith("http"):
        url = "https://" + url

    console.print(f"[cyan]Downloading video: {url}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(filename, total=total or None)
                with out_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        f.write(chunk)
                        progress.advance(task, len(chunk))
    except requests.RequestException as exc:
        console.print(f"[red]Download failed: {exc}")
        if out_path.exists():
            out_path.unlink()
        return None

    console.print(f"[green]Video saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a single MeetingBank sample (text + video) for local testing."
    )
    parser.add_argument(
        "--index", "-i", type=int, default=0,
        help="Zero-based record index within the split (default: 0).",
    )
    parser.add_argument(
        "--split", "-s", default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: test).",
    )
    parser.add_argument(
        "--random", "-r", action="store_true",
        help="Pick a random record instead of --index.",
    )
    parser.add_argument(
        "--output", "-o", metavar="FILE",
        help="Write sample JSON to FILE. Video (if downloaded) is saved alongside it.",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Skip video download; fetch transcript and summary only.",
    )
    parser.add_argument(
        "--token", metavar="HF_TOKEN",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token (or set HF_TOKEN env var).",
    )
    args = parser.parse_args()

    # ---- Text ---------------------------------------------------------------
    sample = fetch_text_sample(args.split, args.index, args.random)

    # ---- Video --------------------------------------------------------------
    video_path: Path | None = None
    if not args.no_audio:
        out_dir = Path(args.output).parent if args.output else Path(".")
        video_path = fetch_video(sample["uid"], out_dir)
        if video_path:
            sample["video_path"] = str(video_path)

    # ---- Output -------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        console.print(f"[green]Saved sample '{sample['uid']}' → {out_path}")
    else:
        console.print(Rule(f"[bold cyan]Meeting: {sample['uid']}"))
        console.print("\n[bold yellow]Transcript")
        console.print(sample["transcript"])
        console.print(Rule("[bold yellow]Summary"))
        console.print(sample["summary"])
        if video_path:
            console.print(Rule(f"[bold yellow]Video → {video_path}"))
        console.print(Rule("[bold green]Done"))


if __name__ == "__main__":
    main()
