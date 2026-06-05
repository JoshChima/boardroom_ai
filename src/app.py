"""
Meeting Intelligence — Phase 2
  1. Transcribe a meeting audio file (or live mic recording) with per-segment timestamps.
  2. Summarize the full meeting via the local LLM (Ollama / Llama 3.1).
  3. Answer questions about the meeting — questions come from a markdown file
     (--questions path) or are typed directly in the terminal.

Usage:
    python app.py meeting.wav [--questions questions.md] [--model llama3.1]
    python app.py              # no file → records from microphone

# --- DEPRECATED (Phase 1 voice assistant) ---
# The Bark TTS pipeline has been commented out.
# See tts.py for the Bark TextToSpeechService class (kept for reference).
# from tts import TextToSpeechService
# tts = TextToSpeechService()
# --------------------------------------------
"""

import argparse
import re
import sys
import time
import threading
from pathlib import Path
from queue import Queue
from typing import Union

import ollama
import numpy as np
import sounddevice as sd
import torch
import whisper
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TimeRemainingColumn, TransferSpeedColumn
from rich.rule import Rule
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

console = Console()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = """\
You are an expert meeting analyst. Below is a timestamped transcript of a meeting.

<transcript>
{transcript}
</transcript>

Provide a concise summary covering:
- Main topics discussed
- Key decisions made
- Action items and owners (if mentioned)
"""

_QA_PROMPT = """\
You are an expert meeting analyst. Below is a timestamped transcript of a meeting.

<transcript>
{transcript}
</transcript>

Question: {question}

Instructions:
- If the question was addressed in the meeting, start your answer with "Addressed:" \
and provide a concise summary of how it was answered. Cite the approximate timestamp \
where relevant.
- If the question was NOT addressed, start with "Not addressed:" and briefly note \
any related topics that were discussed, if any.
"""

# ---------------------------------------------------------------------------
# Audio recording (mic fallback)
# ---------------------------------------------------------------------------


def record_from_microphone() -> np.ndarray:
    """Record from the default microphone; return a float32 mono array at 16 kHz."""
    data_queue: Queue = Queue()
    stop_event = threading.Event()

    def _capture(indata, frames, t, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    def _record_loop():
        with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=_capture):
            while not stop_event.is_set():
                time.sleep(0.1)

    console.input("\n[cyan]Press Enter to start recording…")
    recording_thread = threading.Thread(target=_record_loop, daemon=True)
    recording_thread.start()

    console.input("[cyan]Recording… press Enter to stop.")
    stop_event.set()
    recording_thread.join()

    raw = b"".join(list(data_queue.queue))
    audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcribe_meeting(audio: Union[str, np.ndarray]) -> str:
    """
    Transcribe *audio* (a file path or a float32 numpy array) with Whisper and
    return a timestamped transcript, e.g.:

        [00:12] The budget proposal was approved unanimously.
        [01:45] Action item: Sarah to send the revised deck by Friday.
    """
    console.print("[cyan]Loading Whisper (base.en)…")
    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        console.print("[green]CUDA detected — using fp16 for transcription.")
    stt = whisper.load_model("base.en")

    label = audio if isinstance(audio, str) else "microphone recording"
    console.print(f"[cyan]Transcribing: {label}")
    result = stt.transcribe(audio, fp16=use_fp16, verbose=False)

    lines = []
    for seg in result["segments"]:
        ts = _fmt_ts(seg["start"])
        lines.append(f"[{ts}] {seg['text'].strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def ensure_model_available(model: str) -> None:
    """Pull *model* from Ollama if it is not already installed locally."""
    canonical = model if ":" in model else f"{model}:latest"

    try:
        installed_names = {m.model for m in ollama.list().models}
    except Exception as exc:
        console.print(f"[red]Cannot reach Ollama — is it running?\n{exc}")
        sys.exit(1)

    if canonical in installed_names:
        console.print(f"[green]Model '{model}' is already available.")
        return

    console.print(f"[yellow]Model '{model}' not found locally — pulling from Ollama…")
    try:
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = None
            for response in ollama.pull(model, stream=True):
                completed = response.completed or 0
                total = response.total or 0
                status = response.status or ""
                if task_id is None and total:
                    task_id = progress.add_task(f"Pulling {model}", total=total)
                if task_id is not None and total:
                    progress.update(task_id, completed=completed,
                                    description=f"Pulling {model} — {status}")
    except ollama.ResponseError as exc:
        console.print(f"[red]Pull failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]Unexpected error during pull: {exc}")
        sys.exit(1)

    console.print(f"[green]Model '{model}' pulled successfully.")


def summarize_meeting(transcript: str, llm: ChatOllama) -> str:
    chain = ChatPromptTemplate.from_template(_SUMMARY_PROMPT) | llm
    return chain.invoke({"transcript": transcript}).content.strip()


def answer_question(question: str, transcript: str, llm: ChatOllama) -> str:
    chain = ChatPromptTemplate.from_template(_QA_PROMPT) | llm
    return chain.invoke({"transcript": transcript, "question": question}).content.strip()


# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------


def load_questions_from_markdown(md_path: str) -> list:
    """
    Extract questions from a markdown file.
    Supports bullet lists ("- question", "* question") and numbered lists
    ("1. question", "1) question").
    """
    text = Path(md_path).read_text(encoding="utf-8")
    questions = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^[-*]\s+(.+)", line) or re.match(r"^\d+[.)]\s+(.+)", line)
        if m:
            questions.append(m.group(1).strip())
    return questions


def collect_questions_from_terminal() -> list:
    console.print("\n[yellow]Enter your questions, one per line.")
    console.print("[yellow]Press Enter on a blank line when done.\n")
    questions = []
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            if questions:
                break
        else:
            questions.append(line)
    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a meeting recording, summarize it, and answer questions."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="Path to the meeting audio file (wav, mp3, m4a, …). "
             "If omitted, audio is recorded from the microphone.",
    )
    parser.add_argument(
        "--questions",
        "-q",
        metavar="FILE",
        help="Markdown file containing questions (bullet or numbered list). "
             "If omitted, questions are entered interactively.",
    )
    parser.add_argument(
        "--model",
        default="llama3.1",
        metavar="MODEL",
        help="Ollama model to use (default: llama3.1).",
    )
    args = parser.parse_args()

    ensure_model_available(args.model)
    llm = ChatOllama(model=args.model)

    # ---- Step 1: Acquire audio ---------------------------------------------
    if args.audio:
        if not Path(args.audio).exists():
            console.print(f"[red]Audio file not found: {args.audio}")
            sys.exit(1)
        audio_source = args.audio
    else:
        console.print("[yellow]No audio file provided — switching to microphone recording.")
        audio_source = record_from_microphone()
        if audio_source.size == 0:
            console.print("[red]No audio captured. Exiting.")
            sys.exit(1)

    # ---- Step 2: Transcribe ------------------------------------------------
    console.print(Rule("[bold cyan]Transcription"))
    with console.status("Transcribing…", spinner="earth"):
        transcript = transcribe_meeting(audio_source)
    console.print(transcript)

    # ---- Step 3: Summarize -------------------------------------------------
    console.print(Rule("[bold cyan]Meeting Summary"))
    with console.status("Summarizing…", spinner="earth"):
        summary = summarize_meeting(transcript, llm)
    console.print(summary)

    # ---- Step 4: Q&A -------------------------------------------------------
    questions = []
    if args.questions:
        if not Path(args.questions).exists():
            console.print(f"[red]Questions file not found: {args.questions}")
        else:
            questions = load_questions_from_markdown(args.questions)
            if not questions:
                console.print(
                    f"[yellow]No questions parsed from {args.questions} — "
                    "falling back to terminal input."
                )

    if not questions:
        questions = collect_questions_from_terminal()

    if questions:
        console.print(Rule("[bold cyan]Q&A"))
        for i, q in enumerate(questions, 1):
            console.print(f"\n[bold yellow]Q{i}: {q}")
            with console.status(f"Answering Q{i}…", spinner="earth"):
                answer = answer_question(q, transcript, llm)
            console.print(answer)

    console.print(Rule("[bold green]Done"))


if __name__ == "__main__":
    main()
