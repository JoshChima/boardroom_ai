# Boardroom — Local Meeting Intelligence

A fully local meeting intelligence tool that transcribes, summarizes, and answers questions about your meetings — no cloud, no API keys, no data leaving your machine.

Pass in a pre-recorded audio or video file, or let it record the meeting live from your microphone. It transcribes the audio with timestamps, generates a summary, answers specific questions about the meeting content, and saves a full markdown report for later review. Optionally provide a meeting agenda to give the LLM context about the purpose of the meeting before it processes the transcript.

---

## Acknowledgements

This project was built on the foundation laid by **Duy Huynh**, whose guide on assembling a local voice assistant with Whisper, Ollama, and Bark was the starting point for this project. The meeting intelligence features in this repo grew out of that work.

[Build Your Own Voice Assistant and Run it Locally — Duy Huynh](https://medium.com/@vndee.huynh/build-your-own-voice-assistant-and-run-it-locally-whisper-ollama-bark-c80e6f815cba)

---

## Key Features

- **Timestamped transcription** — every segment of speech is labelled with its start time so you can jump directly to the relevant moment.
- **Automatic meeting summary** — covers main topics, key decisions, and action items.
- **Agenda-aware analysis** — pass in a pre-meeting agenda file and the LLM uses it as context for what the meeting was intended to cover, producing more focused summaries and answers.
- **Q&A over the transcript** — provide questions in a markdown file or type them directly in the terminal; the LLM tells you whether each was addressed and summarises the answer with a timestamp reference.
- **Session reports** — save the full session output (transcript, summary, Q&A) to a timestamped markdown file in a folder of your choice for later review.
- **Video file support** — mp4 and other video formats are accepted directly; FFmpeg extracts the audio automatically.
- **Two audio input modes** — pass a pre-recorded file (wav, mp3, mp4, m4a, and more) or record straight from your microphone.
- **Automatic model management** — on startup the app checks whether the selected Ollama model is installed locally and pulls it automatically if not, with a live progress bar.
- **Fully local** — speech recognition runs via Whisper, language generation runs via Ollama. Nothing is sent to an external service.
- **GPU acceleration** — automatically uses an NVIDIA GPU (CUDA) if one is available, for faster transcription.

---

## Tools and Packages

| Component | Tool |
|---|---|
| Speech-to-text | [OpenAI Whisper](https://github.com/openai/whisper) |
| Language model | [Ollama](https://ollama.com) running Llama 3.1 (or any Ollama model) |
| LLM interface | [LangChain](https://www.langchain.com) (`langchain-ollama`) |
| Microphone recording | [sounddevice](https://python-sounddevice.readthedocs.io) |
| GPU support | [PyTorch](https://pytorch.org) |
| Terminal UI | [Rich](https://github.com/Textualize/rich) |
| Sample data | [MeetingBank](https://meetingbank.github.io) via [Hugging Face](https://huggingface.co/datasets/huuuyeah/meetingbank) |

---

## Prerequisites

### 1. Python

Python **3.10 or later** is required (the codebase uses `X | Y` union type syntax). Check your version with:

```powershell
python --version
```

### 2. Ollama

Ollama runs the language model locally. Install it from [ollama.com](https://ollama.com).

Start the Ollama server if it isn't already running:

```powershell
ollama serve
```

The app automatically checks whether your selected model is available locally when it starts. If it isn't, it pulls it for you — no manual download step required. The default model is `llama3.1`; pass `--model <name>` to use any other Ollama model (e.g. `--model mistral`).

### 3. FFmpeg

FFmpeg is required for Whisper to decode audio and video files (including mp4).

```powershell
winget install ffmpeg
```

After installation, close and reopen your terminal so the PATH update takes effect. Verify with:

```powershell
ffmpeg -version
```

### 4. System audio libraries

These are needed by `sounddevice` for microphone recording.

| OS | Steps |
|---|---|
| **Windows** | No extra step — sounddevice ships with prebuilt PortAudio binaries. |
| **macOS** | `brew install portaudio` |
| **Debian / Ubuntu** | `sudo apt-get install portaudio19-dev` |

### 5. NVIDIA GPU (optional but recommended)

If you have an NVIDIA GPU, install the CUDA-enabled build of PyTorch following the instructions at [pytorch.org/get-started](https://pytorch.org/get-started/locally/). The app detects CUDA automatically and uses fp16 for faster transcription — no configuration needed.

---

## Setup

```powershell
# 1. Clone the repository
git clone <repo-url>
cd boardroom_spt

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage — Main Application

All commands are run from the project root with the virtual environment active.

### Basic usage — audio or video file

```powershell
python src/app.py "meeting.wav"
python src/app.py "meeting.mp4"
```

Supported formats include wav, mp3, mp4, m4a, flac, ogg, and any other format FFmpeg can read.

### Record live from the microphone

Run the app with no audio argument and it will prompt you to record:

```powershell
python src/app.py
```

```
No audio file provided — switching to microphone recording.

Press Enter to start recording…
Recording… press Enter to stop.
```

### Pass a questions file

```powershell
python src/app.py "meeting.mp4" --questions "questions.md"
```

### Pass a meeting agenda

Provide a markdown file describing the purpose and intended topics of the meeting. The LLM uses this as pre-meeting context — it does not replace the transcript as the source of what was actually said and decided.

```powershell
python src/app.py "meeting.mp4" --agenda "agenda.md"
```

### Save a session report

Pass a folder path with `--output-dir` and a timestamped markdown report will be written there after the session completes. The report contains the transcript, summary, and all Q&A pairs.

```powershell
python src/app.py "meeting.mp4" --output-dir "reports/"
```

The output file is named `<audio-filename>_<YYYYMMDD_HHMMSS>.md`.

### Full example

```powershell
python src/app.py "data/seattle_sample/full_061316V.mp4" `
  --questions "data/seattle_sample/questions.md" `
  --agenda   "data/seattle_sample/agenda.md" `
  --output-dir "data/seattle_sample/reports" `
  --model llama3.1
```

### All flags

| Flag | Short | Description |
|---|---|---|
| *(positional)* | | Path to audio/video file. Omit to record from microphone. |
| `--questions` | `-q` | Markdown file with questions (bullet or numbered list). |
| `--agenda` | `-a` | Markdown file with the pre-meeting agenda. |
| `--output-dir` | `-o` | Folder to save the session report. |
| `--model` | | Ollama model to use (default: `llama3.1`). |

---

## Usage — Sample Downloader

`src/fetch_sample.py` downloads a single meeting sample from the [MeetingBank dataset](https://huggingface.co/datasets/huuuyeah/meetingbank) on Hugging Face for use in local testing.

### Setup

Create a `.env` file in the project root with your Hugging Face access token:

```
HF_TOKEN=hf_your_token_here
```

### Basic usage

```powershell
# First record from the test split (transcript + summary + video)
python src/fetch_sample.py

# Specific index
python src/fetch_sample.py --index 5

# Random record
python src/fetch_sample.py --random

# Save to a JSON file (video saved alongside it)
python src/fetch_sample.py --index 3 --output data/sample.json

# Text only — skip video download
python src/fetch_sample.py --index 3 --no-audio --output data/sample.json

# Different split
python src/fetch_sample.py --split train --index 0
```

### All flags

| Flag | Short | Description |
|---|---|---|
| `--index` | `-i` | Zero-based record index within the split (default: `0`). |
| `--split` | `-s` | Dataset split: `train`, `validation`, or `test` (default: `test`). |
| `--random` | `-r` | Pick a random record instead of `--index`. |
| `--output` | `-o` | Save sample JSON to this file path. |
| `--no-audio` | | Skip video download; fetch transcript and summary only. |
| `--token` | | Hugging Face access token (falls back to `HF_TOKEN` env var / `.env`). |

### Output JSON format

```json
{
  "uid": "SeattleCityCouncil_06132016_Res 31669",
  "id": 0,
  "split": "test",
  "index": 0,
  "transcript": "...",
  "summary": "...",
  "video_path": "data/full_061316V.mp4"
}
```

> **Note:** Video downloads only work for Seattle City Council meetings. The Granicus CDN links used by other cities (Alameda, Boston, Denver, King County, Long Beach) are no longer active.

---

## File Formats

### Agenda file (`agenda.md`)

The agenda is a pre-meeting document describing the purpose of the meeting and the topics intended to be discussed. It should reflect what was planned, not what happened. Plain markdown — headings, bullet points, and paragraphs are all fine.

**Example:**

```markdown
# Agenda — Project Kickoff
**Date:** January 15, 2026

## Purpose
Align the team on scope, timeline, and ownership ahead of the Q1 build cycle.

## Topics
- Review the proposed project scope and confirm sign-off
- Agree on a delivery timeline and key milestones
- Assign ownership for each workstream
```

### Questions file (`questions.md`)

A plain markdown file with questions as a bullet or numbered list. Headings and blank lines are ignored.

**Example:**

```markdown
# Questions

- Was the delivery timeline confirmed?
- Who owns the client onboarding workstream?
- Were any risks or blockers raised?
```

---

## Project Structure

```
boardroom_spt/
├── src/
│   ├── app.py              # Main application — transcription, summarisation, Q&A, reports
│   ├── fetch_sample.py     # Sample downloader — pulls a single MeetingBank record
│   ├── tts.py              # Deprecated Phase 1 Bark TTS service (kept for reference)
│   └── questions.md        # Example questions file
├── data/                   # Local data directory (gitignored)
│   └── seattle_sample/     # Example sample downloaded via fetch_sample.py
├── requirements.txt
├── .env                    # HF_TOKEN (gitignored)
└── README.md
```

---

## Notes

- **First run** — Whisper downloads the `base.en` model (~140 MB) on first use. It is cached locally after that.
- **Transcription accuracy** — `base.en` is fast and works well for clear English speech. For noisier recordings or other languages, swap to a larger Whisper model by editing the `whisper.load_model("base.en")` call in `app.py`.
- **Long meetings** — very long transcripts may approach the context limit of smaller Ollama models. Switching to a model with a larger context window (e.g. `llama3.1:70b` if your hardware supports it) improves results for hour-long recordings.
- **MeetingBank video availability** — Seattle City Council meetings are served from `video.seattle.gov` and download reliably. The Granicus CDN links for all other cities are no longer active.
