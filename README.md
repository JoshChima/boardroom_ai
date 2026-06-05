# Boardroom — Local Meeting Intelligence

A fully local meeting intelligence tool that transcribes, summarizes, and answers questions about your meetings — no cloud, no API keys, no data leaving your machine.

Pass in a pre-recorded audio file or let it record the meeting live from your microphone. It transcribes the audio with timestamps, generates a summary, and then answers specific questions about the meeting content, telling you whether each question was addressed and where in the transcript to find it.

---

## Acknowledgements

This project was built on the foundation laid by **Duy Huynh**, whose guide on assembling a local voice assistant with Whisper, Ollama, and Bark was the starting point for this project. The meeting intelligence features in this repo grew out of that work.

[Build Your Own Voice Assistant and Run it Locally — Duy Huynh](https://medium.com/@vndee.huynh/build-your-own-voice-assistant-and-run-it-locally-whisper-ollama-bark-c80e6f815cba)

---

## Key Features

- **Timestamped transcription** — every segment of speech is labelled with its start time so you can jump directly to the relevant moment.
- **Automatic meeting summary** — covers main topics, key decisions, and action items.
- **Q&A over the transcript** — provide questions in a markdown file or type them directly in the terminal; the LLM tells you whether each was addressed and summarises the answer.
- **Two audio input modes** — pass a pre-recorded file (wav, mp3, m4a, and more) or record straight from your microphone.
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

---

## Prerequisites

### 1. Python

Python **3.9 – 3.11** is required. You can check your version with:

```bash
python --version
```

### 2. Ollama

Ollama runs the language model locally. Install it from [ollama.com](https://ollama.com).

Start the Ollama server if it isn't already running:

```bash
ollama serve
```

The app automatically checks whether your selected model is available locally when it starts. If it isn't, it pulls it for you — no manual download step required. The default model is `llama3.1`; pass `--model <name>` to use any other Ollama model (e.g. `--model mistral`).

### 3. System audio libraries

These are needed by `sounddevice` for microphone recording.

| OS | Steps |
|---|---|
| **Windows** | No extra step — sounddevice ships with prebuilt PortAudio binaries. |
| **macOS** | `brew install portaudio` |
| **Debian / Ubuntu** | `sudo apt-get install portaudio19-dev` |

### 4. NVIDIA GPU (optional but recommended)

If you have an NVIDIA GPU, install the CUDA-enabled build of PyTorch following the instructions at [pytorch.org/get-started](https://pytorch.org/get-started/locally/). The app detects CUDA automatically and uses fp16 for faster transcription — no configuration needed.

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd src

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Pass a pre-recorded audio file

```bash
python app.py meeting.wav
```

Supported formats include wav, mp3, m4a, flac, ogg, and any format FFmpeg can read.

### Record live from the microphone

Run the app with no audio argument and it will prompt you to record:

```bash
python app.py
```

```
No audio file provided — switching to microphone recording.

Press Enter to start recording…
Recording… press Enter to stop.
```

Once you stop recording, the audio is passed through the same transcription, summarisation, and Q&A pipeline as a file would be.

### Pass a questions file

Create a markdown file with your questions as a bullet or numbered list (see format below) and pass it with `--questions`:

```bash
python app.py meeting.wav --questions questions.md
```

### Enter questions interactively

If you do not pass `--questions`, the app will prompt you to type questions in the terminal after the summary is displayed:

```
Enter your questions, one per line.
Press Enter on a blank line when done.

> Was the project deadline confirmed?
> Who is responsible for the client follow-up?
>
```

### Use a different Ollama model

```bash
python app.py meeting.wav --model mistral
```

---

## Questions File Format

The questions file is a plain markdown document. Both bullet lists and numbered lists are supported — you can mix them freely. Headings and blank lines are ignored.

**Example `questions.md`:**

```markdown
# Sprint Retrospective — Questions

- Was the release date moved or confirmed?
- Were any blockers raised that are still unresolved?
- Who owns the follow-up with the design team?

4. Was budget discussed?
5. Are there any action items assigned to engineering?
```

Each line that starts with `-`, `*`, or a number followed by `.` or `)` is treated as a question.

---

## Example Output

```
────────────────── Transcription ──────────────────
[00:08] Alright, let's get started. First item is the Q3 roadmap.
[00:45] Sarah confirmed the release is still on track for the 14th.
[01:12] The design review is blocked — we're waiting on assets from the client.
...

────────────────── Meeting Summary ──────────────────
Main topics discussed:
- Q3 roadmap and release schedule
- Design review blocker pending client assets
- Budget approval for contractor headcount

Key decisions:
- Release date confirmed as the 14th.

Action items:
- James to chase the client for assets by end of week.

────────────────────── Q&A ──────────────────────────

Q1: Was the release date confirmed?
Addressed: Yes, at [00:45] Sarah confirmed the release remains on track for the 14th.

Q2: Are there any unresolved blockers?
Addressed: Yes, at [01:12] a design review blocker was raised — the team is waiting on assets from the client. James was assigned to follow up.

Q3: Was headcount budget approved?
Addressed: Yes, budget for contractor headcount was approved during the meeting.
```

---

## Project Structure

```
boardroom_ai/
├── src/
│   ├── app.py            # Main application — transcription, summarisation, Q&A
│   ├── tts.py            # Deprecated Phase 1 Bark TTS service (kept for reference)
│   └── requirements.txt
└── README.md
```

---

## Notes

- **First run** — Whisper downloads the `base.en` model (~140 MB) on first use. It is cached locally after that.
- **Transcription accuracy** — `base.en` is fast and works well for clear English speech. For noisier recordings or other languages, swap to a larger Whisper model by editing the `whisper.load_model("base.en")` call in `app.py`.
- **Long meetings** — very long transcripts may approach the context limit of smaller Ollama models. Switching to a model with a larger context window (e.g. `llama3.1:70b` if your hardware supports it) improves results for hour-long recordings.
