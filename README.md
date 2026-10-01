# MLX Whisper Studio

A desktop GUI for transcribing audio and video on Apple Silicon. It runs
Whisper through Apple's [MLX](https://github.com/ml-explore/mlx) framework, so
transcription uses the Mac GPU instead of falling back to CPU.

Drop in a file (or a batch of them), pick a model and language, and get back
subtitles in TXT, SRT, VTT, or JSON.

## Features

- MLX Whisper backend — runs on the Apple Silicon GPU
- Drag-and-drop, with a queue for transcribing multiple files in one run
- Audio and video input (video audio tracks are decoded via `ffmpeg`)
- Output formats: TXT, SRT, VTT, JSON
- Model selection from Tiny up to Large v3, or a custom Hugging Face repo / local path
- Language auto-detection, or pin a specific language
- Initial prompt support to improve domain or mixed-language accuracy
- Optional offline translation of the transcript

## Requirements

- macOS on Apple Silicon (M1 or newer)
- Python 3.10+
- `ffmpeg` — `brew install ffmpeg`

## Setup

```bash
./setup.sh
```

This creates a `venv/` and installs the dependencies from `requirements.txt`.

## Run

```bash
./run.sh
```

## Usage notes

- Output is written next to the input file by default.
- When more than one output format is selected, files are saved in a subfolder
  named `<input_basename>_outputs`.
- Each GUI job writes into a private temporary folder first. Failed or cancelled
  jobs remove only their own temporary files, preserving earlier transcripts.
- Successful GUI jobs never overwrite existing outputs: a repeated name gets
  `_2`, `_3`, etc. The same suffix is used for the whole set, including translations.
- Cancel waits for the worker to stop before cleaning up. Settings are captured
  when you start the queue, so edits during a run apply to the next run.
- Models download on first use and are cached afterward.
- "Custom" language accepts any Whisper language code (e.g. `zh`, `ja`).
- "Custom model" accepts a Hugging Face repo ID or a local model path.
- Offline translation uses Hugging Face models (default
  `Helsinki-NLP/opus-mt-ja-zh`) and writes `*_zh` files for each selected format.

## Project layout

- `app.py` — Tkinter GUI: file queue, drag-and-drop, settings, log output
- `mlx_worker.py` — CLI worker that runs MLX Whisper and writes the output files
- `output_files.py` — staged output publication and existing-file protection
- `setup.sh` / `run.sh` — environment setup and launch helpers

## Development checks

Output safety and queue lifecycle tests use only Python's standard library
(including Tkinter). They do not open a GUI, run MLX, or download models:

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py mlx_worker.py output_files.py tests
```

Actual speech recognition and native GUI interaction still need an Apple Silicon
Mac with the application dependencies installed.
