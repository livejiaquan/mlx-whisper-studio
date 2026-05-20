"""CLI worker for MLX Whisper transcription outputs."""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Optional

import mlx_whisper


def parse_args():
    parser = argparse.ArgumentParser(description="MLX Whisper worker")
    parser.add_argument("--input", required=True, help="Path to audio/video file")
    parser.add_argument("--model", required=True, help="MLX model repo or local path")
    parser.add_argument(
        "--language",
        default="auto",
        help="Language code (e.g. zh) or auto",
    )
    parser.add_argument("--prompt", default=None, help="Initial prompt text")
    parser.add_argument(
        "--formats",
        default="txt",
        help="Comma-separated formats: txt,srt,vtt,json",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--output-base", default=None, help="Output base name")
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Transcribe or translate",
    )
    parser.add_argument(
        "--translate-to",
        default=None,
        help="Optional translation target language (e.g. zh)",
    )
    parser.add_argument(
        "--translation-model",
        default=None,
        help="Optional translation model override",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_formats(raw: str) -> List[str]:
    formats = [item.strip().lower() for item in raw.split(",") if item.strip()]
    allowed = {"txt", "srt", "vtt", "json"}
    invalid = [fmt for fmt in formats if fmt not in allowed]
    if invalid:
        raise ValueError(f"Unsupported formats: {', '.join(invalid)}")
    return formats


def format_timestamp(seconds: float, separator: str) -> str:
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    if secs >= 60:
        minutes += 1
        secs = 0
    if minutes >= 60:
        hours += 1
        minutes = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def segments_to_srt(segments: Iterable[dict]) -> str:
    lines = []
    for idx, segment in enumerate(segments, 1):
        start = format_timestamp(segment["start"], ",")
        end = format_timestamp(segment["end"], ",")
        text = segment["text"].strip()
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def segments_to_vtt(segments: Iterable[dict]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        start = format_timestamp(segment["start"], ".")
        end = format_timestamp(segment["end"], ".")
        text = segment["text"].strip()
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip() + "\n"


def write_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def write_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_output_paths(output_dir: str, base: str, formats: List[str]) -> dict:
    mapping = {}
    for fmt in formats:
        mapping[fmt] = os.path.join(output_dir, f"{base}.{fmt}")
    return mapping


def translate_segments(
    segments: Iterable[dict],
    target_language: str,
    model_name: Optional[str],
) -> Optional[List[str]]:
    if not target_language:
        return None
    default_models = {
        "zh": "Helsinki-NLP/opus-mt-ja-zh",
    }
    model_id = model_name or default_models.get(target_language)
    if not model_id:
        raise ValueError(f"No translation model for {target_language}")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device)
    model.eval()

    texts = [segment.get("text", "").strip() for segment in segments]
    translations: List[str] = []
    batch_size = 8
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=256)
        translations.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return translations


def main():
    args = parse_args()
    formats = normalize_formats(args.formats)
    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    base = args.output_base
    if not base:
        base = os.path.splitext(os.path.basename(args.input))[0]

    language = None if args.language == "auto" else args.language
    prompt = args.prompt.strip() if args.prompt else None

    print(f"[worker] Input: {args.input}", flush=True)
    print(f"[worker] Model: {args.model}", flush=True)
    print(f"[worker] Output dir: {output_dir}", flush=True)
    print(f"[worker] Formats: {', '.join(formats)}", flush=True)

    result = mlx_whisper.transcribe(
        args.input,
        path_or_hf_repo=args.model,
        language=language,
        initial_prompt=prompt,
        task=args.task,
        verbose=False,
    )

    translate_to = args.translate_to
    if translate_to in (None, "", "none"):
        translate_to = None

    translated_segments = None
    translation_error = None
    if translate_to:
        try:
            translated_segments = translate_segments(
                result.get("segments", []),
                translate_to,
                args.translation_model,
            )
        except Exception as exc:  # noqa: BLE001
            translation_error = str(exc)
            print(f"[worker] Translation failed: {translation_error}", flush=True)
            translated_segments = None
            translate_to = None

    outputs = build_output_paths(output_dir, base, formats)
    if "txt" in outputs:
        write_text(outputs["txt"], result.get("text", "").strip())
    if "srt" in outputs:
        write_text(outputs["srt"], segments_to_srt(result.get("segments", [])))
    if "vtt" in outputs:
        write_text(outputs["vtt"], segments_to_vtt(result.get("segments", [])))
    if "json" in outputs:
        payload = {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "language": result.get("language"),
            "model": args.model,
            "translation_error": translation_error,
        }
        write_json(outputs["json"], payload)

    if translate_to and translated_segments:
        translated_base = f"{base}_{translate_to}"
        translated_outputs = build_output_paths(output_dir, translated_base, formats)
        translated_text = "".join(translated_segments).strip()
        if "txt" in translated_outputs:
            write_text(translated_outputs["txt"], translated_text)
        if "srt" in translated_outputs:
            segments = result.get("segments", [])
            translated = [
                {**segment, "text": translated_segments[idx]}
                for idx, segment in enumerate(segments)
            ]
            write_text(translated_outputs["srt"], segments_to_srt(translated))
        if "vtt" in translated_outputs:
            segments = result.get("segments", [])
            translated = [
                {**segment, "text": translated_segments[idx]}
                for idx, segment in enumerate(segments)
            ]
            write_text(translated_outputs["vtt"], segments_to_vtt(translated))
        if "json" in translated_outputs:
            payload = {
                "text": translated_text,
                "segments": [
                    {**segment, "text": translated_segments[idx]}
                    for idx, segment in enumerate(result.get("segments", []))
                ],
                "language": translate_to,
                "model": args.translation_model,
            }
            write_json(translated_outputs["json"], payload)

    print("[worker] Done", flush=True)
    print(json.dumps({"outputs": outputs}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
