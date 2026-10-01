"""Keep each transcription's temporary files separate from existing output."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from typing import List


@dataclass
class OutputPlan:
    output_dir: str
    staging_dir: str
    base: str
    formats: List[str]
    created_dir: bool


def _discard_reserved(reserved) -> None:
    for destination, handle in reserved:
        # A buffered close can itself fail (for example on a full disk). Keep
        # trying to remove all files created by this attempt in that case.
        with suppress(OSError):
            handle.close()
        with suppress(OSError):
            destination.unlink()


def prepare_outputs(file_path: str, formats: List[str], output_root: str) -> OutputPlan:
    base = Path(file_path).stem
    root = Path(output_root or Path(file_path).parent).expanduser().absolute()
    destination = root / f"{base}_outputs" if len(formats) > 1 else root
    created_dir = not destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".mlx-whisper-", dir=destination)
    return OutputPlan(str(destination), staging, base, list(formats), created_dir)


def publish_outputs(plan: OutputPlan) -> List[str]:
    """Publish a complete successful run without ever overwriting an old file.

    Exclusive creation reserves an entire output set before copying. If another
    run owns a name, retry with one shared suffix (including translated files).
    Only files created by this attempt can be removed if publication fails.
    """
    staging = Path(plan.staging_dir)
    sources = sorted(staging.iterdir())
    required = {f"{plan.base}.{fmt}" for fmt in plan.formats}
    if not required.issubset({path.name for path in sources}):
        raise RuntimeError("Transcription finished without all requested output files")
    if any(path.is_symlink() or not path.is_file() for path in sources):
        raise RuntimeError("Unexpected transcription output")
    if any(not path.name.startswith(plan.base) for path in sources):
        raise RuntimeError("Unexpected transcription output name")

    attempt = 1
    while True:
        suffix = "" if attempt == 1 else f"_{attempt}"
        paths = [
            Path(plan.output_dir) / f"{plan.base}{suffix}{source.name[len(plan.base):]}"
            for source in sources
        ]
        reserved = []
        try:
            for destination in paths:
                reserved.append((destination, destination.open("xb")))
        except FileExistsError:
            _discard_reserved(reserved)
            attempt += 1
            continue
        except BaseException:
            _discard_reserved(reserved)
            raise

        try:
            for source, (_, handle) in zip(sources, reserved):
                with source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, handle)
                handle.close()
        except BaseException:
            _discard_reserved(reserved)
            raise
        return [str(path) for path in paths]


def cleanup_outputs(plan: OutputPlan | None) -> None:
    """Remove only this run's private staging directory, never final outputs."""
    if plan is None:
        return
    shutil.rmtree(plan.staging_dir)
    if plan.created_dir:
        try:
            os.rmdir(plan.output_dir)
        except OSError:
            # A successful run or another writer may now own files here.
            pass
