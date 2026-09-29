"""The two training recipes, as ordered pipeline steps.

``fr_ca``     stage 1: the public Quebec French corpus on top of the official v3.
``personal``  stage 2: your own clips (plus part of the corpus) on top of a
              fine-tuned checkpoint from ``data/models``.

Every step is either a command (run as a subprocess, output logged) or an
in-process action.  A step can be skipped when its output already exists.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.training import RECIPE_DEFAULTS
from trainer.config import TrainerSettings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Step:
    key: str
    label: str
    argv: list[str] | None = None
    action: Callable[[], None] | None = None
    skip: Callable[[], bool] | None = None
    training: bool = False


def count_rows(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def validation_split(n_clips: int) -> float:
    """5 %, but at least one validation clip (the toolkit divides by the validation batch count)."""
    if n_clips <= 1:
        return 0.5
    return min(0.5, max(0.05, 1.5 / n_clips))


def first_holdout_clip(audio_data: Path) -> Path | None:
    holdout = audio_data / "holdout.csv"
    if not holdout.is_file():
        return None
    with holdout.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            clip = audio_data / row["file_name"]
            if clip.is_file():
                return clip
    return None


def ensure_toolkit_source(cache: Path) -> Path:
    """Download the pinned toolkit once (SHA-256 checked) and reuse it for every run."""
    from scripts.finetune import setup_toolkit

    cache.mkdir(parents=True, exist_ok=True)
    for name, digest in setup_toolkit.UPSTREAM_SHA256.items():
        target = cache / name
        if target.is_file() and setup_toolkit.sha256(target.read_bytes()) == digest:
            continue
        target.write_text(setup_toolkit.fetch_upstream(name), encoding="utf-8")
    return cache


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    tmp.replace(dst)  # atomic: the app never sees a half-written checkpoint
    # rename() is a no-op when both names already point to the same inode (republishing a
    # hard-linked file): the temporary name would stay behind.
    tmp.unlink(missing_ok=True)


def plan(run: dict, settings: TrainerSettings, device: str) -> list[Step]:
    from scripts.finetune import setup_toolkit

    params = {**RECIPE_DEFAULTS[run["recipe"]], **run["params"]}
    run_dir = settings.runs_dir / run["id"]
    output = params["output_name"]
    merged = run_dir / "merged_model" / f"{output}.safetensors"
    python = sys.executable
    steps: list[Step] = []

    if run["recipe"] == "fr_ca":
        data_dir, base = settings.qc_data, "v3"
        have_data = (data_dir / "metadata.csv").is_file
        steps += [
            Step("download", "Downloading the Quebec corpus (1.8 GB)",
                 argv=[python, "-m", "scripts.finetune.prepare_qc_dataset", "download", "--dest", str(settings.qc_src)],
                 skip=have_data),
            Step("prepare", "Preparing the Quebec corpus",
                 argv=[python, "-m", "scripts.finetune.prepare_qc_dataset", "from-processed",
                       "--src", str(settings.qc_src / "dataset" / "processed"), "--out", str(data_dir)],
                 skip=have_data),
        ]
    else:
        data_dir, base = run_dir / "mix", str(settings.models_dir / params["base_model"])
        steps.append(Step("mix", "Mixing your clips with the Quebec corpus",
                          argv=[python, "-m", "scripts.finetune.build_personal_dataset", "--own",
                                str(settings.own_data), "--qc", str(settings.qc_data), "--out", str(data_dir)]))

    def setup() -> None:
        source = ensure_toolkit_source(settings.toolkit_src)
        rank = int(params["lora_rank"])
        cfg = setup_toolkit.RunConfig(
            data_dir=str(data_dir.absolute()), run_dir=str(run_dir.absolute()), base_t3=base, output_name=output,
            epochs=int(params["epochs"]), learning_rate=float(params["learning_rate"]), lora_rank=rank,
            lora_alpha=2.0 * rank, grad_accum=int(params["grad_accum"]),
            val_split=validation_split(count_rows(data_dir / "metadata.csv")),
        )
        setup_toolkit.setup(cfg, source)

    def publish() -> None:
        link_or_copy(merged, settings.models_dir / f"{output}.safetensors")
        (merged.parent / f"{output}.pt").unlink(missing_ok=True)  # same weights, 2 GB: not needed any more

    validate = [python, "-m", "scripts.finetune.validate_t3_checkpoint", str(merged), "--base", base, "--strict-load"]
    reference = first_holdout_clip(data_dir if run["recipe"] == "fr_ca" else settings.own_data)
    if reference is not None:
        validate += ["--smoke-reference", str(reference), "--smoke-out", str(run_dir / "smoke"), "--device", device]

    steps += [
        Step("toolkit", "Setting up the training toolkit", action=setup),
        Step("train", "Training", argv=[python, str(run_dir / "toolkit" / "lora.py")], training=True),
        Step("export", "Exporting the checkpoint", argv=[python, str(run_dir / "toolkit" / "fix_merged_model.py")]),
        Step("validate", "Validating the checkpoint", argv=validate),
        Step("publish", "Publishing to data/models", action=publish),
    ]
    return steps
