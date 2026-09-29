"""Trainer settings, read from the same environment (.env) as the app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass
class TrainerSettings:
    data_dir: Path = Path("data")
    poll_seconds: float = 3.0
    heartbeat_seconds: float = 10.0
    # A 3090 (24 GB) trains the 0.5B T3 with LoRA comfortably; below 16 GB free, wait.
    min_free_vram_gb: float = 16.0
    # How long a run waits for the app to release the GPU before failing.
    gpu_wait_seconds: float = 900.0
    # An app heartbeat older than this means the app is not running.
    app_stale_seconds: float = 30.0
    # CPU training takes days; only for tests and development.
    allow_cpu: bool = False

    @classmethod
    def from_env(cls) -> TrainerSettings:
        return cls(
            data_dir=Path(_env("DATA_DIR", "data")),
            poll_seconds=float(_env("TRAINER_POLL_SECONDS", "3")),
            min_free_vram_gb=float(_env("TRAINER_MIN_FREE_VRAM_GB", "16")),
            gpu_wait_seconds=float(_env("TRAINER_GPU_WAIT_SECONDS", "900")),
            allow_cpu=_env("TRAINER_ALLOW_CPU", "0").lower() in {"1", "true", "yes", "on"},
        )

    # Layout of the shared data volume ------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "voiceclone.sqlite3"

    @property
    def finetune_dir(self) -> Path:
        return self.data_dir / "finetune"

    @property
    def runs_dir(self) -> Path:
        return self.finetune_dir / "runs"

    @property
    def toolkit_src(self) -> Path:
        return self.finetune_dir / "toolkit-src"

    @property
    def qc_src(self) -> Path:
        return self.finetune_dir / "qc_src"

    @property
    def qc_data(self) -> Path:
        return self.finetune_dir / "qc" / "audio_data"

    @property
    def own_data(self) -> Path:
        return self.finetune_dir / "me" / "audio_data"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"
