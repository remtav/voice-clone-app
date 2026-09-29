"""Contract shared by the web app and the trainer service (no heavy imports).

Both processes use the same SQLite database and data volume; these names are
how they find each other's state.
"""

from __future__ import annotations

from pathlib import Path

APP_HEARTBEAT_KEY = "app_heartbeat"          # {"t", "engine_loaded"}: written by the app's worker
TRAINER_HEARTBEAT_KEY = "trainer_heartbeat"  # {"t", "gpu", "busy", "allow_cpu"}: written by the trainer
ACTIVE_MODEL_KEY = "active_t3"               # {"model"}: the checkpoint chosen in the UI
RUN_LOG_NAME = "trainer.log"                 # in data/finetune/runs/<run id>/
RUN_METRICS_NAME = "training_metrics.png"    # written by the LoRA toolkit in the same directory

RECIPE_DEFAULTS = {
    # stage 1: public Quebec corpus on top of the official v3
    "fr_ca": {"epochs": 4, "lora_rank": 16, "learning_rate": 2e-5, "grad_accum": 8},
    # stage 2: your recordings on top of a fine-tuned Quebec checkpoint
    "personal": {"epochs": 2, "lora_rank": 16, "learning_rate": 1e-5, "grad_accum": 8},
}


def read_log_tail(run_dir: Path, lines: int = 200) -> list[str]:
    path = run_dir / RUN_LOG_NAME
    if not path.is_file():
        return []
    with path.open("rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 256 * 1024))
        return fh.read().decode("utf-8", "replace").splitlines()[-lines:]
