"""Entry point of the trainer container: ``python -m trainer``."""

from __future__ import annotations

import logging
import signal

from app.db import Database
from trainer.config import TrainerSettings
from trainer.service import Trainer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = TrainerSettings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(Database(settings.db_path), settings)
    # docker stop sends SIGTERM: stop the current run cleanly (it is marked failed, it cannot resume).
    signal.signal(signal.SIGTERM, lambda *_: trainer.stopping.set())
    signal.signal(signal.SIGINT, lambda *_: trainer.stopping.set())
    trainer.run_forever()


if __name__ == "__main__":
    main()
