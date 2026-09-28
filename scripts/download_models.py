"""Download and load the configured engine once, so the first request is fast.

Usage (inside the container or a venv with the engine deps installed):

    python -m scripts.download_models
"""

from __future__ import annotations

import json
import logging
import time

from app.config import Settings
from app.engines import create_engine


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    engine = create_engine(settings)
    started = time.time()
    engine.load()
    print(json.dumps(engine.info(), indent=2))
    print(f"Loaded in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
