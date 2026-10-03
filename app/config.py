"""Runtime configuration, read from environment variables.

Every setting has a sensible default so the app runs with no configuration
at all (using the local ``data/`` directory).  See ``.env.example`` for the
full list of variables.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass
class Settings:
    # Storage
    data_dir: Path = Path("data")

    # Engine selection
    engine: str = "chatterbox"  # "chatterbox" or "fake" (for tests / CPU dev)
    chatterbox_model: str = "multilingual"  # "multilingual", "english" or "turbo"
    # Multilingual T3 checkpoint: "v2", "v3", or a fine-tuned ``.safetensors`` file
    # (absolute, or relative to ``data_dir``), e.g. "models/t3_fr_ca.safetensors".
    chatterbox_t3_model: str = "v3"
    device: str = "auto"  # "auto", "cuda" or "cpu"
    preload_model: bool = True

    # Authentication (empty password disables auth: local use only!)
    app_password: str = ""
    secret_key: str = ""
    session_hours: int = 24 * 7

    # Limits
    max_text_chars: int = 5000
    max_chunk_chars: int = 300
    max_upload_mb: int = 50
    max_reference_seconds: int = 30
    min_reference_seconds: float = 1.0
    history_limit: int = 200

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.engine = self.engine.lower().strip()
        self.chatterbox_model = self.chatterbox_model.lower().strip()
        # Not lowercased: it may be a file path.
        self.chatterbox_t3_model = self.chatterbox_t3_model.strip() or "v3"
        if not self.secret_key:
            # Derive a stable secret so sessions survive restarts even when no
            # explicit SECRET_KEY is configured.  Anyone who knows the password
            # can log in anyway, so deriving from it does not weaken security.
            seed = self.app_password or "voice-clone-app-no-password"
            self.secret_key = hashlib.sha256(f"voice-clone-app:{seed}".encode()).hexdigest()

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=Path(os.environ.get("DATA_DIR", "data")),
            engine=os.environ.get("TTS_ENGINE", "chatterbox"),
            chatterbox_model=os.environ.get("CHATTERBOX_MODEL", "multilingual"),
            chatterbox_t3_model=os.environ.get("CHATTERBOX_T3_MODEL", "v3"),
            device=os.environ.get("DEVICE", "auto"),
            preload_model=_bool("PRELOAD_MODEL", True),
            app_password=os.environ.get("APP_PASSWORD", ""),
            secret_key=os.environ.get("SECRET_KEY", ""),
            session_hours=_int("SESSION_HOURS", 24 * 7),
            max_text_chars=_int("MAX_TEXT_CHARS", 5000),
            max_chunk_chars=_int("MAX_CHUNK_CHARS", 300),
            max_upload_mb=_int("MAX_UPLOAD_MB", 50),
            max_reference_seconds=_int("MAX_REFERENCE_SECONDS", 30),
            min_reference_seconds=_float("MIN_REFERENCE_SECONDS", 1.0),
            history_limit=_int("HISTORY_LIMIT", 200),
        )

    # Derived paths -----------------------------------------------------
    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chatterbox_t3_path(self) -> Path | None:
        """Resolved path of a fine-tuned T3 checkpoint, or None for an official one (v2/v3)."""
        value = self.chatterbox_t3_model
        if not value.endswith(".safetensors"):
            return None
        path = Path(value)
        return path if path.is_absolute() else self.data_dir / path

    @property
    def models_dir(self) -> Path:
        """Fine-tuned T3 checkpoints the app can switch to (written by the trainer)."""
        return self.data_dir / "models"

    @property
    def finetune_dir(self) -> Path:
        return self.data_dir / "finetune"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "voiceclone.sqlite3"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.voices_dir, self.outputs_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)
