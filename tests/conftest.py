from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

ACTIVE = {"queued", "running", "cancelling"}


def make_wav(path: Path, seconds: float = 2.0, sr: int = 16000, channels: int = 1) -> Path:
    t = np.arange(int(sr * seconds)) / sr
    tone = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    sf.write(str(path), data, sr)
    return path


def wait_for_job(client: TestClient, job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ACTIVE:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        engine="fake",
        app_password="",
        preload_model=True,
        max_chunk_chars=60,
        max_text_chars=600,
        history_limit=50,
    )


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    return make_wav(tmp_path / "reference.wav")


def upload_voice(client: TestClient, path: Path, name: str = "Test voice", language: str = "en") -> dict:
    with path.open("rb") as fh:
        res = client.post(
            "/api/voices",
            files={"file": (path.name, fh, "audio/wav")},
            data={"name": name, "language": language},
        )
    assert res.status_code == 201, res.text
    return res.json()
