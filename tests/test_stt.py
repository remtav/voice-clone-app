"""Speech-to-text drafts of reference transcripts, without faster-whisper installed."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.stt import FakeTranscriber, FasterWhisperTranscriber, create_transcriber
from tests.conftest import make_wav, upload_voice


def post_clip(client: TestClient, path: Path, language: str = "fr"):
    with path.open("rb") as fh:
        return client.post("/api/transcribe", files={"file": (path.name, fh, "audio/wav")},
                           data={"language": language})


class FakeWhisperModel:
    instances: list[FakeWhisperModel] = []
    fail_on: set[str] = set()

    def __init__(self, model, device="auto", compute_type="auto"):
        self.model, self.device = model, device
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, language=None, **kwargs):
        if self.device in FakeWhisperModel.fail_on:
            raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
        segments = [types.SimpleNamespace(text=" Bon, je vais"), types.SimpleNamespace(text="  vous expliquer. ")]
        return iter(segments), types.SimpleNamespace(language=language or "fr")


@pytest.fixture
def fake_whisper(monkeypatch):
    FakeWhisperModel.instances, FakeWhisperModel.fail_on = [], set()
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return FakeWhisperModel


def test_registry(tmp_path: Path):
    assert create_transcriber(Settings(data_dir=tmp_path, stt_engine="none")) is None
    assert isinstance(create_transcriber(Settings(data_dir=tmp_path, stt_engine="fake")), FakeTranscriber)
    stt = create_transcriber(Settings(data_dir=tmp_path, stt_model="ele-sage/whisper-qc-ct2", stt_device="cpu"))
    assert isinstance(stt, FasterWhisperTranscriber)
    assert (stt.model, stt.device) == ("ele-sage/whisper-qc-ct2", "cpu")
    with pytest.raises(ValueError):
        create_transcriber(Settings(data_dir=tmp_path, stt_engine="sphinx"))


def test_faster_whisper_joins_segments_and_loads_once(fake_whisper, tmp_path: Path):
    stt = FasterWhisperTranscriber("large-v3-turbo", device="cuda")
    result = stt.transcribe(tmp_path / "a.wav", "fr")
    assert (result.text, result.language) == ("Bon, je vais vous expliquer.", "fr")
    stt.transcribe(tmp_path / "a.wav")
    assert len(fake_whisper.instances) == 1 and stt.loaded
    stt.unload()
    assert not stt.loaded


def test_faster_whisper_falls_back_to_cpu(fake_whisper, tmp_path: Path):
    fake_whisper.fail_on = {"cuda"}
    stt = FasterWhisperTranscriber("large-v3-turbo", device="cuda")
    assert stt.transcribe(tmp_path / "a.wav", "fr").text == "Bon, je vais vous expliquer."
    assert [m.device for m in fake_whisper.instances] == ["cuda", "cpu"]
    assert stt.device == "cpu"


def test_transcribe_endpoint_drafts_without_saving(client, settings, wav_file):
    assert client.get("/api/config").json()["stt"]["available"]
    res = post_clip(client, wav_file)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["text"] == FakeTranscriber.TEXT
    assert body["language"] == "fr" and not body["truncated"] and body["duration"] == pytest.approx(2.0, abs=0.05)
    assert client.get("/api/voices").json() == []
    assert list(settings.uploads_dir.iterdir()) == []  # temporary files are removed


def test_transcribe_endpoint_flags_cut_clips(client, settings, tmp_path):
    res = post_clip(client, make_wav(tmp_path / "long.wav", seconds=settings.max_reference_seconds + 3))
    assert res.status_code == 200 and res.json()["truncated"]


def test_transcribe_rejects_bad_uploads(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with bad.open("rb") as fh:
        assert client.post("/api/transcribe", files={"file": ("notes.txt", fh, "text/plain")}).status_code == 400


def test_transcribe_saved_voice(client, wav_file):
    voice = upload_voice(client, wav_file)
    assert voice["transcript"] is None
    res = client.post(f"/api/voices/{voice['id']}/transcribe")
    assert res.status_code == 200, res.text
    assert res.json()["transcript"] == FakeTranscriber.TEXT
    assert client.get(f"/api/voices/{voice['id']}").json()["transcript"] == FakeTranscriber.TEXT
    assert client.post("/api/voices/missing/transcribe").status_code == 404


def test_transcribe_refused_during_training(client, wav_file, monkeypatch):
    monkeypatch.setattr(client.app.state.db, "active_run", lambda: {"id": "run"})
    assert post_clip(client, wav_file).status_code == 409


def test_stt_disabled(tmp_path: Path, wav_file):
    settings = Settings(data_dir=tmp_path / "data", engine="fake", stt_engine="none", preload_model=False)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/config").json()["stt"] is None
        assert post_clip(client, wav_file).status_code == 503


def test_worker_unloads_stt_for_training(client, wav_file, monkeypatch):
    post_clip(client, wav_file)
    stt = client.app.state.transcriber
    assert stt.loaded
    monkeypatch.setattr(client.app.state.db, "active_run", lambda: {"id": "run"})
    client.app.state.worker._housekeeping()
    assert not stt.loaded
