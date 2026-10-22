"""Qwen3-TTS engine and voice transcripts, without torch or qwen-tts installed."""

from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.db import Database
from app.engines import create_engine
from app.engines.base import SynthesisParams
from app.engines.chatterbox import ChatterboxEngine
from app.engines.fake import FakeEngine
from app.engines.qwen import QwenEngine
from tests.conftest import make_wav, upload_voice, wait_for_job


class FakeQwenModel:
    prompts: list[dict] = []
    generations: list[dict] = []

    @classmethod
    def from_pretrained(cls, source, device_map=None, dtype=None):
        cls.loaded_from = (source, device_map)
        return cls()

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        FakeQwenModel.prompts.append({"ref_audio": ref_audio, "ref_text": ref_text, "xvec": x_vector_only_mode})
        return f"prompt-{len(FakeQwenModel.prompts)}"

    def generate_voice_clone(self, **kwargs):
        FakeQwenModel.generations.append(kwargs)
        return [np.zeros(12000, dtype=np.float32)], 12000


@pytest.fixture
def fake_qwen(monkeypatch):
    FakeQwenModel.prompts, FakeQwenModel.generations = [], []
    torch = types.ModuleType("torch")
    torch.float32, torch.bfloat16 = "float32", "bfloat16"
    torch.manual_seed = lambda seed: None
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    torch.inference_mode = _NoGrad
    qwen_tts = types.ModuleType("qwen_tts")
    qwen_tts.Qwen3TTSModel = FakeQwenModel
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo_id: f"/hf-cache/{repo_id}"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_tts)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return FakeQwenModel


def test_registry_builds_qwen(tmp_path: Path):
    engine = create_engine(Settings(data_dir=tmp_path, engine="qwen", qwen_model="/models/qwen-base"))
    assert isinstance(engine, QwenEngine)
    assert engine.model_name == "/models/qwen-base"
    assert "fr" in engine.languages and "ar" not in engine.languages


def test_only_chatterbox_multilingual_switches_t3():
    assert ChatterboxEngine("multilingual").switchable_t3
    assert not ChatterboxEngine("english").switchable_t3
    assert FakeEngine().switchable_t3
    assert not QwenEngine().switchable_t3


def test_qwen_clones_with_transcript_and_caches_prompt(fake_qwen, tmp_path: Path):
    ref = make_wav(tmp_path / "ref.wav")
    engine = QwenEngine(device="cpu")
    params = SynthesisParams(language="fr", temperature=0.9, seed=3, reference_text="  Bon, je vais vous expliquer. ")
    audio = engine.synthesize("Un mégawatt, c'est mille kilowatts.", ref, params)
    engine.synthesize("Deuxième phrase.", ref, params)

    assert fake_qwen.loaded_from == ("/hf-cache/Qwen/Qwen3-TTS-12Hz-1.7B-Base", "cpu")
    assert fake_qwen.prompts == [{"ref_audio": str(ref), "ref_text": "Bon, je vais vous expliquer.", "xvec": False}]
    first = fake_qwen.generations[0]
    assert first["language"] == "French" and first["voice_clone_prompt"] == "prompt-1"
    assert first["temperature"] == first["subtalker_temperature"] == 0.9
    assert first["max_new_tokens"] > 40
    # 12 kHz codec output is resampled to the engine's 24 kHz.
    assert audio.dtype == np.float32 and len(audio) == 24000


def test_qwen_without_transcript_clones_timbre_only(fake_qwen, tmp_path: Path):
    engine = QwenEngine(device="cpu")
    engine.synthesize("Hello.", make_wav(tmp_path / "ref.wav"), SynthesisParams(language="en"))
    assert fake_qwen.prompts[0]["xvec"] is True and fake_qwen.prompts[0]["ref_text"] is None


def test_qwen_rejects_unsupported_language(fake_qwen, tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported language"):
        QwenEngine(device="cpu").synthesize("Hej.", make_wav(tmp_path / "ref.wav"), SynthesisParams(language="sv"))


def test_voice_transcript_is_stored_edited_and_used(client, wav_file, monkeypatch):
    seen: list[str | None] = []
    original = FakeEngine.synthesize

    def spy(self, text, reference_wav, params):
        seen.append(params.reference_text)
        return original(self, text, reference_wav, params)

    monkeypatch.setattr(FakeEngine, "synthesize", spy)
    with wav_file.open("rb") as fh:
        voice = client.post(
            "/api/voices",
            files={"file": ("me.wav", fh, "audio/wav")},
            data={"name": "Me", "language": "fr", "transcript": " Bon,\n je vais   vous expliquer. "},
        ).json()
    assert voice["transcript"] == "Bon, je vais vous expliquer."

    edited = client.patch(f"/api/voices/{voice['id']}", json={"transcript": "Bon, on y va."}).json()
    assert edited["transcript"] == "Bon, on y va."
    job = client.post("/api/generate", json={"voice_id": voice["id"], "text": "Allô."}).json()
    assert wait_for_job(client, job["id"])["status"] == "done"
    assert seen == ["Bon, on y va."]

    assert client.patch(f"/api/voices/{voice['id']}", json={"transcript": ""}).json()["transcript"] is None
    assert upload_voice(client, wav_file)["transcript"] is None


def test_transcript_rejected_when_clip_is_cut(client, settings, tmp_path):
    long_clip = make_wav(tmp_path / "long.wav", seconds=settings.max_reference_seconds + 2)
    with long_clip.open("rb") as fh:
        res = client.post(
            "/api/voices", files={"file": ("long.wav", fh, "audio/wav")}, data={"transcript": "Some words."}
        )
    assert res.status_code == 400 and "MAX_REFERENCE_SECONDS" in res.json()["detail"]
    assert client.get("/api/voices").json() == []


def test_old_database_gains_transcript_column(tmp_path: Path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE voices (id TEXT PRIMARY KEY, name TEXT NOT NULL, language TEXT, filename TEXT NOT NULL,"
            " duration REAL, created_at REAL NOT NULL)"
        )
        conn.execute("INSERT INTO voices VALUES ('abc', 'Old', 'en', 'abc.wav', 3.0, 1.0)")
    db = Database(path)
    assert db.get_voice("abc")["transcript"] is None
    assert db.update_voice("abc", transcript="Hello there.")["transcript"] == "Hello there."
