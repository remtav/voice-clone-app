"""ChatterboxEngine checkpoint selection, without torch or chatterbox installed."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from app.config import Settings
from app.engines import create_engine
from app.engines.chatterbox import SHARED_MULTILINGUAL_ASSETS, ChatterboxEngine


class FakeMultilingualTTS:
    calls: list[tuple] = []

    @classmethod
    def from_pretrained(cls, device, t3_model=None):
        cls.calls.append(("from_pretrained", device, t3_model))
        return cls()

    @classmethod
    def from_local(cls, ckpt_dir, device, t3_model=None):
        cls.calls.append(("from_local", ckpt_dir, device, t3_model))
        return cls()


@pytest.fixture
def fake_chatterbox(monkeypatch):
    FakeMultilingualTTS.calls = []
    downloads: list[dict] = []

    def snapshot_download(**kwargs):
        downloads.append(kwargs)
        return "/hf-cache/chatterbox"

    package = types.ModuleType("chatterbox")
    mtl_tts = types.ModuleType("chatterbox.mtl_tts")
    mtl_tts.REPO_ID = "ResembleAI/chatterbox"
    mtl_tts.ChatterboxMultilingualTTS = FakeMultilingualTTS
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "chatterbox", package)
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mtl_tts)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return downloads


def test_registry_passes_t3_settings(tmp_path: Path):
    settings = Settings(data_dir=tmp_path, chatterbox_t3_model="models/t3_fr_ca.safetensors")
    engine = create_engine(settings)
    assert isinstance(engine, ChatterboxEngine)
    assert engine.t3_model == "models/t3_fr_ca.safetensors"
    assert engine.t3_path == tmp_path / "models" / "t3_fr_ca.safetensors"


def test_custom_t3_requires_multilingual(tmp_path: Path):
    with pytest.raises(ValueError, match="multilingual"):
        ChatterboxEngine(variant="turbo", t3_path=tmp_path / "t3.safetensors")


def test_info_reports_t3_checkpoint(tmp_path: Path):
    assert ChatterboxEngine(t3_model="v2").info()["t3_model"] == "v2"
    custom = ChatterboxEngine(t3_model="x/t3_fr_ca.safetensors", t3_path=tmp_path / "t3_fr_ca.safetensors")
    assert custom.info()["t3_model"] == "t3_fr_ca.safetensors"
    assert "t3_model" not in ChatterboxEngine(variant="english").info()


def test_official_checkpoint_is_selected_by_name(fake_chatterbox):
    engine = ChatterboxEngine(t3_model="v2")
    engine.device = "cpu"
    engine._load_official_multilingual()
    assert FakeMultilingualTTS.calls == [("from_pretrained", "cpu", "v2")]
    assert engine.model_id == "ResembleAI/chatterbox (multilingual v2)"
    assert fake_chatterbox == []


def test_custom_checkpoint_loads_shared_assets_and_absolute_t3(fake_chatterbox, tmp_path: Path):
    t3 = tmp_path / "models" / "t3_fr_ca.safetensors"
    t3.parent.mkdir()
    t3.write_bytes(b"weights")
    engine = ChatterboxEngine(t3_model="models/t3_fr_ca.safetensors", t3_path=t3)
    engine.device = "cpu"
    engine._load_custom_t3()

    assert len(fake_chatterbox) == 1
    assert fake_chatterbox[0]["repo_id"] == "ResembleAI/chatterbox"
    assert fake_chatterbox[0]["allow_patterns"] == SHARED_MULTILINGUAL_ASSETS
    assert not any("t3" in name for name in SHARED_MULTILINGUAL_ASSETS)
    assert FakeMultilingualTTS.calls == [("from_local", "/hf-cache/chatterbox", "cpu", str(t3.absolute()))]
    assert engine.model_id.endswith("custom T3 t3_fr_ca.safetensors)")


def test_symlinked_checkpoint_keeps_its_safetensors_name(fake_chatterbox, tmp_path: Path):
    # Hugging Face cache blobs are content hashes without a suffix; chatterbox's
    # from_local rejects T3 names that do not end in ".safetensors".
    blob = tmp_path / "blobs" / "dcf1bc111ebd"
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    link = tmp_path / "t3_fr_ca.safetensors"
    link.symlink_to(blob)
    engine = ChatterboxEngine(t3_model="t3_fr_ca.safetensors", t3_path=link)
    engine.device = "cpu"
    engine._load_custom_t3()
    assert FakeMultilingualTTS.calls[-1][3].endswith("t3_fr_ca.safetensors")


def test_missing_custom_checkpoint_fails_clearly(fake_chatterbox, tmp_path: Path):
    engine = ChatterboxEngine(t3_model="t3.safetensors", t3_path=tmp_path / "missing.safetensors")
    engine.device = "cpu"
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        engine._load_custom_t3()
    assert fake_chatterbox == []
