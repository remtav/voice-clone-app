from __future__ import annotations

from pathlib import Path

from app.config import Settings


def test_t3_model_defaults_to_official_v3(monkeypatch):
    monkeypatch.delenv("CHATTERBOX_T3_MODEL", raising=False)
    settings = Settings.from_env()
    assert settings.chatterbox_t3_model == "v3"
    assert settings.chatterbox_t3_path is None


def test_t3_model_blank_falls_back_to_v3():
    assert Settings(chatterbox_t3_model="  ").chatterbox_t3_model == "v3"


def test_official_t3_names_have_no_path():
    assert Settings(chatterbox_t3_model="v2").chatterbox_t3_path is None


def test_relative_t3_file_resolves_under_data_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CHATTERBOX_T3_MODEL", " models/t3_fr_ca.safetensors ")
    settings = Settings.from_env()
    assert settings.chatterbox_t3_model == "models/t3_fr_ca.safetensors"
    assert settings.chatterbox_t3_path == tmp_path / "models" / "t3_fr_ca.safetensors"


def test_absolute_t3_file_is_kept(tmp_path: Path):
    target = tmp_path / "Custom_T3.safetensors"
    settings = Settings(data_dir=tmp_path / "data", chatterbox_t3_model=str(target))
    assert settings.chatterbox_t3_path == target
