"""Phase 5: model card generation and its personal-data guard."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.finetune import model_card


def run_dir(tmp_path: Path, base_t3: str = "v3", own_rows: bool = False) -> Path:
    data = tmp_path / "audio_data"
    data.mkdir()
    with (data / "metadata.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_name", "transcription"])
        writer.writerow(["audio/qc_a.wav", "texte"])
        if own_rows:
            writer.writerow(["audio/own_me_0001.wav", "ma voix"])
    run = tmp_path / "run"
    (run / "merged_model").mkdir(parents=True)
    manifest = {"config": {"data_dir": str(data), "base_t3": base_t3, "lora_rank": 16, "lora_alpha": 32.0,
                           "epochs": 4, "learning_rate": 2e-5},
                "upstream": {"repo": "owner/toolkit", "commit": "e9816b4e8292"}}
    (run / "toolkit.json").write_text(json.dumps(manifest))
    return run


def test_run_config_is_found_from_the_merged_checkpoint(tmp_path: Path):
    run = run_dir(tmp_path)
    assert model_card.run_config_for(run / "merged_model" / "t3_fr_ca.safetensors")["config"]["lora_rank"] == 16
    assert model_card.run_config_for(tmp_path / "elsewhere" / "x" / "t3.safetensors") == {}


def test_public_run_is_accepted(tmp_path: Path):
    model_card.ensure_public(json.loads((run_dir(tmp_path) / "toolkit.json").read_text()))
    model_card.ensure_public({})  # no provenance: nothing to object to


def test_own_recordings_are_refused(tmp_path: Path):
    config = json.loads((run_dir(tmp_path, own_rows=True) / "toolkit.json").read_text())
    with pytest.raises(model_card.PersonalDataError, match="own recordings"):
        model_card.ensure_public(config)


def test_second_stage_checkpoint_is_refused(tmp_path: Path):
    config = json.loads((run_dir(tmp_path, base_t3="/runs/fr_ca/t3_fr_ca.safetensors") / "toolkit.json").read_text())
    with pytest.raises(model_card.PersonalDataError, match="second-stage"):
        model_card.ensure_public(config)


def test_render_card(tmp_path: Path):
    config = json.loads((run_dir(tmp_path) / "toolkit.json").read_text())
    summary = {"tensors": 292, "dtypes": ["F32"], "text_emb": (2454, 1024), "speech_emb": (8194, 1024)}
    report = "# fr-CA fine-tune evaluation\n\n| a |\n\n## Go / no-go\n\n- **fr_ca @ cfg 0.8: PASS**\n"
    card = model_card.render_card("t3_fr_ca", "me/Chatterbox-fr-ca", "fr-CA", "fr", summary, "ab" * 32, 2143989752,
                                  config, report)
    assert card.startswith("---\nlicense: mit\n")
    assert 'hf_hub_download("me/Chatterbox-fr-ca", "t3_fr_ca.safetensors")' in card
    assert "LoRA rank 16, alpha 32.0" in card and "@ `e9816b4`" in card
    assert "### Go / no-go" in card and "# fr-CA fine-tune evaluation" not in card
    assert "Tensor count: `292`" in card and "Speech embedding shape: `(8194, 1024)`" in card
    unevaluated = model_card.render_card("t3", "r", "fr-CA", "fr", summary, "0", 1, {}, None)
    assert "_Not evaluated._" in unevaluated


def test_cli_refuses_personal_checkpoint(tmp_path: Path, capsys):
    run = run_dir(tmp_path, own_rows=True)
    checkpoint = run / "merged_model" / "t3_fr_ca_me.safetensors"
    checkpoint.write_bytes(b"x")
    assert model_card.main(["--checkpoint", str(checkpoint), "--out", str(tmp_path / "README.md")]) == 1
    assert "Keep this checkpoint private" in capsys.readouterr().err
    assert not (tmp_path / "README.md").exists()
