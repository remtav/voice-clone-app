"""Phase 2 tooling: toolkit patching, LoRA merge and checkpoint validation (no torch needed)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.finetune import merge_adapter, setup_toolkit
from scripts.finetune.t3_state import PROJECTION_RE
from scripts.finetune.validate_t3_checkpoint import compare_state, speaking_rate_problem

# Every snippet setup_toolkit rewrites, laid out as in the pinned upstream lora.py.
FAKE_LORA = '''import random
from pathlib import Path

AUDIO_DATA_DIR = "./audio_data"
BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
MAX_AUDIO_LENGTH = 400.0
MIN_AUDIO_LENGTH = 1.0
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
GRADIENT_ACCUMULATION_STEPS = 8
SAVE_EVERY_N_STEPS = 200
CHECKPOINT_DIR = "checkpoints_lora"
DEVICE = "cpu"
MAX_TEXT_LENGTH = 1000
VALIDATION_SPLIT = 0.1


class AudioSample:
    language_id: str = "ar"


def main():
    metrics_tracker = MetricsTracker(save_path="training_metrics.png", update_interval=2.0)
    random.shuffle(samples)
    for i in range(batch_size):
        tokens, _ = s3_tokzr.forward([audio_16k])
        target_tokens_list.append(tokens)
    model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    sample = dict(
                language_id='ar'  # Arabic language ID
    )
    merged_model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    torch.save(merged_model.t3.state_dict(), merged_dir / "t3_mtl23ls_v2.pt")
'''
# Upstream has trailing spaces on some config lines; keep them to prove the patcher copes.
for _line in ("LEARNING_RATE = 2e-5", "WARMUP_STEPS = 500", "MAX_AUDIO_LENGTH = 400.0",
              "LORA_RANK = 32", "LORA_ALPHA = 64", "LORA_DROPOUT = 0.05"):
    FAKE_LORA = FAKE_LORA.replace(f"{_line}\n", f"{_line}  \n")
FAKE_FIX = '''from pathlib import Path
merged_dir = Path("checkpoints_lora/merged_model")
t3_cfg_path = merged_dir / "t3_mtl23ls_v2.pt"
output_path = merged_dir / "t3_mtl23ls_v2.safetensors"
'''


def config(tmp_path: Path, **overrides) -> setup_toolkit.RunConfig:
    values = {"data_dir": str(tmp_path / "audio_data"), "run_dir": str(tmp_path / "run")} | overrides
    return setup_toolkit.RunConfig(**values)


# --------------------------------------------------------------------------- setup_toolkit
def test_patch_lora_rewrites_every_motif(tmp_path: Path):
    cfg = config(tmp_path, epochs=3, lora_rank=8, lora_alpha=16.0, output_name="t3_test")
    text = setup_toolkit.patch_lora(FAKE_LORA, cfg)
    compile(text, "lora.py", "exec")
    assert f"AUDIO_DATA_DIR = {cfg.data_dir!r}" in text
    assert f"CHECKPOINT_DIR = {cfg.run_dir!r}" in text
    assert "EPOCHS = 3\n" in text and "LORA_RANK = 8\n" in text and "LORA_ALPHA = 16.0\n" in text
    assert "LANGUAGE_ID = 'fr'" in text and "BASE_T3 = 'v3'" in text
    assert "'ar'" not in text and '"ar"' not in text
    assert text.count("load_base_model(DEVICE)") == 2
    assert "ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)" not in text
    assert "frozen.requires_grad_(False)" in text
    assert "random.seed(SEED)\n    random.shuffle(samples)" in text
    assert "value=model.t3.hp.start_speech_token" in text and "value=model.t3.hp.stop_speech_token" in text
    assert 'merged_dir / f"{OUTPUT_NAME}.pt"' in text
    assert 'Path(CHECKPOINT_DIR) / "training_metrics.png"' in text


def test_upstream_speech_targets_can_be_kept(tmp_path: Path):
    text = setup_toolkit.patch_lora(FAKE_LORA, config(tmp_path, speech_bos_eos=False))
    assert "start_speech_token" not in text and "stop_speech_token" not in text


def test_patch_fails_loudly_when_upstream_changes(tmp_path: Path):
    changed = FAKE_LORA.replace("    random.shuffle(samples)\n", "")
    with pytest.raises(setup_toolkit.PatchError, match="seeded split"):
        setup_toolkit.patch_lora(changed, config(tmp_path))
    duplicated = FAKE_LORA + "EPOCHS = 7\n"
    with pytest.raises(setup_toolkit.PatchError, match="EPOCHS"):
        setup_toolkit.patch_lora(duplicated, config(tmp_path))


def test_patch_fix_merged(tmp_path: Path):
    cfg = config(tmp_path, output_name="t3_fr_ca")
    text = setup_toolkit.patch_fix_merged(FAKE_FIX, cfg)
    assert f"Path({cfg.run_dir!r}) / \"merged_model\"" in text
    assert '"t3_fr_ca.pt"' in text and '"t3_fr_ca.safetensors"' in text and "t3_mtl23ls_v2" not in text


@pytest.fixture
def upstream(tmp_path: Path, monkeypatch) -> Path:
    source = tmp_path / "upstream"
    source.mkdir()
    (source / "lora.py").write_text(FAKE_LORA, encoding="utf-8")
    (source / "fix_merged_model.py").write_text(FAKE_FIX, encoding="utf-8")
    monkeypatch.setattr(setup_toolkit, "UPSTREAM_SHA256", {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ("lora.py", "fix_merged_model.py")
    })
    return source


def test_cli_writes_patched_toolkit_and_manifest(tmp_path: Path, upstream: Path):
    run = tmp_path / "runs" / "r1"
    code = setup_toolkit.main(["--run-dir", str(run), "--data-dir", "rel/audio_data", "--rank", "8",
                               "--source-dir", str(upstream)])
    assert code == 0
    assert (run / "toolkit" / "lora.py").is_file() and (run / "toolkit" / "fix_merged_model.py").is_file()
    manifest = json.loads((run / "toolkit.json").read_text(encoding="utf-8"))
    assert manifest["config"]["lora_alpha"] == 16.0  # defaults to 2 × rank
    assert Path(manifest["config"]["data_dir"]).is_absolute()
    assert manifest["upstream"]["commit"] == setup_toolkit.TOOLKIT_COMMIT


def test_cli_keeps_symlinked_base_name(tmp_path: Path, upstream: Path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    link = tmp_path / "t3_fr_ca.safetensors"
    link.symlink_to(blob)
    run = tmp_path / "run"
    assert setup_toolkit.main(["--run-dir", str(run), "--data-dir", "d", "--base-t3", str(link),
                               "--source-dir", str(upstream)]) == 0
    assert json.loads((run / "toolkit.json").read_text())["config"]["base_t3"].endswith("t3_fr_ca.safetensors")


def test_cli_rejects_tampered_upstream(tmp_path: Path, upstream: Path, capsys):
    (upstream / "lora.py").write_text(FAKE_LORA + "# changed\n", encoding="utf-8")
    assert setup_toolkit.main(["--run-dir", str(tmp_path / "r"), "--data-dir", "d",
                               "--source-dir", str(upstream)]) == 1
    assert "SHA-256" in capsys.readouterr().err


@pytest.mark.parametrize("overrides, message", [
    ({"batch_size": 2}, "batch_size"),
    ({"base_t3": "v4"}, "base-t3"),
    ({"output_name": "../evil"}, "output-name"),
    ({"val_split": 0.0}, "val-split"),
])
def test_run_config_validation(tmp_path: Path, overrides, message):
    with pytest.raises(ValueError, match=message):
        config(tmp_path, **overrides).validate()


# --------------------------------------------------------------------------- merge_adapter
Q = "layers.0.self_attn.q_proj"


def small_state() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        f"tfmr.{Q}.weight": rng.standard_normal((4, 3)).astype(np.float32),
        "text_emb.weight": rng.standard_normal((5, 3)).astype(np.float32),
    }


def test_merge_lora_matches_toolkit_formula():
    state = small_state()
    before = {k: v.copy() for k, v in state.items()}
    a = np.arange(6, dtype=np.float32).reshape(2, 3) / 10
    b = np.arange(8, dtype=np.float32).reshape(4, 2) / 10
    merged = merge_adapter.merge_lora(state, {Q: (a, b)}, alpha=4.0)
    np.testing.assert_allclose(merged[f"tfmr.{Q}.weight"], before[f"tfmr.{Q}.weight"] + (b @ a) * 2.0, rtol=1e-6)
    np.testing.assert_array_equal(merged["text_emb.weight"], before["text_emb.weight"])
    np.testing.assert_array_equal(state[f"tfmr.{Q}.weight"], before[f"tfmr.{Q}.weight"])  # input untouched
    assert merged[f"tfmr.{Q}.weight"].dtype == np.float32


def test_merge_lora_rejects_mismatches():
    a, b = np.ones((2, 3), np.float32), np.ones((4, 2), np.float32)
    with pytest.raises(KeyError, match="k_proj"):
        merge_adapter.merge_lora(small_state(), {"layers.0.self_attn.k_proj": (a, b)}, 4.0)
    with pytest.raises(ValueError, match="do not fit"):
        merge_adapter.merge_lora(small_state(), {Q: (np.ones((2, 5), np.float32), b)}, 4.0)
    with pytest.raises(ValueError, match="no LoRA"):
        merge_adapter.merge_lora(small_state(), {}, 4.0)


def test_adapter_pairs_reads_both_toolkit_formats():
    a, b = np.ones((2, 3)), np.zeros((4, 2))
    final = {"lora_config": {"rank": 2, "alpha": 4}, "lora_weights": {Q: {"lora_A": a, "lora_B": b}}}
    pairs, alpha = merge_adapter.adapter_pairs(final)
    assert list(pairs) == [Q] and alpha == 4.0
    checkpoint = {"lora_state_dict": {f"{Q}.lora_A": a, f"{Q}.lora_B": b}}
    pairs, alpha = merge_adapter.adapter_pairs(checkpoint)
    assert list(pairs) == [Q] and alpha is None
    with pytest.raises(ValueError, match="Incomplete"):
        merge_adapter.adapter_pairs({"lora_state_dict": {f"{Q}.lora_A": a}})
    with pytest.raises(ValueError, match="Not a LoRA"):
        merge_adapter.adapter_pairs({"model": {}})


def test_run_config_is_read_next_to_the_adapter(tmp_path: Path):
    (tmp_path / "toolkit.json").write_text(json.dumps({"config": {"lora_alpha": 32.0, "base_t3": "v3"}}))
    assert merge_adapter.run_config(tmp_path / "checkpoint_epoch0_step10.pt")["lora_alpha"] == 32.0
    assert merge_adapter.run_config(tmp_path / "elsewhere" / "x.pt") == {}


# --------------------------------------------------------------------------- validate
@pytest.mark.parametrize("key", [
    "tfmr.layers.0.self_attn.q_proj.weight", "tfmr.layers.29.self_attn.o_proj.weight",
    "tfmr.layers.3.mlp.gate_proj.weight", "tfmr.layers.3.mlp.up_proj.weight", "tfmr.layers.3.mlp.down_proj.weight",
])
def test_projection_pattern_matches_lora_targets(key):
    assert PROJECTION_RE.match(key)


@pytest.mark.parametrize("key", ["tfmr.layers.0.input_layernorm.weight", "text_emb.weight",
                                 "tfmr.layers.0.self_attn.q_proj.bias", "cond_enc.spkr_enc.weight"])
def test_projection_pattern_rejects_other_tensors(key):
    assert not PROJECTION_RE.match(key)


def test_validation_accepts_a_projection_only_change():
    base = small_state()
    candidate = {k: v.copy() for k, v in base.items()}
    candidate[f"tfmr.{Q}.weight"][0, 0] += 0.5
    report = compare_state(base, candidate)
    assert report.ok, report.errors
    assert report.changed == [f"tfmr.{Q}.weight"] and report.max_abs_delta == pytest.approx(0.5)


def test_validation_flags_a_noop_lora():
    base = small_state()
    assert "no effect" in compare_state(base, dict(base)).errors[0]
    assert compare_state(base, dict(base), require_change=False).ok


def test_validation_flags_structural_problems():
    base = small_state()
    candidate = {k: v.copy() for k, v in base.items()}
    candidate["text_emb.weight"] = candidate["text_emb.weight"] + 1  # not a LoRA target
    candidate[f"tfmr.{Q}.weight"] = candidate[f"tfmr.{Q}.weight"].astype(np.float16)
    candidate[f"tfmr.{Q}.lora_A"] = np.zeros((2, 3), np.float32)
    errors = " | ".join(compare_state(base, candidate).errors)
    assert "unexpected tensor" in errors and "LoRA tensors left" in errors
    assert "dtype float16" in errors and "non-LoRA tensor(s) changed" in errors
    missing = {k: v for k, v in base.items() if k != "text_emb.weight"}
    assert "missing" in compare_state(base, missing).errors[0]
    reshaped = dict(base) | {"text_emb.weight": np.zeros((3, 5), np.float32)}
    assert "shape" in " ".join(compare_state(base, reshaped).errors)


SHORT = "Tu viens-tu souper icitte à soir ?"
LONG = "Fait qu'on a fini par jaser toute la soirée sur la galerie."


# Durations measured on CPU with the same reference: official v3 gave 2.0 s and
# 11.8 s; an over-aggressive fine-tune (LR 1e-3) gave 40 s for each sentence.
@pytest.mark.parametrize("sentence, seconds, expected", [
    (SHORT, 2.0, None),
    (LONG, 11.8, None),
    (SHORT, 40.0, "runaway"),
    (LONG, 40.0, "runaway"),
    (SHORT, 0.3, "truncated"),
])
def test_speaking_rate_flags_runaway_and_truncation(sentence, seconds, expected):
    problem = speaking_rate_problem(sentence, seconds)
    assert (problem is None) if expected is None else (expected in problem)
