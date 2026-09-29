"""Phase 3 evaluation harness, with fake models (no torch, no downloads)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.finetune.eval import abx, accent_probe, battery, report, score, text


# --------------------------------------------------------------------------- text
def test_quebec_forms_and_asr_standardization_score_zero():
    reference = "Faque j'ai pris mon char pis chu allé au dépanneur, tsé."
    asr = "Ça fait que j’ai pris mon char puis je suis allé au dépanneur, tu sais."
    assert text.wer(reference, asr) == 0.0
    assert text.cer(reference, asr) == 0.0


def test_normalize_drops_punctuation_hyphens_and_case():
    assert text.normalize("Tu viens-tu souper icitte à soir ?") == "tu viens tu souper ici ce soir"


def test_error_rates():
    assert text.edit_distance(list("abc"), list("abd")) == 1
    assert text.wer("un deux trois quatre", "un deux quatre") == pytest.approx(0.25)
    assert text.wer("", "") == 0.0
    assert text.cer("chat", "chats") == pytest.approx(0.25)


def test_numbers_are_spelled_when_num2words_is_available():
    pytest.importorskip("num2words")
    assert text.wer("Ça coûte quatre-vingt-dix-sept dollars", "Ça coûte 97 dollars") == 0.0


# --------------------------------------------------------------------------- battery
def test_sentence_battery_covers_every_category():
    sentences = battery.load_sentences()
    assert len(sentences) == 50
    counts: dict[str, int] = {}
    for category, _ in sentences:
        counts[category] = counts.get(category, 0) + 1
    assert set(counts) == {"affrication", "laxing", "a_backing", "diphthongs", "lexicon", "numbers",
                           "questions", "prosody"}
    assert min(counts.values()) >= 6
    assert battery.load_sentences(limit=3) == sentences[:3]


def test_grid_is_the_full_product_with_unique_paths():
    cells = battery.build_grid({"base": "v3", "fr_ca": "x.safetensors"}, [0.5, 0.8],
                               {"me": "/me.wav", "qc1": "/qc1.wav", "qc2": "/qc2.wav"}, battery.load_sentences())
    assert len(cells) == 600
    assert len({c.wav for c in cells}) == 600
    assert cells[0].wav == "base/cfg0.50/me/00.wav"


def test_parse_pairs_validates():
    assert battery.parse_pairs(["base=v3", "ft=a=b.safetensors"], "checkpoint") == {"base": "v3",
                                                                                  "ft": "a=b.safetensors"}
    with pytest.raises(ValueError, match="name=value"):
        battery.parse_pairs(["v3"], "checkpoint")
    with pytest.raises(ValueError, match="duplicate"):
        battery.parse_pairs(["a=1", "a=2"], "voice")


# --------------------------------------------------------------------------- score
def manifest_rows() -> list[dict[str, str]]:
    return [{"checkpoint": "base", "cfg_weight": "0.8", "voice": "me", "reference": "/ref.wav", "sentence_id": "0",
             "category": "lexicon", "text": "On va-tu au dépanneur ?", "wav": "base/0.wav", "seconds": "2.0"}]


def test_score_rows_uses_every_scorer_and_skips_done_clips():
    class FakeProbe:
        def predict_proba(self, features):
            return np.array([0.75])

    rows = list(score.score_rows(
        manifest_rows(), Path("/battery"),
        transcribe=lambda wav: "On va-tu au dépanneur",
        speaker=lambda wav: np.array([1.0, 0.0]) if wav.name == "0.wav" else np.array([1.0, 1.0]),
        probe=FakeProbe(), embed=lambda wav: np.zeros(3),
    ))
    assert rows[0]["wer"] == 0.0 and rows[0]["p_qc"] == 0.75
    assert rows[0]["speaker_sim"] == pytest.approx(1 / math.sqrt(2), abs=1e-4)


def test_score_rows_only_fills_missing_columns():
    class FakeProbe:
        def predict_proba(self, features):
            return np.array([0.4])

    done = {"base/0.wav": {**manifest_rows()[0], "hypothesis": "x", "wer": "0.5", "cer": "0.2", "p_qc": ""}}
    [row] = score.score_rows(manifest_rows(), Path("/b"), transcribe=lambda w: 1 / 0,  # ASR must not rerun
                             probe=FakeProbe(), embed=lambda w: np.zeros(2), done=done)
    assert row["wer"] == "0.5" and row["p_qc"] == 0.4


def test_cosine():
    assert score.cosine(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert score.cosine(np.array([1.0, 0.0]), np.array([0.0, 3.0])) == pytest.approx(0.0)


# --------------------------------------------------------------------------- ABX
def abx_manifest(n: int = 12) -> list[dict[str, str]]:
    return [{"checkpoint": ckpt, "cfg_weight": "0.8", "voice": "me", "sentence_id": str(i), "text": f"phrase {i}",
             "wav": f"{ckpt}/{i:02d}.wav"} for ckpt in ("base", "fr_ca") for i in range(n)]


def test_make_pairs_is_blind_and_deterministic():
    pairs = abx.make_pairs(abx_manifest(), "base", "fr_ca", 0.8, "me", n=10, seed=3)
    assert pairs == abx.make_pairs(abx_manifest(), "base", "fr_ca", 0.8, "me", n=10, seed=3)
    assert len(pairs) == 10
    for p in pairs:
        cand_wav = p["a_wav"] if p["candidate"] == "A" else p["b_wav"]
        assert cand_wav.startswith("fr_ca/") and p["a_wav"][-6:] == p["b_wav"][-6:]
    assert {p["candidate"] for p in pairs} == {"A", "B"}
    with pytest.raises(ValueError, match="no clip shared"):
        abx.make_pairs(abx_manifest(), "base", "fr_ca", 0.5, None, n=10, seed=0)


def test_score_answers_ignores_blank_choices():
    answers = [{"pair": "01", "choice": "a"}, {"pair": "02", "choice": "B"}, {"pair": "03", "choice": ""}]
    assert abx.score_answers(answers, {"01": "A", "02": "A", "03": "B"}) == (1, 2)


def test_abx_make_and_read_roundtrip(tmp_path: Path):
    rows = abx_manifest(4)
    for row in rows:
        (tmp_path / row["wav"]).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / row["wav"]).write_bytes(row["wav"].encode())
    with (tmp_path / "manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert abx.main(["make", "--battery", str(tmp_path), "--candidate", "fr_ca", "-n", "4"]) == 0
    key = json.loads((tmp_path / "abx" / "key.json").read_text())["key"]
    with (tmp_path / "abx" / "pairs.csv").open(newline="") as fh:
        answers = list(csv.DictReader(fh))
    for row in answers:  # the listener always picks the fine-tune
        assert (tmp_path / "abx" / row["a"]).is_file()
        row["choice"] = key[row["pair"]]
    with (tmp_path / "abx" / "pairs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=abx.PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(answers)
    assert abx.read_abx(tmp_path) == (4, 4)


# --------------------------------------------------------------------------- report
def scores(ckpt: str, cfg: float, wer: float, sim: float, p_qc: float, n: int = 4) -> list[dict[str, str]]:
    return [{"checkpoint": ckpt, "cfg_weight": str(cfg), "wer": str(wer), "cer": str(wer / 2),
             "speaker_sim": str(sim), "p_qc": str(p_qc), "seconds": "3"} for _ in range(n)]


def test_report_passes_a_better_accent_and_rejects_a_worse_one():
    rows = (scores("base", 0.8, 0.07, 0.48, 0.17) + scores("good", 0.8, 0.08, 0.66, 0.40)
            + scores("mumbly", 0.8, 0.20, 0.66, 0.60) + scores("flat", 0.8, 0.07, 0.48, 0.20))
    decisions = report.evaluate(report.aggregate(rows), "base")
    status = {d.candidate.checkpoint: d.passed for d in decisions}
    assert status == {"good": True, "mumbly": False, "flat": False}
    assert report.best(decisions).candidate.checkpoint == "good"


def test_report_compares_at_the_same_cfg_and_uses_abx():
    rows = scores("base", 0.5, 0.07, 0.5, 0.2) + scores("ft", 0.5, 0.07, 0.5, 0.5) + scores("ft", 0.8, 0.07, 0.5, 0.5)
    with pytest.raises(ValueError, match="cfg 0.8"):
        report.evaluate(report.aggregate(rows), "base")
    agg = report.aggregate(scores("base", 0.8, 0.07, 0.5, 0.2) + scores("ft", 0.8, 0.07, 0.5, 0.5))
    [lost] = report.evaluate(agg, "base", abx={("ft", 0.8): (5, 10)})
    assert not lost.passed and not lost.checks["ABX ≥ 70%"]


def test_report_without_probe_skips_the_accent_check():
    agg = report.aggregate([{**r, "p_qc": ""} for r in scores("base", 0.8, 0.07, 0.5, 0) + scores("ft", 0.8, 0.07,
                                                                                                  0.5, 0)])
    [decision] = report.evaluate(agg, "base")
    assert decision.passed and not any("p_qc" in name for name in decision.checks)


def test_report_cli_writes_markdown(tmp_path: Path):
    rows = scores("base", 0.8, 0.07, 0.48, 0.17) + scores("fr_ca", 0.8, 0.08, 0.66, 0.40)
    with (tmp_path / "scores.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert report.main(["--battery", str(tmp_path)]) == 0
    markdown = (tmp_path / "report.md").read_text()
    assert "fr_ca @ cfg 0.8: PASS" in markdown and "Use `fr_ca` with CFG weight 0.8" in markdown


# --------------------------------------------------------------------------- accent probe
def test_probe_learns_and_roundtrips(tmp_path: Path):
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(0.6, 1, (150, 16)), rng.normal(-0.6, 1, (150, 16))
    probe = accent_probe.train_probe(pos, neg, seed=1)
    assert probe.heldout_accuracy > 0.9
    probe.save(tmp_path / "probe.npz")
    loaded = accent_probe.Probe.load(tmp_path / "probe.npz")
    np.testing.assert_allclose(loaded.predict_proba(pos[:5]), probe.predict_proba(pos[:5]))
    assert loaded.layer == accent_probe.DEFAULT_LAYER and loaded.heldout_accuracy == probe.heldout_accuracy


def test_dataset_wavs_reads_train_and_holdout(tmp_path: Path):
    for name, files in (("metadata.csv", ["audio/a.wav", "audio/b.wav"]), ("holdout.csv", ["audio/c.wav"])):
        with (tmp_path / name).open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file_name", "transcription"])
            writer.writerows([f, "x"] for f in files)
    assert [p.name for p in accent_probe.dataset_wavs(tmp_path, None, 0)] == ["a.wav", "b.wav", "c.wav"]
    assert len(accent_probe.dataset_wavs(tmp_path, 2, 0)) == 2
