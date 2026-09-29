"""Phase 4: segmenting your own recordings and mixing them with the Quebec corpus."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.finetune import build_personal_dataset as mix
from scripts.finetune import prepare_qc_dataset as prep
from scripts.finetune.segment_recording import Segment, Word, cut_recording, padded_bounds, plan_segments


def words(*spec: tuple[float, float, str]) -> list[Word]:
    return [Word(*s) for s in spec]


# --------------------------------------------------------------------------- segmentation
def test_cuts_at_pauses_once_long_enough():
    ws = words((0.0, 1.0, " Bonjour"), (1.1, 2.5, " tout le monde."), (3.0, 4.0, " On"), (4.1, 5.6, " commence."))
    assert plan_segments(ws) == [Segment(0.0, 2.5, "Bonjour tout le monde."), Segment(3.0, 5.6, "On commence.")]


def test_short_first_phrase_merges_with_the_next():
    ws = words((0.0, 0.8, " Oui."), (1.5, 2.5, " Je"), (2.6, 3.4, " viens."))
    assert plan_segments(ws) == [Segment(0.0, 3.4, "Oui. Je viens.")]


def test_never_exceeds_max_seconds_even_without_pauses():
    ws = [Word(i * 1.0, i * 1.0 + 0.95, f" m{i}") for i in range(30)]
    segments = plan_segments(ws, max_seconds=12.0)
    assert all(s.duration <= 12.0 for s in segments)
    assert " ".join(s.text for s in segments) == " ".join(f"m{i}" for i in range(30))


def test_word_pieces_join_without_spaces_and_short_tail_is_dropped():
    ws = words((0.0, 0.5, " Dites"), (0.5, 0.9, "-moi,"), (1.0, 1.2, " s"), (1.2, 1.5, "'il"),
               (1.6, 2.4, " vous plaît."), (4.0, 4.3, " euh"))
    assert plan_segments(ws) == [Segment(0.0, 2.4, "Dites-moi, s'il vous plaît.")]


def test_padding_stays_between_neighbours():
    segs = [Segment(1.0, 3.0, "a"), Segment(3.1, 5.0, "b")]
    assert padded_bounds(segs, total=5.05) == [(0.9, pytest.approx(3.05)), (pytest.approx(3.05), 5.05)]


def test_cut_recording_writes_clips(tmp_path: Path):
    sr = 24000
    rec = tmp_path / "rec.wav"
    sf.write(str(rec), np.zeros(sr * 6, np.float32), sr)
    ws = words((0.0, 1.2, " Tu"), (1.3, 2.4, " viens-tu ?"), (3.0, 4.0, " Oui"), (4.1, 5.2, " j'arrive."))
    clips = cut_recording(rec, tmp_path, ws, "me", name="session1")
    assert [c.name for c in clips] == ["me_session1_0000.wav", "me_session1_0001.wav"]
    assert clips[0].text == "Tu viens-tu ?" and clips[0].client_id == "me"
    assert clips[0].duration == pytest.approx(sf.info(str(clips[0].source)).duration, abs=1e-3)


def test_single_speaker_holdout_is_per_clip():
    clips = [prep.Clip(Path(f"/x/c{i}.wav"), f"texte {i}", 3.0, "me") for i in range(20)]
    assert prep.split_holdout(clips, 3)[1] == []  # by speaker: the only speaker must stay in training
    train, holdout = prep.split_holdout(clips, 3, by_speaker=False)
    assert len(holdout) == 3 and len(train) == 17


# --------------------------------------------------------------------------- mixing
def rows(prefix: str, n: int) -> list[dict[str, str]]:
    return [{"file_name": f"audio/{prefix}{i}.wav", "transcription": f"t{i}", "duration_seconds": "3",
             "client_id": prefix} for i in range(n)]


def test_mix_rows_oversamples_own_and_samples_qc():
    mixed = mix.mix_rows(rows("me", 4), rows("qc", 10), oversample=3, qc_fraction=0.3, seed=1)
    own = [r for r in mixed if r["file_name"].startswith("audio/own_")]
    qc = [r for r in mixed if r["file_name"].startswith("audio/qc_")]
    assert len(own) == 12 and len({r["file_name"] for r in own}) == 4 and len(qc) == 3
    assert mixed == mix.mix_rows(rows("me", 4), rows("qc", 10), oversample=3, qc_fraction=0.3, seed=1)


@pytest.mark.parametrize("oversample, fraction", [(0, 0.3), (2, 1.5)])
def test_mix_rows_validates(oversample, fraction):
    with pytest.raises(ValueError):
        mix.mix_rows(rows("me", 1), rows("qc", 1), oversample, fraction, 0)


def write_audio_data(root: Path, train: list[dict[str, str]], holdout: list[dict[str, str]]) -> None:
    (root / "audio").mkdir(parents=True)
    for name, data in (("metadata.csv", train), ("holdout.csv", holdout)):
        with (root / name).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=prep.FIELDS)
            writer.writeheader()
            writer.writerows(data)
        for row in data:
            (root / row["file_name"]).write_bytes(row["file_name"].encode())


def test_build_personal_dataset_cli(tmp_path: Path):
    own, qc = tmp_path / "own", tmp_path / "qc"
    write_audio_data(own, rows("me", 3), rows("hold", 1))
    write_audio_data(qc, rows("qc", 4), [])
    out = tmp_path / "mix"
    assert mix.main(["--own", str(own), "--qc", str(qc), "--out", str(out), "--qc-fraction", "0.5"]) == 0
    train, holdout = mix.read_rows(out / "metadata.csv"), mix.read_rows(out / "holdout.csv")
    assert len(train) == 3 * 3 + 2 and [r["file_name"] for r in holdout] == ["audio/own_hold0.wav"]
    for row in train + holdout:
        assert (out / row["file_name"]).is_file()
    assert len(list((out / "audio").iterdir())) == 3 + 1 + 2  # only the files actually used


def test_build_personal_dataset_needs_own_clips(tmp_path: Path):
    (tmp_path / "own").mkdir()
    assert mix.main(["--own", str(tmp_path / "own"), "--qc", str(tmp_path), "--out", str(tmp_path / "o")]) == 1
