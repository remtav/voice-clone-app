from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.finetune import prepare_qc_dataset as prep
from tests.conftest import make_wav


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def clip(name: str, speaker: str = "", duration: float = 3.0) -> prep.Clip:
    return prep.Clip(Path(f"/src/{name}.wav"), f"texte {name}", duration, speaker)


# --------------------------------------------------------------------------- text
def test_normalize_text_fixes_typography_but_keeps_quebec_forms():
    raw = "  Faque  chu allé au dépanneur , tsé ?  J’ai dit « icitte » pis… "
    assert prep.normalize_text(raw) == "Faque chu allé au dépanneur, tsé ? J'ai dit \" icitte \" pis..."


@pytest.mark.parametrize("value", ["Québécois", "français du Québec", "Canadien", "Canadian French", "QC", "Montréal"])
def test_quebec_accent_tags_match(value):
    assert prep.matches_qc_accent(value)


@pytest.mark.parametrize("value", ["", "France", "Belgique", "Suisse", "Français de France"])
def test_other_accent_tags_do_not_match(value):
    assert not prep.matches_qc_accent(value)


# --------------------------------------------------------------------------- holdout
def test_holdout_is_deterministic_and_disjoint():
    clips = [clip(f"c{i}") for i in range(100)]
    train, holdout = prep.split_holdout(clips, 10)
    assert len(holdout) == 10 and len(train) == 90
    assert {c.name for c in train}.isdisjoint(c.name for c in holdout)
    # Selection depends on file names only, not on input order.
    assert {c.name for c in prep.split_holdout(list(reversed(clips)), 10)[1]} == {c.name for c in holdout}


def test_holdout_keeps_speakers_out_of_training():
    clips = [clip(f"{s}-{i}", speaker=s) for s in "abcdefgh" for i in range(6)]
    train, holdout = prep.split_holdout(clips, 10)
    held_speakers = {c.client_id for c in holdout}
    assert len(holdout) >= 10
    assert held_speakers.isdisjoint(c.client_id for c in train)


def test_holdout_never_empties_training():
    train, holdout = prep.split_holdout([clip("only", speaker="x")], 40)
    assert len(train) == 1 and holdout == []


# --------------------------------------------------------------------------- option A
@pytest.fixture
def processed(tmp_path: Path) -> Path:
    src = tmp_path / "processed"
    wavs = src / "wavs" / "qc"
    wavs.mkdir(parents=True)
    make_wav(wavs / "ok1.wav", seconds=2.0, sr=24000)
    make_wav(wavs / "ok2.wav", seconds=3.0, sr=24000)
    make_wav(wavs / "dup.wav", seconds=2.0, sr=24000)
    make_wav(wavs / "short.wav", seconds=0.5, sr=24000)
    make_wav(wavs / "long.wav", seconds=16.0, sr=24000)
    make_wav(src / "wavs" / "flat.wav", seconds=2.0, sr=24000)  # tolerated layout
    lines = [
        "audio_file|text",
        "/workspace/data/processed/wavs/qc/ok1.wav|Tu viens-tu souper icitte à soir ?",
        "/workspace/data/processed/wavs/qc/ok2.wav|Il faut que je stationne mon char.",
        "/workspace/data/processed/wavs/qc/dup.wav|il faut que je stationne mon char",
        "/workspace/data/processed/wavs/qc/short.wav|Trop court pour servir.",
        "/workspace/data/processed/wavs/qc/long.wav|Beaucoup trop long pour servir.",
        "/workspace/data/processed/wavs/qc/gone.wav|Ce fichier n'a pas été publié.",
        "/workspace/data/processed/wavs/flat.wav|Un chemin sans sous-dossier qc.",
        "",
    ]
    (src / "metadata.csv").write_text("\n".join(lines), encoding="utf-8")
    return src


def test_read_processed_filters_and_counts(processed: Path):
    clips, stats = prep.read_processed(processed)
    assert [c.name for c in clips] == ["ok1.wav", "ok2.wav", "flat.wav"]
    assert clips[0].text == "Tu viens-tu souper icitte à soir ?"
    assert clips[0].duration == pytest.approx(2.0, abs=0.01)
    assert stats == {"missing_audio": 1, "out_of_bounds": 2, "duplicate_text": 1}


def test_read_processed_rejects_unknown_layout(tmp_path: Path):
    (tmp_path / "metadata.csv").write_text("path,text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected header"):
        prep.read_processed(tmp_path)


def test_from_processed_cli_writes_toolkit_layout(processed: Path, tmp_path: Path, capsys):
    out = tmp_path / "audio_data"
    assert prep.main(["from-processed", "--src", str(processed), "--out", str(out), "--holdout", "1"]) == 0
    train, holdout = read_rows(out / "metadata.csv"), read_rows(out / "holdout.csv")
    assert list(train[0]) == prep.FIELDS
    assert len(train) == 2 and len(holdout) == 1
    for row in train + holdout:
        audio = out / row["file_name"]
        assert audio.is_file() and row["file_name"].startswith("audio/")
        assert float(row["duration_seconds"]) == pytest.approx(sf.info(str(audio)).duration, abs=0.01)
    assert "train: 2 clips" in capsys.readouterr().out


def test_cli_fails_without_usable_clips(tmp_path: Path):
    src = tmp_path / "empty"
    src.mkdir()
    (src / "metadata.csv").write_text("audio_file|text\n", encoding="utf-8")
    assert prep.main(["from-processed", "--src", str(src), "--out", str(tmp_path / "out")]) == 1


def test_write_dataset_rejects_name_clashes(tmp_path: Path):
    a, b = tmp_path / "a" / "x.wav", tmp_path / "b" / "x.wav"
    for p in (a, b):
        p.parent.mkdir()
        make_wav(p)
    with pytest.raises(ValueError, match="Duplicate"):
        prep.write_dataset([prep.Clip(a, "un texte", 2.0), prep.Clip(b, "autre texte", 2.0)], tmp_path / "out")


@pytest.mark.parametrize("mode", ["hardlink", "symlink", "copy"])
def test_place_modes(tmp_path: Path, mode: str):
    src = make_wav(tmp_path / "src.wav")
    dst = tmp_path / "dst.wav"
    prep.place(src, dst, mode)
    prep.place(src, dst, mode)  # idempotent
    assert dst.read_bytes() == src.read_bytes()
    assert dst.is_symlink() == (mode == "symlink")


# --------------------------------------------------------------------------- option B
def cv_row(path: str, sentence: str, accents: str = "Québécois", speaker: str = "s1", up: int = 2, down: int = 0):
    return {"client_id": speaker, "path": path, "sentence": sentence, "accents": accents,
            "up_votes": str(up), "down_votes": str(down)}


@pytest.mark.parametrize("value, expected", [
    ("Français de France", True), ("Belgique", True), ("Swiss French", True),
    ("Québécois, France", False), ("Canadien", False), ("Français", False), ("", False),
])
def test_europe_accent_tags(value, expected):
    assert prep.matches_europe_accent(value) is expected


def test_select_cv_rows_europe_set():
    rows = [cv_row("a.mp3", "Une phrase de France.", accents="Français de France"),
            cv_row("b.mp3", "Une phrase du Québec.", accents="Québécois")]
    selected, stats = prep.select_cv_rows(rows, accent="europe")
    assert [r["path"] for r in selected] == ["a.mp3"] and stats == {"not_europe": 1}


def test_select_cv_rows_applies_every_filter():
    rows = [
        cv_row("a.mp3", "Première phrase du locuteur."),
        cv_row("b.mp3", "Accent de France.", accents="France"),
        cv_row("c.mp3", "Pas assez de votes.", up=1),
        cv_row("d.mp3", "Un vote contre.", down=1),
        cv_row("e.mp3", "première phrase du locuteur"),
        cv_row("f.mp3", "Deuxième phrase du locuteur."),
        cv_row("g.mp3", "Troisième phrase, au-delà du plafond."),
        cv_row("h.mp3", "Autre locuteur, même accent.", speaker="s2"),
    ]
    selected, stats = prep.select_cv_rows(rows, max_per_speaker=2)
    assert [r["path"] for r in selected] == ["a.mp3", "f.mp3", "h.mp3"]
    assert stats == {"not_qc": 1, "votes": 2, "duplicate_or_empty_text": 1, "speaker_cap": 1}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_from_common_voice_converts_and_trims(tmp_path: Path):
    cv = tmp_path / "cv"
    clips = cv / "clips"
    clips.mkdir(parents=True)
    sr = 48000
    tone = 0.4 * np.sin(2 * np.pi * 220 * np.arange(int(sr * 2.0)) / sr)
    silence = np.zeros(sr)
    # Common Voice ships mp3; ffmpeg probes content, so a WAV payload is fine here.
    sf.write(str(clips / "qc.mp3"), np.concatenate([silence, tone, silence]).astype(np.float32), sr, format="WAV")
    sf.write(str(clips / "fr.mp3"), tone.astype(np.float32), sr, format="WAV")
    header = ["client_id", "path", "sentence", "up_votes", "down_votes", "accents"]
    with (cv / "validated.tsv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(header)
        writer.writerow(["s1", "qc.mp3", "On va-tu au dépanneur ?", "3", "0", "Québécois"])
        writer.writerow(["s2", "fr.mp3", "Nous allons à l'épicerie.", "3", "0", "France"])
        writer.writerow(["s3", "missing.mp3", "Ce clip est absent.", "3", "0", "Canadien"])
    out = tmp_path / "audio_data"
    assert prep.main(["from-common-voice", "--cv-dir", str(cv), "--out", str(out), "--holdout", "0"]) == 0
    rows = read_rows(out / "metadata.csv")
    assert [r["file_name"] for r in rows] == ["audio/qc.wav"]
    assert rows[0]["client_id"] == "s1"
    info = sf.info(str(out / "audio" / "qc.wav"))
    assert info.samplerate == prep.TARGET_SR and info.channels == 1
    assert 1.8 < info.duration < 2.3  # the 1 s of silence on each side is trimmed
