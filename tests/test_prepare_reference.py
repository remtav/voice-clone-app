"""scripts.prepare_reference: trims silence and normalises a reference clip."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.prepare_reference import SR, noise_floor_db, prepare


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_prepare_trims_silence_and_resamples(tmp_path: Path):
    sr = 44100
    t = np.arange(int(sr * 2)) / sr
    speech = (0.05 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)
    src = tmp_path / "raw.wav"
    sf.write(str(src), np.concatenate([silence, speech, silence]), sr)

    audio = prepare(src, tmp_path / "out" / "reference.wav")
    info = sf.info(str(tmp_path / "out" / "reference.wav"))
    assert info.samplerate == SR and info.channels == 1
    # About 2 s of the original 4 s remain; louder than the -26 dBFS input.
    assert 1.5 < len(audio) / SR < 2.6
    assert np.abs(audio).max() > 0.1


def test_noise_floor():
    assert noise_floor_db(np.full(1000, 0.001, dtype=np.float32)) == pytest.approx(-60, abs=0.1)
