from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from app.audio import AudioError, concat_with_silence, convert_to_wav, duration, peak_normalize, resample
from tests.conftest import make_wav


def test_convert_resamples_mixes_down_and_trims(tmp_path):
    src = make_wav(tmp_path / "in.wav", seconds=5.0, sr=44100, channels=2)
    dst = tmp_path / "out.wav"
    seconds = convert_to_wav(src, dst, sr=24000, max_seconds=2)
    assert seconds == pytest.approx(2.0, abs=0.05)
    info = sf.info(str(dst))
    assert info.samplerate == 24000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert duration(dst) == pytest.approx(2.0, abs=0.05)


def test_convert_rejects_garbage(tmp_path):
    src = tmp_path / "bad.wav"
    src.write_bytes(b"definitely not audio")
    with pytest.raises(AudioError):
        convert_to_wav(src, tmp_path / "out.wav")


def test_resample_lengths():
    audio = np.zeros(16000, dtype=np.float32)
    assert resample(audio, 16000, 24000).shape == (24000,)
    assert resample(audio, 16000, 16000) is audio or resample(audio, 16000, 16000).shape == (16000,)


def test_concat_with_silence():
    a = np.ones(100, dtype=np.float32)
    b = np.ones(50, dtype=np.float32)
    out = concat_with_silence([a, b], sr=1000, gap_seconds=0.1)
    assert out.shape == (250,)
    assert out[100:200].sum() == 0
    assert concat_with_silence([], 1000).shape == (0,)


def test_peak_normalize():
    loud = np.array([0.0, 2.0, -4.0], dtype=np.float32)
    normalized = peak_normalize(loud, peak=0.95)
    assert np.max(np.abs(normalized)) == pytest.approx(0.95)
    quiet = np.array([0.1, -0.2], dtype=np.float32)
    assert np.array_equal(peak_normalize(quiet), quiet)
    silent = np.zeros(10, dtype=np.float32)
    assert np.array_equal(peak_normalize(silent), silent)
