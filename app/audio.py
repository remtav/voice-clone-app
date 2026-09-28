"""Audio helpers: reference-clip conversion, concatenation and writing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 24000


class AudioError(Exception):
    """Raised when an audio file cannot be decoded or converted."""


def convert_to_wav(src: Path, dst: Path, sr: int = TARGET_SR, max_seconds: float | None = None) -> float:
    """Convert any audio/video file to a mono 16-bit PCM WAV at ``sr`` Hz.

    Uses ffmpeg when available (handles webm/opus from browser recordings,
    mp3, m4a, ...).  Falls back to libsndfile for WAV/FLAC/OGG inputs.
    Returns the duration of the written file in seconds.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr)]
        if max_seconds:
            cmd += ["-t", str(max_seconds)]
        cmd += ["-acodec", "pcm_s16le", "-f", "wav", str(dst)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AudioError(f"Could not decode audio: {proc.stderr.strip()[:400] or 'ffmpeg failed'}")
    else:
        _convert_with_soundfile(src, dst, sr, max_seconds)
    return duration(dst)


def _convert_with_soundfile(src: Path, dst: Path, sr: int, max_seconds: float | None) -> None:
    try:
        data, in_sr = sf.read(str(src), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - libsndfile raises many types
        raise AudioError(
            "Could not decode audio. ffmpeg is not installed, so only WAV/FLAC/OGG files are supported."
        ) from exc
    mono = data.mean(axis=1)
    if in_sr != sr:
        mono = resample(mono, in_sr, sr)
    if max_seconds:
        mono = mono[: int(max_seconds * sr)]
    sf.write(str(dst), mono, sr, subtype="PCM_16")


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interpolation resampling (fallback only; ffmpeg is preferred)."""
    if src_sr == dst_sr or audio.size == 0:
        return audio.astype(np.float32)
    n_out = int(round(audio.shape[0] * dst_sr / src_sr))
    src_t = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    dst_t = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def concat_with_silence(pieces: list[np.ndarray], sr: int, gap_seconds: float = 0.25) -> np.ndarray:
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(sr * gap_seconds), dtype=np.float32)
    out: list[np.ndarray] = []
    for i, piece in enumerate(pieces):
        if i:
            out.append(gap)
        out.append(np.asarray(piece, dtype=np.float32).reshape(-1))
    return np.concatenate(out)


def peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    current = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current <= 1e-6:
        return audio
    if current <= peak:
        return audio
    return (audio * (peak / current)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")
