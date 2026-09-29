"""A dependency-free engine used for tests and GPU-less development.

It produces a beepy "robot" tone whose length scales with the text, so the
whole upload/generate/download pipeline can be exercised without torch.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from app.engines.base import LANGUAGE_NAMES, Engine, SynthesisParams


class FakeEngine(Engine):
    name = "fake"
    variant = "tone"
    sample_rate = 24000
    multilingual = True
    languages = dict(LANGUAGE_NAMES)

    def __init__(self) -> None:
        super().__init__()
        self.t3_model = "v3"

    def load(self) -> None:
        self._loaded = True

    def set_t3(self, t3_model: str, t3_path: Path | None) -> None:
        if t3_path is not None and not t3_path.is_file():
            raise FileNotFoundError(f"T3 checkpoint not found: {t3_path}")
        self.t3_model = t3_path.name if t3_path is not None else t3_model

    def info(self) -> dict:
        data = super().info()
        data["t3_model"] = self.t3_model
        return data

    def synthesize(self, text: str, reference_wav: Path, params: SynthesisParams) -> np.ndarray:
        if not self._loaded:
            self.load()
        sr = self.sample_rate
        seed = params.seed if params.seed is not None else zlib.crc32(text.encode("utf-8"))
        rng = np.random.default_rng(seed)
        base_freq = 160.0 + (zlib.crc32(reference_wav.name.encode()) % 140)
        out: list[np.ndarray] = []
        for word in text.split():
            n = int(sr * (0.05 + 0.02 * min(len(word), 12)))
            t = np.arange(n) / sr
            freq = base_freq * (1.0 + 0.15 * params.exaggeration * rng.standard_normal())
            envelope = np.minimum(1.0, np.minimum(t * 40.0, (t[-1] - t) * 40.0 + 1e-3))
            out.append(0.3 * np.sin(2 * np.pi * freq * t) * envelope)
            out.append(np.zeros(int(sr * 0.03), dtype=np.float32))
        if not out:
            out.append(np.zeros(int(sr * 0.1), dtype=np.float32))
        audio = np.concatenate(out).astype(np.float32)
        audio += (0.002 * params.temperature) * rng.standard_normal(audio.shape).astype(np.float32)
        return audio
