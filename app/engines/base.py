"""Engine interface shared by all TTS back-ends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Languages supported by Chatterbox Multilingual (v2/v3).
LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


@dataclass
class SynthesisParams:
    language: str = "en"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    seed: int | None = None


class Engine(ABC):
    """A text-to-speech engine that can clone a voice from a reference clip."""

    name: str = "base"
    variant: str = ""
    sample_rate: int = 24000
    multilingual: bool = False
    languages: dict[str, str] = {"en": "English"}

    def __init__(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load model weights (slow; called once, lazily or at startup)."""

    @abstractmethod
    def synthesize(self, text: str, reference_wav: Path, params: SynthesisParams) -> np.ndarray:
        """Return mono float32 audio at ``self.sample_rate`` for ``text``."""

    def info(self) -> dict:
        return {
            "name": self.name,
            "variant": self.variant,
            "sample_rate": self.sample_rate,
            "multilingual": self.multilingual,
            "languages": self.languages,
            "loaded": self.loaded,
        }
