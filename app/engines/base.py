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


@dataclass
class ConversionParams:
    """Parameters for speech-to-speech voice conversion.

    Unlike TTS, the accent, rhythm and prosody all come from the *source*
    recording; the engine only transfers the target voice's timbre.  These
    knobs trade timbre-match strength against source-fidelity, which is what
    lets a Quebec-accented take keep its accent while sounding like the
    target voice.
    """

    # How strongly to pull the output toward the target timbre.  Lower values
    # preserve more of the source (accent, breathiness); higher values lock
    # onto the target voice at the cost of some source character.
    strength: float = 0.75
    # Optional pitch shift in semitones (e.g. male->female target).  0 = keep
    # the source pitch contour, which is where most of the accent lives.
    pitch_shift: float = 0.0
    # Diffusion / decoder steps for engines that expose them (quality vs speed).
    steps: int = 25
    seed: int | None = None


class Engine(ABC):
    """A voice engine that can clone a voice from a reference clip.

    Every engine does text-to-speech (:meth:`synthesize`).  Engines that also
    do speech-to-speech voice conversion set :attr:`supports_conversion` and
    override :meth:`convert`; the worker routes "convert" jobs to them.
    """

    name: str = "base"
    variant: str = ""
    sample_rate: int = 24000
    multilingual: bool = False
    languages: dict[str, str] = {"en": "English"}
    # True when the engine implements :meth:`convert` (speech-to-speech).
    supports_conversion: bool = False

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

    def unload(self) -> None:
        """Free the model (and its GPU memory) so another process can use the GPU."""
        self._loaded = False

    def set_t3(self, t3_model: str, t3_path: Path | None) -> None:
        """Switch the T3 checkpoint (``v2``/``v3`` or a fine-tuned file), in place if loaded."""
        raise NotImplementedError(f"Engine '{self.name}' has no swappable T3 checkpoint")

    def convert(self, source_wav: Path, reference_wav: Path, params: ConversionParams) -> np.ndarray:
        """Convert ``source_wav`` to the voice in ``reference_wav``.

        Returns mono float32 audio at ``self.sample_rate``.  The linguistic
        content, timing and accent are taken from ``source_wav``; only the
        timbre comes from ``reference_wav``.  Engines that do not support
        conversion leave this as-is.
        """
        raise NotImplementedError(f"Engine '{self.name}' does not support voice conversion")

    def info(self) -> dict:
        return {
            "name": self.name,
            "variant": self.variant,
            "sample_rate": self.sample_rate,
            "multilingual": self.multilingual,
            "languages": self.languages,
            "supports_conversion": self.supports_conversion,
            "loaded": self.loaded,
        }
