"""Speech-to-text for reference clips: writes the transcript Qwen3-TTS needs.

``STT_ENGINE`` picks the back-end:

* ``faster-whisper`` (default): Whisper through CTranslate2.  The model loads
  on the first request and stays loaded; it falls back to the CPU when CUDA
  cannot be used (no GPU, or CTranslate2 cannot find the CUDA libraries).
* ``fake``: returns a fixed sentence, for tests and GPU-less development.
* ``none``: off; the transcript is typed by hand.

Whisper writes standard spellings ("pis" becomes "puis" in Quebec French), so
the transcript is a draft for the user to check, never saved without review.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings

log = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    language: str | None = None
    words: list[Word] | None = None  # only when word timestamps were asked for


class Transcriber(ABC):
    name: str = "base"

    def __init__(self, model: str = "") -> None:
        self.model = model
        self._loaded = False
        self.load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def available(self) -> bool:
        """False when the back-end's package is not installed."""
        return True

    @abstractmethod
    def transcribe(self, wav: Path, language: str | None = None, words: bool = False) -> Transcript:
        """Transcribe ``wav``; ``language`` None means auto-detect.

        ``words`` also fills ``Transcript.words`` with per-word timestamps.
        """

    def unload(self) -> None:
        self._loaded = False

    def info(self) -> dict:
        return {"name": self.name, "model": self.model, "available": self.available, "loaded": self.loaded,
                "load_error": self.load_error}


class FakeTranscriber(Transcriber):
    name = "fake"
    TEXT = "This is a fake transcript of the reference clip."

    def transcribe(self, wav: Path, language: str | None = None, words: bool = False) -> Transcript:
        self._loaded = True
        timed = None
        if words:  # spread the words evenly over the clip
            import soundfile as sf

            info = sf.info(str(wav))
            seconds = info.frames / info.samplerate
            tokens = self.TEXT.split()
            step = seconds / len(tokens)
            timed = [Word(i * step, (i + 1) * step - 0.02, t) for i, t in enumerate(tokens)]
        return Transcript(self.TEXT, language or "en", timed)


class FasterWhisperTranscriber(Transcriber):
    name = "faster-whisper"

    def __init__(self, model: str, device: str = "auto", compute_type: str = "auto") -> None:
        super().__init__(model)
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self, device: str) -> None:
        from faster_whisper import WhisperModel

        started = time.time()
        self._model = WhisperModel(self.model, device=device, compute_type=self.compute_type)
        self._loaded = True
        self.load_error = None
        log.info("STT model %s loaded on %s in %.1fs", self.model, device, time.time() - started)

    def _run(self, wav: Path, language: str | None, words: bool) -> Transcript:
        segments, info = self._model.transcribe(
            str(wav), language=language, beam_size=5, vad_filter=True, word_timestamps=words
        )
        segments = list(segments)  # lazy: decoding happens here
        text = " ".join(s.text.strip() for s in segments)
        timed = None
        if words:
            timed = [Word(float(w.start), float(w.end), w.word.strip())
                     for seg in segments for w in (getattr(seg, "words", None) or [])]
        return Transcript(" ".join(text.split()), getattr(info, "language", language), timed)

    def transcribe(self, wav: Path, language: str | None = None, words: bool = False) -> Transcript:
        with self._lock:
            try:
                if self._model is None:
                    self._load(self.device)
                return self._run(wav, language, words)
            except (RuntimeError, OSError, ValueError) as exc:
                # A missing libcublas/cuDNN often surfaces only on the first decode.
                if self.device == "cpu":
                    self.load_error = f"{type(exc).__name__}: {exc}"
                    raise
                log.warning("STT on %s failed (%s); falling back to the CPU", self.device, exc)
                self.device = "cpu"
                self._model = None
                self._load("cpu")
                return self._run(wav, language, words)

    def unload(self) -> None:
        with self._lock:
            if self._model is not None:
                log.info("Unloading the STT model")
            self._model = None
            self._loaded = False


def create_transcriber(settings: Settings) -> Transcriber | None:
    if settings.stt_engine in {"", "none", "off"}:
        return None
    if settings.stt_engine == "fake":
        return FakeTranscriber("fake")
    if settings.stt_engine == "faster-whisper":
        return FasterWhisperTranscriber(settings.stt_model, settings.stt_device, settings.stt_compute_type)
    raise ValueError(f"Unknown STT_ENGINE '{settings.stt_engine}' (expected 'faster-whisper', 'fake' or 'none')")
