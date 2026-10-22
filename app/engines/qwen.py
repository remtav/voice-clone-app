"""Qwen3-TTS (Alibaba Qwen team) engine.

Qwen3-TTS is Apache-2.0 (code and weights).  The *Base* checkpoints clone a
voice from a reference clip in two modes:

* **with the clip's transcript** (in-context mode): the model continues the
  reference as if the same person kept talking, so it copies the accent and the
  intonation, not only the timbre.  This is what makes a Quebec French voice
  stay Quebec French.  Give each voice its transcript.
* **without a transcript** (speaker-embedding mode): only the timbre is cloned;
  accent and prosody fall back to the model's defaults.

Unlike Chatterbox, longer references help: 30 to 60 seconds of lively speech
(questions, emphasis, pauses) gives noticeably less flat intonation than a
10-second clip.  Raise ``MAX_REFERENCE_SECONDS`` to 60 for this engine.

Qwen3-TTS needs ``transformers==4.57.3`` while Chatterbox needs 5.x, so it is
built into its own image: ``ENGINE_REQUIREMENTS=requirements-engine-qwen.txt``
(see the README).
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np

from app.engines.base import LANGUAGE_NAMES, Engine, SynthesisParams

log = logging.getLogger(__name__)

# ISO code -> the language name Qwen3-TTS expects.
QWEN_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}
# The codec produces 12 tokens per second; about 14 characters are spoken per second.
TOKENS_PER_SECOND = 12
CHARS_PER_SECOND = 14
# Voice prompts are cheap to keep and slow-ish to build (codec + speaker encoder).
PROMPT_CACHE_SIZE = 16


class QwenEngine(Engine):
    name = "qwen"
    multilingual = True

    def __init__(self, model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device: str = "auto") -> None:
        super().__init__()
        self.model_name = model
        self.variant = Path(model).name
        self.requested_device = device
        self.device = device
        self.languages = {code: LANGUAGE_NAMES[code] for code in QWEN_LANGUAGES}
        self.sample_rate = 24000
        self.model = None
        self.model_id = ""
        self._prompts: dict[tuple, object] = {}

    def _resolve_device(self) -> str:
        import torch

        if self.requested_device in ("", "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.requested_device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("DEVICE=%s requested but CUDA is not available; falling back to CPU", self.requested_device)
            return "cpu"
        return self.requested_device

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from qwen_tts import Qwen3TTSModel

        self.device = self._resolve_device()
        log.info("Loading Qwen3-TTS '%s' on %s ...", self.model_name, self.device)
        source = self.model_name
        if not Path(source).is_dir():
            # The speech tokenizer lives in a sub-folder of the repo: fetch the whole snapshot.
            from huggingface_hub import snapshot_download

            source = snapshot_download(self.model_name)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = Qwen3TTSModel.from_pretrained(source, device_map=self.device, dtype=dtype)
        self.model_id = self.model_name
        self._prompts.clear()
        self._loaded = True
        log.info("Qwen3-TTS loaded (%s)", self.model_id)

    def unload(self) -> None:
        self.model = None
        self._prompts.clear()
        self._loaded = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("Qwen3-TTS unloaded (GPU memory released)")

    def _voice_prompt(self, reference_wav: Path, reference_text: str | None):
        text = (reference_text or "").strip()
        key = (str(reference_wav), reference_wav.stat().st_mtime_ns, text)
        prompt = self._prompts.get(key)
        if prompt is None:
            if not text:
                log.warning("Voice %s has no transcript: cloning the timbre only (accent and intonation "
                            "will not follow the reference)", reference_wav.name)
            assert self.model is not None
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(reference_wav), ref_text=text or None, x_vector_only_mode=not text
            )
            if len(self._prompts) >= PROMPT_CACHE_SIZE:
                self._prompts.pop(next(iter(self._prompts)))
            self._prompts[key] = prompt
        return prompt

    def synthesize(self, text: str, reference_wav: Path, params: SynthesisParams) -> np.ndarray:
        import torch

        if not self._loaded:
            self.load()
        assert self.model is not None

        language = (params.language or "en").lower()
        if language not in QWEN_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'")
        if params.seed is not None:
            torch.manual_seed(int(params.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(params.seed))

        prompt = self._voice_prompt(reference_wav, params.reference_text)
        # Cap the output at 1.6x the expected length so a run-away generation stops early.
        max_new_tokens = int(len(text) / CHARS_PER_SECOND * TOKENS_PER_SECOND * 1.6) + 40
        temperature = float(params.temperature)
        with torch.inference_mode():
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=QWEN_LANGUAGES[language],
                voice_clone_prompt=prompt,
                temperature=temperature,
                subtalker_temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if int(sr) != self.sample_rate:
            from app.audio import resample

            audio = resample(audio, int(sr), self.sample_rate)
        return audio

    def info(self) -> dict:
        data = super().info()
        data.update({"device": self.device, "model_id": self.model_id or self.model_name})
        try:
            import torch

            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                data["gpu"] = {
                    "name": torch.cuda.get_device_name(idx),
                    "memory_allocated_mb": round(torch.cuda.memory_allocated(idx) / 2**20),
                    "memory_reserved_mb": round(torch.cuda.memory_reserved(idx) / 2**20),
                    "memory_total_mb": round(torch.cuda.get_device_properties(idx).total_memory / 2**20),
                }
        except Exception:  # noqa: BLE001 - torch missing or CUDA broken; info is best-effort
            pass
        return data
