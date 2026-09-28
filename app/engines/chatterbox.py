"""Chatterbox (Resemble AI) engine.

Chatterbox is MIT-licensed (code *and* weights), does zero-shot voice cloning
from a few seconds of reference audio, and its multilingual v3 model covers
23 languages.  Every generated file carries Resemble's inaudible PerTh
watermark.

Three variants are exposed via ``CHATTERBOX_MODEL``:

* ``multilingual`` (default): ``ChatterboxMultilingualTTS`` with the v3
  checkpoint, 23 languages.
* ``english``: the original English-only ``ChatterboxTTS``.
* ``turbo``: ``ChatterboxTurboTTS``, English only, faster, supports
  paralinguistic tags such as ``[laugh]`` or ``[chuckle]``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.engines.base import LANGUAGE_NAMES, Engine, SynthesisParams

log = logging.getLogger(__name__)

VARIANTS = ("multilingual", "english", "turbo")


class ChatterboxEngine(Engine):
    name = "chatterbox"

    def __init__(self, variant: str = "multilingual", device: str = "auto") -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown CHATTERBOX_MODEL '{variant}' (expected one of {', '.join(VARIANTS)})")
        self.variant = variant
        self.requested_device = device
        self.device = device
        self.multilingual = variant == "multilingual"
        self.languages = dict(LANGUAGE_NAMES) if self.multilingual else {"en": "English"}
        self.model = None
        self.model_id = ""

    # ------------------------------------------------------------------
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
        self.device = self._resolve_device()
        log.info("Loading Chatterbox '%s' on %s ...", self.variant, self.device)

        if self.variant == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            try:
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device, t3_model="v3")
                self.model_id = "ResembleAI/chatterbox (multilingual v3)"
            except TypeError:
                # Older chatterbox-tts releases (PyPI 0.1.7) predate the v3 selector.
                log.warning("Installed chatterbox-tts has no v3 multilingual checkpoint support; using v2")
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                self.model_id = "ResembleAI/chatterbox (multilingual v2)"
        elif self.variant == "english":
            from chatterbox.tts import ChatterboxTTS

            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.model_id = "ResembleAI/chatterbox (english)"
        else:
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            self.model_id = "ResembleAI/chatterbox-turbo"

        self.sample_rate = int(getattr(self.model, "sr", 24000))
        self._loaded = True
        log.info("Chatterbox loaded (%s, sr=%d)", self.model_id, self.sample_rate)

    # ------------------------------------------------------------------
    def synthesize(self, text: str, reference_wav: Path, params: SynthesisParams) -> np.ndarray:
        import torch

        if not self._loaded:
            self.load()
        assert self.model is not None

        if params.seed is not None:
            torch.manual_seed(int(params.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(params.seed))

        kwargs = {
            "audio_prompt_path": str(reference_wav),
            "exaggeration": float(params.exaggeration),
            "cfg_weight": float(params.cfg_weight),
            "temperature": float(params.temperature),
        }
        if self.multilingual:
            language = (params.language or "en").lower()
            if language not in self.languages:
                raise ValueError(f"Unsupported language '{language}'")
            kwargs["language_id"] = language

        with torch.inference_mode():
            wav = self.model.generate(text, **kwargs)

        audio = wav.detach().cpu().float().numpy().reshape(-1)
        return audio.astype(np.float32)

    def info(self) -> dict:
        data = super().info()
        data.update({"device": self.device, "model_id": self.model_id})
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
