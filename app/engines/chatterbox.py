"""Chatterbox (Resemble AI) engine.

Chatterbox is MIT-licensed (code *and* weights), does zero-shot voice cloning
from a few seconds of reference audio, and its multilingual v3 model covers
23 languages.  Every generated file carries Resemble's inaudible PerTh
watermark.

Three variants are exposed via ``CHATTERBOX_MODEL``:

* ``multilingual`` (default): ``ChatterboxMultilingualTTS`` with the v3
  checkpoint, 23 languages.  ``CHATTERBOX_T3_MODEL`` swaps the T3 (text to
  speech-token) checkpoint for ``v2`` or a fine-tuned ``.safetensors`` file,
  such as a regional-accent finetune (see ``docs/finetune-fr-ca-plan.md``).
* ``english``: the original English-only ``ChatterboxTTS``.
* ``turbo``: ``ChatterboxTurboTTS``, English only, faster, supports
  paralinguistic tags such as ``[laugh]`` or ``[chuckle]``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from app.engines.base import LANGUAGE_NAMES, Engine, SynthesisParams

log = logging.getLogger(__name__)

VARIANTS = ("multilingual", "english", "turbo")
# Everything ``ChatterboxMultilingualTTS.from_pretrained`` downloads except the
# T3 checkpoint, which a fine-tuned file replaces.
SHARED_MULTILINGUAL_ASSETS = [
    "ve.pt",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]


class ChatterboxEngine(Engine):
    name = "chatterbox"

    def __init__(
        self,
        variant: str = "multilingual",
        device: str = "auto",
        t3_model: str = "v3",
        t3_path: Path | None = None,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown CHATTERBOX_MODEL '{variant}' (expected one of {', '.join(VARIANTS)})")
        if t3_path is not None and variant != "multilingual":
            raise ValueError("CHATTERBOX_T3_MODEL files only apply to CHATTERBOX_MODEL=multilingual")
        self.variant = variant
        self.t3_model = t3_model
        self.t3_path = t3_path
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
            if self.t3_path is not None:
                self._load_custom_t3()
            else:
                self._load_official_multilingual()
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

    def _load_official_multilingual(self) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        try:
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device, t3_model=self.t3_model)
            self.model_id = f"ResembleAI/chatterbox (multilingual {self.t3_model})"
        except TypeError:
            # Older chatterbox-tts releases (PyPI 0.1.7) predate the checkpoint selector.
            log.warning("Installed chatterbox-tts has no T3 checkpoint selector; using its default (v2)")
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            self.model_id = "ResembleAI/chatterbox (multilingual v2)"

    def _load_custom_t3(self) -> None:
        from chatterbox.mtl_tts import REPO_ID, ChatterboxMultilingualTTS
        from huggingface_hub import snapshot_download

        assert self.t3_path is not None
        if not self.t3_path.is_file():
            raise FileNotFoundError(f"CHATTERBOX_T3_MODEL points to a missing file: {self.t3_path}")
        ckpt_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            revision="main",
            allow_patterns=SHARED_MULTILINGUAL_ASSETS,
            token=os.getenv("HF_TOKEN"),
        )
        # from_local joins ckpt_dir / t3_model, and joining an absolute path yields
        # that path, so the fine-tuned file can live outside the Hugging Face cache.
        # absolute(), not resolve(): from_local requires the ".safetensors" suffix,
        # which a symlink target (e.g. a content-addressed cache blob) may lack.
        self.model = ChatterboxMultilingualTTS.from_local(
            ckpt_dir, self.device, t3_model=str(self.t3_path.absolute())
        )
        self.model_id = f"ResembleAI/chatterbox (multilingual, custom T3 {self.t3_path.name})"

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
        if self.multilingual:
            data["t3_model"] = self.t3_path.name if self.t3_path is not None else self.t3_model
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
