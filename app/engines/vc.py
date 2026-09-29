"""Speech-to-speech voice conversion engine (sketch).

Why this exists
---------------
Chatterbox (and every text-to-speech model) decides the *accent* from its
language token, not from the reference clip.  Feeding French text with a
``fr`` token yields a France-French accent no matter how Quebec-accented the
reference voice is, because the accent lives in the model's learned phonetics.

Voice conversion sidesteps this entirely.  You *speak* the sentence yourself
(or a Quebec speaker does), and the model keeps that recording's linguistic
content, timing, rhythm and accent while replacing only the timbre with the
target voice.  The accent is preserved because it never passes through a
text->phoneme step.

Concrete backend
----------------
This skeleton is written around **seed-vc** (Plachtaa, MIT), a zero-shot
converter that needs only one short target reference clip -- the exact same
kind of clip the app already stores as a "voice".  Other drop-in options,
should you prefer them, expose a similar "(source, target) -> wav" call:

* **kNN-VC** -- WavLM features + kNN matching; strongest source-prosody
  (accent) preservation, wants ~1 min of target audio.
* **OpenVoice v2** (MyShell, MIT) -- explicitly separates tone colour from
  style, so it preserves source accent/rhythm by design.

Only the small ``_load_model`` / ``_run_model`` section below is
backend-specific; swap it to change engines.

Status: SKETCH.  ``load``/``convert`` raise ``NotImplementedError`` until a
backend is vendored (see ``requirements-engine.txt`` note and
``docs/voice-conversion.md``).  The class is import-safe with no extra deps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.engines.base import ConversionParams, Engine, SynthesisParams

log = logging.getLogger(__name__)


class VoiceConversionEngine(Engine):
    name = "vc"
    variant = "seed-vc"
    supports_conversion = True
    # VC is speech-to-speech, so there is no language token and no accent
    # imposed by the model -- accent comes from the source recording.
    multilingual = False
    languages = {"en": "Any (accent taken from the source audio)"}

    def __init__(self, model: str = "seed-vc", device: str = "auto") -> None:
        super().__init__()
        self.requested_device = device
        self.device = device
        self.model_name = model
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
        log.info("Loading voice-conversion model '%s' on %s ...", self.model_name, self.device)

        # ---- backend-specific: load weights here --------------------------
        # Example (seed-vc), once the package is vendored in
        # requirements-engine.txt:
        #
        #   from seed_vc import SeedVC
        #   self.model = SeedVC.from_pretrained(device=self.device)
        #   self.sample_rate = int(self.model.sr)
        #   self.model_id = "Plachtaa/seed-vc"
        #
        # kNN-VC alternative:
        #   import torch
        #   self.model = torch.hub.load("bshall/knn-vc", "knn_vc",
        #                               prematched=True, trust_repo=True,
        #                               device=self.device)
        #   self.sample_rate = 16000
        # -------------------------------------------------------------------
        raise NotImplementedError(
            "VoiceConversionEngine is a sketch: vendor a backend (seed-vc / kNN-VC / "
            "OpenVoice) and fill in load()/_run_model(). See docs/voice-conversion.md."
        )

    # ------------------------------------------------------------------
    def synthesize(self, text: str, reference_wav: Path, params: SynthesisParams) -> np.ndarray:
        # A pure VC engine cannot produce speech from text on its own; it needs
        # a spoken source.  The worker never calls this for a "convert" job, but
        # we fail loudly in case someone points a TTS job at this engine.
        raise NotImplementedError(
            "The 'vc' engine does voice conversion (speech-to-speech), not "
            "text-to-speech. Submit a source recording via the convert flow, "
            "or use a TTS engine (chatterbox) for text input."
        )

    def convert(self, source_wav: Path, reference_wav: Path, params: ConversionParams) -> np.ndarray:
        import torch

        if not self._loaded:
            self.load()
        assert self.model is not None

        if params.seed is not None:
            torch.manual_seed(int(params.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(params.seed))

        with torch.inference_mode():
            audio = self._run_model(source_wav, reference_wav, params)

        return np.asarray(audio, dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------------
    def _run_model(self, source_wav: Path, reference_wav: Path, params: ConversionParams) -> np.ndarray:
        """Backend-specific inference. Return mono float32 at self.sample_rate.

        Example (seed-vc)::

            wav = self.model.convert(
                source=str(source_wav),         # the Quebec-accented take
                target=str(reference_wav),      # the target voice's timbre
                diffusion_steps=params.steps,
                length_adjust=1.0,
                inference_cfg_rate=params.strength,
                pitch_shift=params.pitch_shift,
            )
            return wav.detach().cpu().float().numpy().reshape(-1)
        """
        raise NotImplementedError

    def info(self) -> dict:
        data = super().info()
        data.update({"device": self.device, "model_id": self.model_id})
        return data
