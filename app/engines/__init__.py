"""TTS engine registry."""

from __future__ import annotations

from app.config import Settings
from app.engines.base import Engine, SynthesisParams


def create_engine(settings: Settings) -> Engine:
    if settings.engine == "chatterbox":
        from app.engines.chatterbox import ChatterboxEngine

        return ChatterboxEngine(
            variant=settings.chatterbox_model,
            device=settings.device,
            t3_model=settings.chatterbox_t3_model,
            t3_path=settings.chatterbox_t3_path,
        )
    if settings.engine == "qwen":
        from app.engines.qwen import QwenEngine

        return QwenEngine(model=settings.qwen_model, device=settings.device)
    if settings.engine == "vc":
        from app.engines.vc import VoiceConversionEngine

        return VoiceConversionEngine(device=settings.device)
    if settings.engine == "fake":
        from app.engines.fake import FakeEngine

        return FakeEngine()
    raise ValueError(f"Unknown TTS_ENGINE '{settings.engine}' (expected 'chatterbox', 'qwen', 'vc' or 'fake')")


__all__ = ["Engine", "SynthesisParams", "create_engine"]
