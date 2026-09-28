"""Pydantic request/response schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    password: str = Field(..., max_length=1024)


class GenerateRequest(BaseModel):
    voice_id: str = Field(..., min_length=1, max_length=64)
    text: str = Field(..., min_length=1)
    language: str | None = Field(None, max_length=8)
    exaggeration: float = Field(0.5, ge=0.0, le=2.0, description="Emotion intensity (0.5 = neutral)")
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="Adherence to the reference voice / pacing")
    temperature: float = Field(0.8, ge=0.05, le=2.0, description="Sampling randomness")
    seed: int | None = Field(None, ge=0, le=2**31 - 1, description="Fix for reproducible output")


class VoiceUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    language: str | None = Field(None, max_length=8)
