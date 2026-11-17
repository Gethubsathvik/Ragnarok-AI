"""Pydantic v2 request/response models for the web API."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(default="", max_length=8000)
    max_new_tokens: int = Field(default=200, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="0 = greedy")
    top_k: int = Field(default=50, ge=0, le=500, description="0 = off")
    top_p: float = Field(default=0.95, gt=0.0, le=1.0, description="1.0 = off")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A bare file name inside models/ (e.g. "best.pt"). Paths are rejected on purpose.
    name: str = Field(min_length=1, max_length=128, pattern=r"^[\w.\-]+\.pt$")


class ModelInfo(BaseModel):
    loaded: bool
    checkpoint: Optional[str] = None
    step: Optional[int] = None
    params_millions: Optional[float] = None
    device: Optional[str] = None
    cuda_available: bool = False
    gpu_name: Optional[str] = None
    busy: bool = False


class CheckpointList(BaseModel):
    checkpoints: list[str]
