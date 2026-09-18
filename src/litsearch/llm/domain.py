from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMHealthStatus(BaseModel):
    available: bool
    provider: str
    model: str | None = None
    endpoint: str | None = None
    latency_ms: float | None = None
    error: str | None = None