from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ProviderResponse:
    success: bool
    model_id: str
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    status_code: int | None = None
    error: str | None = None
    raw_json: dict[str, Any] = field(default_factory=dict)
