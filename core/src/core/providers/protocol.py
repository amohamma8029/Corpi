"""Provider-neutral call result shared by generation and checking callers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

type JSONSchema = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProviderCallResult:
    output_text: str
    local_request_id: str
    provider_request_id: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thought_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None
