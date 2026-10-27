"""Provider-neutral request/response types for model runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Request:
    """One model call: images (JPEG bytes) followed by a text prompt, answered as JSON."""

    images: tuple[bytes, ...]
    prompt: str
    schema: dict[str, Any]


@dataclass
class Response:
    data: dict[str, Any] | None  # parsed JSON answer, None if refused or unparsable
    raw: str
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    model: str | None = None


class Provider(Protocol):
    name: str
    model: str
    max_side: int  # longest image side the model reads without downscaling

    def complete(self, request: Request) -> Response: ...
