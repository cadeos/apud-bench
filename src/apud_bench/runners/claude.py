"""Claude provider, through the official Anthropic SDK.

No refusal fallback is configured on purpose: a fallback would answer with a different model
and contaminate the score of the model under test. Refusals are recorded as such.
"""

from __future__ import annotations

import base64
import json

import anthropic

from apud_bench.runners.base import Request, Response

DEFAULT_MODEL = "claude-opus-5-5"
# Claude Opus 4.7 and later read images up to a 2576 px long edge without downscaling.
MAX_SIDE = 2576


class ClaudeProvider:
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium") -> None:
        self.model = model
        self.effort = effort
        self.max_side = MAX_SIDE
        self._client = anthropic.Anthropic(max_retries=5)

    def complete(self, request: Request) -> Response:
        content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(image).decode("ascii"),
                },
            }
            for image in request.images
        ]
        content.append({"type": "text", "text": request.prompt})
        message = self._client.messages.create(
            model=self.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": content}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": request.schema},
            },
        )
        text = next((b.text for b in message.content if b.type == "text"), "")
        usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }
        data = None
        if message.stop_reason != "refusal" and text:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
        return Response(data, text, message.stop_reason, usage, message.model)
