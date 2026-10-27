"""Provider for OpenAI and OpenAI-compatible endpoints (OpenRouter, vLLM, Gemini's
compatibility endpoint), through the official OpenAI SDK.
"""

from __future__ import annotations

import base64
import json
import os

import openai

from apud_bench.runners.base import Request, Response


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_side: int = 2048,
    ) -> None:
        self.model = model
        self.max_side = max_side
        self._client = openai.OpenAI(
            base_url=base_url, api_key=os.environ.get(api_key_env), max_retries=5
        )

    def complete(self, request: Request) -> Response:
        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.standard_b64encode(image).decode("ascii")
                },
            }
            for image in request.images
        ]
        content.append({"type": "text", "text": request.prompt})
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": request.schema, "strict": True},
            },
        )
        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = {}
        if completion.usage:
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            }
        data = None
        if text:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
        return Response(data, text, choice.finish_reason, usage, completion.model)
