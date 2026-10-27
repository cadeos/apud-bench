"""Prompts and answer schemas for each evaluation condition.

Boxes are exchanged as ``[x_min, y_min, x_max, y_max]`` in pixels of the image the model was
shown; the runner converts them to and from canvas coordinates.
"""

from __future__ import annotations

from typing import Any

DIPLOMATIC = (
    "Transcribe diplomatically: keep the original spelling, punctuation, abbreviations and "
    "special characters exactly as written (use Unicode/MUFI characters where needed), do not "
    "expand abbreviations, normalise or translate."
)

REGION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}

LINES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["index", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}

BOXES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "found": {"type": "boolean"},
                    "box": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["index", "found", "box"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["boxes"],
    "additionalProperties": False,
}


def _context(languages: list[str]) -> str:
    names = ", ".join(languages)
    return f"The image comes from a historical handwritten document (languages: {names})."


def transcription_region(languages: list[str]) -> str:
    return (
        f"{_context(languages)} It shows a single line of text. "
        f"Transcribe the text of that line. {DIPLOMATIC} "
        'Answer as JSON: {"text": "..."}.'
    )


def transcription_page(languages: list[str], size: tuple[int, int], boxes: list[list[int]]) -> str:
    listing = "\n".join(f"{i}: {box}" for i, box in enumerate(boxes))
    return (
        f"{_context(languages)} The image is {size[0]}x{size[1]} pixels. "
        "Each numbered box below is [x_min, y_min, x_max, y_max] in pixels of this image and "
        f"contains one line of text. Transcribe the line in every box. {DIPLOMATIC}\n\n"
        f"{listing}\n\n"
        'Answer as JSON: {"lines": [{"index": 0, "text": "..."}, ...]} with one entry per box.'
    )


def grounding_page(languages: list[str], size: tuple[int, int], queries: list[str]) -> str:
    listing = "\n".join(f"{i}: {query}" for i, query in enumerate(queries))
    return (
        f"{_context(languages)} The image is {size[0]}x{size[1]} pixels. "
        "Each numbered entry below is the transcription of one line written on this page. "
        "For every entry, find that line and give the tightest box around it as "
        "[x_min, y_min, x_max, y_max] in pixels of this image.\n\n"
        f"{listing}\n\n"
        'Answer as JSON: {"boxes": [{"index": 0, "found": true, "box": [x_min, y_min, x_max, '
        'y_max]}, ...]} with one entry per line; use "found": false and an empty box if the '
        "line is not on the page."
    )
