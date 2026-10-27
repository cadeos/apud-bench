"""Turn split pages into scored items.

Two tasks per line, each item addressed by IIIF identifiers so a model (or an agent using
apud-mcp) works from the manifest and canvas, never from local files:

- ``transcription``: read the text in a given canvas region (``xywh``);
- ``grounding``: given a line of text, return its ``xywh`` box on the canvas.

Grounding skips lines that cannot be located unambiguously: very short ones and lines whose
text occurs more than once on the same page.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from apud_bench.iiif import page_key
from apud_bench.metrics import normalize

MIN_GROUNDING_CHARS = 5


def items(
    subset_id: str, pages: list[dict[str, Any]], base_url: str, tasks: tuple[str, ...]
) -> list[dict[str, Any]]:
    base = f"{base_url.rstrip('/')}/{subset_id}"
    out: list[dict[str, Any]] = []
    for page in pages:
        key = page_key(page["xml_path"])
        common = {
            "subset": subset_id,
            "manifest": f"{base}/manifest.json",
            "canvas": f"{base}/canvas/{key}",
            "canvas_size": [page["width"], page["height"]],
        }
        texts = Counter(normalize(line["text"], "strict") for line in page["lines"])
        for index, line in enumerate(page["lines"]):
            text = line["text"].strip()
            if not text:
                continue
            item_id = f"{subset_id}/{key}/{index}"
            if "transcription" in tasks:
                out.append(
                    {
                        "id": f"T:{item_id}",
                        "task": "transcription",
                        **common,
                        "region": line["bbox"],
                        "reference": text,
                    }
                )
            unique = texts[normalize(text, "strict")] == 1
            if "grounding" in tasks and unique and len(text) >= MIN_GROUNDING_CHARS:
                out.append(
                    {
                        "id": f"G:{item_id}",
                        "task": "grounding",
                        **common,
                        "query": text,
                        "reference": line["bbox"],
                    }
                )
    return out


def select_pages(pages_file: Path, chosen: list[str]) -> list[dict[str, Any]]:
    wanted = set(chosen)
    pages = [json.loads(line) for line in pages_file.read_text(encoding="utf-8").splitlines()]
    selected = [p for p in pages if p["xml_path"] in wanted]
    missing = wanted - {p["xml_path"] for p in selected}
    if missing:
        raise ValueError(f"split pages missing from build: {sorted(missing)[:3]}")
    return selected
