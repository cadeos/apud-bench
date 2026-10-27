"""Frozen evaluation splits.

apud-bench is evaluation-only: a split is the list of pages of a subset that are scored. It is
drawn deterministically, spread across manuscripts so that no single codex dominates, and
stored in the repository together with the source pin it was drawn from. A split is only valid
against that exact pin; rebuilding from different data is an error, not a silent change.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SPLIT_VERSION = "v0"


class SplitError(RuntimeError):
    pass


def splits_dir() -> Path:
    return Path(__file__).parents[2] / "splits"


def document_of(xml_path: str) -> str:
    """Group key for a page: its folder, without the archive's top-level directory."""
    folder = xml_path.rsplit("/", 1)[0] if "/" in xml_path else ""
    return folder.split("/", 1)[1] if "/" in folder else folder


def draw(xml_paths: list[str], pages: int, seed: int = 0) -> list[str]:
    """Pick up to ``pages`` pages round-robin across documents, in a seeded stable order."""

    def rank(path: str) -> str:
        return hashlib.sha256(f"{seed}:{path}".encode()).hexdigest()

    by_document: dict[str, list[str]] = defaultdict(list)
    for path in xml_paths:
        by_document[document_of(path)].append(path)
    queues = [sorted(paths, key=rank) for _, paths in sorted(by_document.items())]
    chosen: list[str] = []
    while len(chosen) < pages and any(queues):
        for queue in queues:
            if queue and len(chosen) < pages:
                chosen.append(queue.pop(0))
    return sorted(chosen)


def write(
    subset_id: str,
    pin: dict[str, Any],
    chosen: list[str],
    seed: int,
    directory: Path | None = None,
) -> Path:
    path = (directory or splits_dir()) / SPLIT_VERSION / f"{subset_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "subset": subset_id,
        "version": SPLIT_VERSION,
        "seed": seed,
        "source": pin,
        "pages": chosen,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read(subset_id: str, pin: dict[str, Any], directory: Path | None = None) -> list[str]:
    path = (directory or splits_dir()) / SPLIT_VERSION / f"{subset_id}.json"
    if not path.exists():
        raise SplitError(f"{subset_id}: no {SPLIT_VERSION} split, run `apud-bench split` first")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["source"] != pin:
        raise SplitError(f"{subset_id}: split was drawn from a different source pin")
    return document["pages"]
