"""Metrics: character and word error rates over normalised text, and box IoU.

Text is compared at two normalisation levels, both always reported:

- ``strict``: Unicode NFC and collapsed whitespace. Diplomatic differences count (``ſ`` vs ``s``,
  abbreviation marks, case).
- ``loose``: additionally NFKD with combining marks removed, case folded, and long s / r
  rotunda mapped to their modern forms. Measures reading, not transcription conventions.

Error rates are micro-averaged: total edits over total reference length.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

LEVELS = ("strict", "loose")
_LOOSE_MAP = str.maketrans({"ſ": "s", "ꝛ": "r", "ʒ": "z"})


def normalize(text: str, level: str) -> str:
    text = " ".join(unicodedata.normalize("NFC", text).split())
    if level == "strict":
        return text
    if level == "loose":
        decomposed = unicodedata.normalize("NFKD", text.translate(_LOOSE_MAP))
        stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
        return " ".join(stripped.casefold().split())
    raise ValueError(f"unknown normalisation level {level!r}")


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences (characters or words)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


class ErrorRate:
    """Accumulates edits and reference length for micro-averaged CER or WER."""

    def __init__(self) -> None:
        self.edits = 0
        self.length = 0

    def add(self, reference: Sequence, hypothesis: Sequence) -> None:
        self.edits += edit_distance(reference, hypothesis)
        self.length += len(reference)

    @property
    def value(self) -> float:
        return self.edits / self.length if self.length else 0.0


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``(x, y, w, h)`` boxes; 0 for degenerate boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)
