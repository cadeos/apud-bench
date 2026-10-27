"""Common page/line model shared by every ground-truth format.

All geometry is in **canvas coordinates**: pixels of the full-resolution page image, origin at
the top-left corner. The benchmark serves each page image as a IIIF canvas of exactly that size,
so these coordinates can be used directly in ``xywh`` fragments and compared by IoU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

Point = tuple[float, float]


@dataclass(frozen=True)
class Line:
    id: str
    text: str
    polygon: tuple[Point, ...]
    baseline: tuple[Point, ...] | None = None
    region_id: str | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Integer ``(x, y, w, h)`` box enclosing the polygon."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        x0, y0 = math.floor(min(xs)), math.floor(min(ys))
        x1, y1 = math.ceil(max(xs)), math.ceil(max(ys))
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "bbox": list(self.bbox),
            "polygon": [list(p) for p in self.polygon],
            "baseline": [list(p) for p in self.baseline] if self.baseline else None,
            "region_id": self.region_id,
        }


@dataclass(frozen=True)
class Page:
    id: str
    image: str | None
    width: int
    height: int
    lines: tuple[Line, ...]
    source_format: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "source_format": self.source_format,
            "lines": [line.to_dict() for line in self.lines],
        }


class FormatError(ValueError):
    """The file is not a ground-truth format we can read, or lacks what we need."""


def parse_points(value: str) -> tuple[Point, ...]:
    """Parse ``"x,y x,y"`` or ``"x y x y"`` point lists (both occur in PAGE and ALTO files)."""
    numbers = [float(n) for n in value.replace(",", " ").split()]
    if len(numbers) % 2:
        raise FormatError(f"odd number of coordinates in {value[:40]!r}")
    return tuple(zip(numbers[0::2], numbers[1::2], strict=True))


def rect_polygon(x: float, y: float, w: float, h: float) -> tuple[Point, ...]:
    return ((x, y), (x + w, y), (x + w, y + h), (x, y + h))


def scale_points(points: tuple[Point, ...], sx: float, sy: float) -> tuple[Point, ...]:
    if sx == 1 and sy == 1:
        return points
    return tuple((round(x * sx, 2), round(y * sy, 2)) for x, y in points)
