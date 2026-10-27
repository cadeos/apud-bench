"""Readers for HTR ground-truth formats, producing pages in canvas coordinates."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from apud_bench.formats.alto import read_alto
from apud_bench.formats.model import FormatError, Line, Page, Point
from apud_bench.formats.page import read_page

__all__ = ["FormatError", "Line", "Page", "Point", "load"]


def load(path: str | Path, image_size: tuple[int, int] | None = None) -> Page:
    """Read a PAGE or ALTO file, detected from its root element.

    ``image_size`` is the real ``(width, height)`` of the page image. Pass it whenever it is
    known: coordinates are then rescaled to it, which fixes ground truth produced on resized
    images and is required for ALTO files measured in ``mm10`` or ``inch1200``.
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    name = root.tag.rsplit("}", 1)[-1]
    if name == "PcGts":
        return read_page(root, path.stem, image_size)
    if name.lower() == "alto":
        return read_alto(root, path.stem, image_size)
    raise FormatError(f"{path.name}: unsupported root element {name!r}")
