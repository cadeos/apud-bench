"""Readers for HTR ground-truth formats, producing pages in canvas coordinates."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from apud_bench.formats.alto import read_alto
from apud_bench.formats.model import FormatError, Line, Page, Point
from apud_bench.formats.page import read_page

__all__ = ["FormatError", "Header", "Line", "Page", "Point", "load", "read_header"]


@dataclass(frozen=True)
class Header:
    """What a ground-truth file declares about its page image, before reading any line."""

    kind: str  # "page" or "alto"
    image: str | None
    width: float
    height: float


def load(path: str | Path, image_size: tuple[int, int] | None = None) -> Page:
    """Read a PAGE or ALTO file, detected from its root element.

    ``image_size`` is the real ``(width, height)`` of the page image. Pass it whenever it is
    known: coordinates are then rescaled to it, which fixes ground truth produced on resized
    images and is required for ALTO files measured in ``mm10`` or ``inch1200``.
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    kind = _kind(root)
    if kind == "page":
        return read_page(root, path.stem, image_size)
    if kind == "alto":
        return read_alto(root, path.stem, image_size)
    raise FormatError(f"{path.name}: unsupported root element {kind!r}")


def read_header(path: str | Path) -> Header | None:
    """Declared image name and size, or ``None`` if the file is not PAGE or ALTO."""
    root = ET.parse(path).getroot()
    kind = _kind(root)
    if kind == "page":
        page = root.find("{*}Page")
        if page is None:
            return None
        return Header(
            kind,
            page.get("imageFilename"),
            float(page.get("imageWidth", 0)),
            float(page.get("imageHeight", 0)),
        )
    if kind == "alto":
        page = root.find("{*}Layout/{*}Page")
        if page is None:
            return None
        return Header(
            kind,
            root.findtext("{*}Description/{*}sourceImageInformation/{*}fileName"),
            float(page.get("WIDTH", 0)),
            float(page.get("HEIGHT", 0)),
        )
    return None


def _kind(root: ET.Element) -> str:
    name = root.tag.rsplit("}", 1)[-1]
    if name == "PcGts":
        return "page"
    if name.lower() == "alto":
        return "alto"
    return name
