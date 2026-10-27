"""ALTO XML reader (v2 to v4).

Covers the variants found in HTR ground truth:

- units ``pixel`` (eScriptorium, Kraken) and ``mm10`` / ``inch1200`` (Transkribus exports);
  non-pixel units need the real image size to be converted;
- line geometry as ``Shape/Polygon`` (``"x,y x,y"`` or ``"x y x y"``) or only as the
  ``HPOS/VPOS/WIDTH/HEIGHT`` rectangle;
- ``BASELINE`` as a point list (ALTO 4.2, eScriptorium) or as a single vertical position.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from apud_bench.formats.model import (
    FormatError,
    Line,
    Page,
    Point,
    parse_points,
    rect_polygon,
    scale_points,
)

PIXEL_UNITS = {"pixel", ""}


def read_alto(root: ET.Element, page_id: str, image_size: tuple[int, int] | None) -> Page:
    page = root.find("{*}Layout/{*}Page")
    if page is None:
        raise FormatError("ALTO without Layout/Page")
    unit = (root.findtext("{*}Description/{*}MeasurementUnit") or "").strip()
    declared_w = float(page.get("WIDTH", 0))
    declared_h = float(page.get("HEIGHT", 0))
    if not declared_w or not declared_h:
        raise FormatError("ALTO Page has no WIDTH/HEIGHT")
    if unit not in PIXEL_UNITS and image_size is None:
        raise FormatError(f"ALTO measured in {unit!r}: the page image size is required")
    width, height = image_size or (round(declared_w), round(declared_h))
    sx, sy = width / declared_w, height / declared_h

    lines: list[Line] = []
    seen: set[int] = set()
    for block in page.iterfind(".//{*}TextBlock"):
        for el in block.findall("{*}TextLine"):
            seen.add(id(el))
            if line := _line(el, block.get("ID"), sx, sy):
                lines.append(line)
    for el in page.iterfind(".//{*}TextLine"):
        if id(el) not in seen and (line := _line(el, None, sx, sy)):
            lines.append(line)

    namespace = root.tag[1:].split("}")[0] if root.tag.startswith("{") else ""
    version = namespace.rsplit("ns-", 1)[-1].rstrip("#/") if "ns-" in namespace else ""
    return Page(
        id=page_id,
        image=root.findtext("{*}Description/{*}sourceImageInformation/{*}fileName"),
        width=width,
        height=height,
        lines=tuple(lines),
        source_format=f"alto-{version}" if version else "alto",
    )


def _line(el: ET.Element, block_id: str | None, sx: float, sy: float) -> Line | None:
    rect = _rect(el)
    polygon_el = el.find("{*}Shape/{*}Polygon")
    if polygon_el is not None and polygon_el.get("POINTS"):
        polygon = parse_points(polygon_el.get("POINTS", ""))
    elif rect is not None:
        polygon = rect_polygon(*rect)
    else:
        return None
    baseline = _baseline(el.get("BASELINE"), rect)
    return Line(
        id=el.get("ID", ""),
        text=_text(el),
        polygon=scale_points(polygon, sx, sy),
        baseline=scale_points(baseline, sx, sy) if baseline else None,
        region_id=block_id,
    )


def _rect(el: ET.Element) -> tuple[float, float, float, float] | None:
    try:
        return tuple(float(el.get(k, "")) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT"))  # type: ignore[return-value]
    except ValueError:
        return None


def _baseline(
    value: str | None, rect: tuple[float, float, float, float] | None
) -> tuple[Point, ...] | None:
    if not value:
        return None
    numbers = value.replace(",", " ").split()
    if len(numbers) >= 4:
        return parse_points(value)
    if len(numbers) == 1 and rect is not None:
        # Single number: ALTO <= 4.1 vertical position. Producers disagree on whether it is
        # absolute or relative to VPOS; treat it as absolute when it falls inside the line box.
        x, y, w, h = rect
        level = float(numbers[0])
        if not y <= level <= y + h:
            level += y
        return ((x, level), (x + w, level))
    return None


def _text(line: ET.Element) -> str:
    parts: list[str] = []
    previous = None
    for child in line:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "String":
            if previous == "String":
                parts.append(" ")
            parts.append(child.get("CONTENT", ""))
        elif tag == "SP":
            parts.append(" ")
        elif tag == "HYP":
            parts.append(child.get("CONTENT", ""))
        previous = tag
    return "".join(parts).strip()
