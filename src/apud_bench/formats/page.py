"""PAGE XML reader (schema versions 2010-03-19 to 2019-07-15).

Handles both coordinate encodings: the ``points="x,y x,y"`` attribute (2013+) and the
``<Point x= y=/>`` children of the 2010 schema.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from apud_bench.formats.model import (
    FormatError,
    Line,
    Page,
    Point,
    parse_points,
    scale_points,
)


def read_page(root: ET.Element, page_id: str, image_size: tuple[int, int] | None) -> Page:
    page = root.find("{*}Page")
    if page is None:
        raise FormatError("PcGts without a Page element")
    declared_w = int(float(page.get("imageWidth", 0)))
    declared_h = int(float(page.get("imageHeight", 0)))
    if not declared_w or not declared_h:
        raise FormatError("Page has no imageWidth/imageHeight")
    width, height = image_size or (declared_w, declared_h)
    sx, sy = width / declared_w, height / declared_h

    lines: list[Line] = []
    seen: set[int] = set()
    for region in page.iterfind(".//{*}TextRegion"):
        for el in region.findall("{*}TextLine"):
            seen.add(id(el))
            if line := _line(el, region.get("id"), sx, sy):
                lines.append(line)
    for el in page.iterfind(".//{*}TextLine"):  # lines outside any TextRegion
        if id(el) not in seen and (line := _line(el, None, sx, sy)):
            lines.append(line)

    namespace = root.tag[1:].split("}")[0] if root.tag.startswith("{") else ""
    version = namespace.rsplit("/", 1)[-1] if "pagecontent" in namespace else ""
    return Page(
        id=page_id,
        image=page.get("imageFilename"),
        width=width,
        height=height,
        lines=tuple(lines),
        source_format=f"page-{version}" if version else "page",
    )


def _line(el: ET.Element, region_id: str | None, sx: float, sy: float) -> Line | None:
    polygon = _coords(el.find("{*}Coords"))
    if not polygon:
        return None
    baseline = _coords(el.find("{*}Baseline"))
    return Line(
        id=el.get("id", ""),
        text=_text(el),
        polygon=scale_points(polygon, sx, sy),
        baseline=scale_points(baseline, sx, sy) if baseline else None,
        region_id=region_id,
    )


def _coords(el: ET.Element | None) -> tuple[Point, ...]:
    if el is None:
        return ()
    if points := el.get("points"):
        return parse_points(points)
    return tuple((float(p.get("x", 0)), float(p.get("y", 0))) for p in el.findall("{*}Point"))


def _text(line: ET.Element) -> str:
    # The line-level TextEquiv with the lowest index is the main reading; fall back to words.
    equivs = line.findall("{*}TextEquiv")
    if equivs:
        best = min(equivs, key=lambda e: int(e.get("index", 0)))
        return (best.findtext("{*}Unicode") or "").strip()
    words = [w.findtext("{*}TextEquiv/{*}Unicode") or "" for w in line.findall("{*}Word")]
    return " ".join(w.strip() for w in words if w.strip())
