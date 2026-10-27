"""Turn an extracted source into normalised pages: pair each PAGE/ALTO file with its image,
read it in canvas coordinates of the real image, and record everything that was skipped.

Output per subset, under ``<data>/build/<subset>/``:

- ``pages.jsonl``: one page per line, with paths relative to the source directory;
- ``report.json``: counts, skip reasons with examples, licence and source pin.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from apud_bench import formats
from apud_bench.registry import Subset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
# Ground truth drawn on a resized image is rescaled; a different aspect ratio means the image
# was cropped or is the wrong one, and rescaling would silently misplace every line.
MAX_ASPECT_DRIFT = 0.02

# Images are only opened to read their size, never decoded, so the decompression-bomb guard
# (which rejects legitimate gigapixel manuscript scans) is not needed here.
Image.MAX_IMAGE_PIXELS = None


@dataclass
class Report:
    subset: str
    license: str
    pin: dict[str, Any]
    pages: int = 0
    lines: int = 0
    empty_lines: int = 0
    lines_out_of_bounds: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def skip(self, reason: str, path: str) -> None:
        self.skipped[reason] += 1
        if len(self.examples[reason]) < 5:
            self.examples[reason].append(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "license": self.license,
            "source": self.pin,
            "pages": self.pages,
            "lines": self.lines,
            "empty_lines": self.empty_lines,
            "lines_out_of_bounds": self.lines_out_of_bounds,
            "skipped": dict(self.skipped),
            "skipped_examples": dict(self.examples),
        }


def build(subset: Subset, pin: dict[str, Any], source: Path, out: Path) -> Report:
    report = Report(subset.id, subset.license, pin)
    images = _index_images(source)
    used_images: set[Path] = set()
    out.mkdir(parents=True, exist_ok=True)

    with (out / "pages.jsonl").open("w", encoding="utf-8") as sink:
        for xml in sorted(source.rglob((subset.build or {}).get("xml_glob", "*.xml"))):
            rel = xml.relative_to(source).as_posix()
            try:
                header = formats.read_header(xml)
            except ET.ParseError:
                report.skip("xml-parse-error", rel)
                continue
            if header is None:
                continue  # METS, metadata and other non ground-truth XML
            image = _match_image(xml, header.image, images)
            if image is None:
                report.skip("no-image", rel)
                continue
            if image in used_images:
                report.skip("duplicate-image", rel)
                continue
            try:
                with Image.open(image) as im:
                    size = im.size
            except (UnidentifiedImageError, OSError):
                report.skip("image-unreadable", rel)
                continue
            if header.width and header.height:
                drift = abs((header.width / header.height) / (size[0] / size[1]) - 1)
                if drift > MAX_ASPECT_DRIFT:
                    report.skip("aspect-mismatch", rel)
                    continue
            try:
                page = formats.load(xml, size)
            except formats.FormatError as error:
                report.skip(f"format-error: {error}", rel)
                continue
            if not page.lines:
                report.skip("no-lines", rel)
                continue

            used_images.add(image)
            report.pages += 1
            report.lines += len(page.lines)
            report.empty_lines += sum(not line.text for line in page.lines)
            report.lines_out_of_bounds += sum(
                not _inside(line.bbox, page.width, page.height) for line in page.lines
            )
            record = {
                "subset": subset.id,
                "xml_path": rel,
                "image_path": image.relative_to(source).as_posix(),
                **page.to_dict(),
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    (out / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _index_images(source: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in source.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            index[path.name.lower()].append(path)
            index[path.stem.lower()].append(path)
    return index


def _match_image(xml: Path, declared: str | None, images: dict[str, list[Path]]) -> Path | None:
    """Prefer the declared file name, then the XML stem; among equals, the nearest file."""
    keys = []
    if declared:
        name = declared.replace("\\", "/").rsplit("/", 1)[-1].lower()
        keys += [name, name.rsplit(".", 1)[0]]
    keys.append(xml.stem.lower())
    for key in keys:
        if candidates := images.get(key):
            return max(candidates, key=lambda p: _shared_parts(p, xml))
    return None


def _shared_parts(a: Path, b: Path) -> int:
    count = 0
    for x, y in zip(a.parent.parts, b.parent.parts, strict=False):
        if x != y:
            break
        count += 1
    return count


def _inside(bbox: tuple[int, int, int, int], width: int, height: int, tolerance: int = 2) -> bool:
    x, y, w, h = bbox
    return (
        x >= -tolerance
        and y >= -tolerance
        and x + w <= width + tolerance
        and (y + h <= height + tolerance)
    )
