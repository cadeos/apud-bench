"""Publish built pages as static IIIF: Image API 3 Level 0 tiles, Presentation 3 manifests,
and ground-truth annotation pages.

Layout under ``<data>/iiif/<subset>/``::

    manifest.json                 one canvas per page, canvas size = image pixel size
    images/<page>/info.json       Level 0 image service
    images/<page>/<region>/<size>/0/default.jpg
    annotations/<page>.json       ground-truth lines as W3C annotations (xywh on the canvas)

Ground-truth annotation pages are deliberately **not** referenced from the manifest, so a
model that browses the manifest cannot read the answers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

TILE_SIZE = 512
JPEG_QUALITY = 85

IMAGE_CONTEXT = "http://iiif.io/api/image/3/context.json"
PRESENTATION_CONTEXT = "http://iiif.io/api/presentation/3/context.json"


def page_key(xml_path: str) -> str:
    """Stable, URL-safe identifier for a page, derived from its ground-truth path."""
    stem = xml_path.rsplit("/", 1)[-1].rsplit(".xml", 1)[0]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.")[:60] or "page"
    return f"{slug}-{hashlib.sha1(xml_path.encode()).hexdigest()[:8]}"


def scale_factors(width: int, height: int, tile: int = TILE_SIZE) -> list[int]:
    """Powers of two up to the first one at which the whole image fits in a single tile."""
    factors = [1]
    while math.ceil(width / factors[-1]) > tile or math.ceil(height / factors[-1]) > tile:
        factors.append(factors[-1] * 2)
    return factors


def tile_requests(
    width: int, height: int, tile: int = TILE_SIZE
) -> Iterator[tuple[str, str, tuple]]:
    """Yield ``(region, size, crop_box)`` for every tile a Level 0 client may request."""
    for factor in scale_factors(width, height, tile):
        step = tile * factor
        for y in range(0, height, step):
            for x in range(0, width, step):
                w, h = min(step, width - x), min(step, height - y)
                region = "full" if (x, y, w, h) == (0, 0, width, height) else f"{x},{y},{w},{h}"
                size = f"{math.ceil(w / factor)},{math.ceil(h / factor)}"
                yield region, size, (x, y, x + w, y + h)


def level_sizes(width: int, height: int) -> list[tuple[int, int]]:
    """Full-image size at every scale factor, smallest first.

    Viewers such as OpenSeadragon map ``sizes`` one-to-one onto pyramid levels when their
    counts match, so ``sizes`` must be exactly the scale-factor levels or tiles are misplaced.
    """
    factors = scale_factors(width, height)
    return [(math.ceil(width / f), math.ceil(height / f)) for f in reversed(factors)]


def write_image_service(image_path: Path, out_dir: Path, service_id: str) -> dict[str, Any]:
    """Write tiles, sizes and ``full/max`` for one image; return its info.json."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size

    def save(region: str, size: str, picture: Image.Image) -> None:
        target = out_dir / region / size / "0" / "default.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        picture.save(target, "JPEG", quality=JPEG_QUALITY)

    for region, size, box in tile_requests(width, height):
        w, h = (int(n) for n in size.split(","))
        save(region, size, image.crop(box).resize((w, h), Image.Resampling.LANCZOS))
    for w, h in level_sizes(width, height):
        resized = (
            image if (w, h) == (width, height) else image.resize((w, h), Image.Resampling.LANCZOS)
        )
        save("full", f"{w},{h}", resized)
    save("full", "max", image)

    info = {
        "@context": IMAGE_CONTEXT,
        "id": service_id,
        "type": "ImageService3",
        "protocol": "http://iiif.io/api/image",
        "profile": "level0",
        "width": width,
        "height": height,
        "sizes": [{"width": w, "height": h} for w, h in level_sizes(width, height)],
        "tiles": [{"width": TILE_SIZE, "scaleFactors": scale_factors(width, height)}],
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def canvas(page: dict[str, Any], base: str, key: str) -> dict[str, Any]:
    canvas_id = f"{base}/canvas/{key}"
    service_id = f"{base}/images/{key}"
    w, h = page["width"], page["height"]
    return {
        "id": canvas_id,
        "type": "Canvas",
        "label": {"none": [page["xml_path"].rsplit("/", 1)[-1]]},
        "width": w,
        "height": h,
        "items": [
            {
                "id": f"{canvas_id}/page",
                "type": "AnnotationPage",
                "items": [
                    {
                        "id": f"{canvas_id}/page/image",
                        "type": "Annotation",
                        "motivation": "painting",
                        "target": canvas_id,
                        "body": {
                            "id": f"{service_id}/full/max/0/default.jpg",
                            "type": "Image",
                            "format": "image/jpeg",
                            "width": w,
                            "height": h,
                            "service": [
                                {"id": service_id, "type": "ImageService3", "profile": "level0"}
                            ],
                        },
                    }
                ],
            }
        ],
    }


def manifest(
    subset_id: str, subset_name: str, license_id: str, canvases: list[dict[str, Any]], base: str
) -> dict[str, Any]:
    return {
        "@context": PRESENTATION_CONTEXT,
        "id": f"{base}/manifest.json",
        "type": "Manifest",
        "label": {"en": [f"apud-bench: {subset_name}"]},
        "summary": {"en": [f"Pages of subset {subset_id}, rebuilt from the original source."]},
        "requiredStatement": {
            "label": {"en": ["Licence"]},
            "value": {"en": [f"{license_id}. See the subset source for attribution."]},
        },
        "items": canvases,
    }


def ground_truth_page(page: dict[str, Any], base: str, key: str) -> dict[str, Any]:
    canvas_id = f"{base}/canvas/{key}"
    page_id = f"{base}/annotations/{key}.json"
    items = []
    for index, line in enumerate(page["lines"]):
        x, y, w, h = line["bbox"]
        items.append(
            {
                "id": f"{page_id}#line-{index}",
                "type": "Annotation",
                "motivation": "supplementing",
                "body": {"type": "TextualBody", "value": line["text"], "format": "text/plain"},
                "target": {
                    "type": "SpecificResource",
                    "source": {"id": canvas_id, "type": "Canvas"},
                    "selector": [
                        {
                            "type": "FragmentSelector",
                            "conformsTo": "http://www.w3.org/TR/media-frags/",
                            "value": f"xywh={x},{y},{w},{h}",
                        },
                        {"type": "SvgSelector", "value": _svg(line["polygon"])},
                    ],
                },
            }
        )
    return {
        "@context": PRESENTATION_CONTEXT,
        "id": page_id,
        "type": "AnnotationPage",
        "items": items,
    }


def _svg(polygon: list[list[float]]) -> str:
    points = " ".join(f"{x:g},{y:g}" for x, y in polygon)
    return f'<svg xmlns="http://www.w3.org/2000/svg"><polygon points="{points}"/></svg>'


def publish(
    subset_id: str,
    subset_name: str,
    license_id: str,
    pages_file: Path,
    source: Path,
    out: Path,
    base_url: str,
) -> int:
    """Publish every page of a built subset; return the number of canvases written."""
    base = f"{base_url.rstrip('/')}/{subset_id}"
    pages = [json.loads(line) for line in pages_file.read_text(encoding="utf-8").splitlines()]
    canvases = []
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for page in pages:
        key = page_key(page["xml_path"])
        write_image_service(
            source / page["image_path"], out / "images" / key, f"{base}/images/{key}"
        )
        canvases.append(canvas(page, base, key))
        (out / "annotations" / f"{key}.json").write_text(
            json.dumps(ground_truth_page(page, base, key), ensure_ascii=False), encoding="utf-8"
        )
    document = manifest(subset_id, subset_name, license_id, canvases, base)
    (out / "manifest.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return len(canvases)
