"""Run a model on task items under one evaluation condition.

Conditions:

- ``transcription:region``: one call per line, the model sees only the IIIF crop of the line;
- ``transcription:page``: the model sees the whole page and a numbered list of line boxes;
- ``grounding:page``: the model sees the whole page and a numbered list of line texts.

Page conditions batch up to ``group_size`` lines of the same canvas into one call. Images are
always fetched through IIIF with ``apud``, exactly as an agent would. Results are appended to
``predictions.jsonl`` and ``calls.jsonl`` in the output directory, so an interrupted run resumes
where it stopped and never pays twice for the same item.
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apud.client import IIIFClient
from apud.iiif import Canvas, ImageService, Region, canvas_to_image

from apud_bench.runners import prompts
from apud_bench.runners.base import Provider, Request, Response

CONDITIONS = ("transcription:region", "transcription:page", "grounding:page")


@dataclass
class Job:
    id: str
    condition: str
    subset: str
    manifest: str
    canvas: str
    items: list[dict[str, Any]]


def plan(
    items: list[dict[str, Any]],
    condition: str,
    group_size: int = 10,
    pages_per_subset: int | None = None,
) -> list[Job]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    task = condition.split(":")[0]
    by_canvas: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for item in items:
        if item["task"] == task:
            by_canvas.setdefault((item["subset"], item["canvas"]), []).append(item)
    pages_seen: dict[str, int] = {}
    jobs: list[Job] = []
    for (subset, canvas), canvas_items in by_canvas.items():
        pages_seen[subset] = pages_seen.get(subset, 0) + 1
        if pages_per_subset is not None and pages_seen[subset] > pages_per_subset:
            continue
        manifest = canvas_items[0]["manifest"]
        size = 1 if condition == "transcription:region" else group_size
        for start in range(0, len(canvas_items), size):
            chunk = canvas_items[start : start + size]
            jobs.append(
                Job(f"{condition}|{chunk[0]['id']}", condition, subset, manifest, canvas, chunk)
            )
    return jobs


class ImageSource:
    """Fetches page and region images through IIIF, caching manifests and image services."""

    def __init__(self, client: IIIFClient | None = None) -> None:
        self._client = client or IIIFClient()
        self._lock = threading.Lock()
        self._canvases: dict[str, Canvas] = {}
        self._services: dict[str, ImageService] = {}

    def _resolve(self, manifest: str, canvas_id: str) -> tuple[Canvas, ImageService]:
        with self._lock:
            if canvas_id not in self._canvases:
                for canvas in self._client.manifest(manifest).canvases:
                    self._canvases[canvas.id] = canvas
            canvas = self._canvases[canvas_id]
            if canvas.image_service is None:
                raise ValueError(f"canvas {canvas_id} has no image service")
            if canvas.image_service not in self._services:
                self._services[canvas.image_service] = self._client.image_service(
                    canvas.image_service
                )
            return canvas, self._services[canvas.image_service]

    def fetch(
        self, manifest: str, canvas_id: str, region: list[int] | None, max_side: int
    ) -> tuple[bytes, tuple[int, int], float]:
        """Return JPEG bytes, their size, and canvas units per returned pixel."""
        canvas, service = self._resolve(manifest, canvas_id)
        x, y, w, h = region or (0, 0, canvas.width, canvas.height)
        image_region = canvas_to_image(Region(x, y, w, h), service.width, service.height, canvas)
        request, data = self._client.region(service, image_region, max_side)
        canvas_per_px = (canvas.width / service.width) / request.scale
        return data, (request.width, request.height), canvas_per_px


def build_request(
    job: Job, source: ImageSource, max_side: int, languages: list[str]
) -> tuple[Request, float]:
    if job.condition == "transcription:region":
        item = job.items[0]
        image, _, scale = source.fetch(job.manifest, job.canvas, item["region"], max_side)
        return Request(
            (image,), prompts.transcription_region(languages), prompts.REGION_SCHEMA
        ), scale
    image, size, scale = source.fetch(job.manifest, job.canvas, None, max_side)
    if job.condition == "transcription:page":
        boxes = [_to_pixels(item["region"], scale) for item in job.items]
        prompt = prompts.transcription_page(languages, size, boxes)
        return Request((image,), prompt, prompts.LINES_SCHEMA), scale
    queries = [item["query"] for item in job.items]
    prompt = prompts.grounding_page(languages, size, queries)
    return Request((image,), prompt, prompts.BOXES_SCHEMA), scale


def parse(job: Job, response: Response, canvas_per_px: float) -> list[dict[str, Any]]:
    """Turn a model answer into predictions; items the answer does not cover get none."""
    data = response.data or {}
    if job.condition == "transcription:region":
        text = data.get("text")
        return [{"id": job.items[0]["id"], "text": text}] if isinstance(text, str) else []
    if job.condition == "transcription:page":
        out = []
        for entry in data.get("lines", []):
            index, text = entry.get("index"), entry.get("text")
            if isinstance(index, int) and 0 <= index < len(job.items) and isinstance(text, str):
                out.append({"id": job.items[index]["id"], "text": text})
        return out
    out = []
    for entry in data.get("boxes", []):
        index, box = entry.get("index"), entry.get("box")
        if not (isinstance(index, int) and 0 <= index < len(job.items)) or not entry.get("found"):
            continue
        if not (
            isinstance(box, list) and len(box) == 4 and all(isinstance(v, int | float) for v in box)
        ):
            continue
        x0, x1 = sorted((box[0], box[2]))
        y0, y1 = sorted((box[1], box[3]))
        bbox = [round(v * canvas_per_px) for v in (x0, y0, x1 - x0, y1 - y0)]
        out.append({"id": job.items[index]["id"], "bbox": bbox})
    return out


def _to_pixels(region: list[int], canvas_per_px: float) -> list[int]:
    x, y, w, h = region
    return [round(v / canvas_per_px) for v in (x, y, x + w, y + h)]


def estimate_image_tokens(size: tuple[int, int]) -> int:
    """Claude's image token estimate for models reading up to 2576 px (28 px patches)."""
    return math.ceil(size[0] / 28) * math.ceil(size[1] / 28)


def run(
    provider: Provider,
    jobs: list[Job],
    source: ImageSource,
    out_dir: Path,
    languages: dict[str, list[str]],
    workers: int = 4,
    dry_run: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = out_dir / "predictions.jsonl"
    done = set()
    if predictions_file.exists():
        done = {json.loads(line)["id"] for line in predictions_file.open(encoding="utf-8")}
    calls_file = out_dir / "calls.jsonl"
    if calls_file.exists() and not dry_run:
        # Items the model already answered without a usable prediction are not retried either.
        for line in calls_file.open(encoding="utf-8"):
            call = json.loads(line)
            if call.get("error") is None:
                done.update(call["items"])
    pending = [job for job in jobs if any(item["id"] not in done for item in job.items)]

    lock = threading.Lock()
    summary = {
        "jobs": len(pending),
        "skipped_jobs": len(jobs) - len(pending),
        "calls": 0,
        "errors": 0,
        "predictions": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_image_tokens": 0,
        "estimated_prompt_tokens": 0,
    }

    def work(job: Job) -> None:
        record: dict[str, Any] = {
            "job": job.id,
            "items": [i["id"] for i in job.items],
            "error": None,
        }
        predictions: list[dict[str, Any]] = []
        try:
            request, scale = build_request(job, source, provider.max_side, languages[job.subset])
            sizes = [_jpeg_size(image) for image in request.images]
            record["image_sizes"] = sizes
            if dry_run:
                record["estimated_image_tokens"] = sum(estimate_image_tokens(s) for s in sizes)
                record["estimated_prompt_tokens"] = len(request.prompt) // 3
            else:
                response = provider.complete(request)
                predictions = parse(job, response, scale)
                record.update(
                    stop_reason=response.stop_reason,
                    usage=response.usage,
                    model=response.model,
                    raw=response.raw[:4000],
                )
        except Exception as error:  # recorded and retried on the next run
            record["error"] = f"{type(error).__name__}: {error}"
        with lock:
            summary["calls"] += 1
            summary["errors"] += record["error"] is not None
            summary["estimated_image_tokens"] += record.get("estimated_image_tokens", 0)
            summary["estimated_prompt_tokens"] += record.get("estimated_prompt_tokens", 0)
            summary["input_tokens"] += record.get("usage", {}).get("input_tokens", 0)
            summary["output_tokens"] += record.get("usage", {}).get("output_tokens", 0)
            if dry_run:
                return
            with calls_file.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            with predictions_file.open("a", encoding="utf-8") as out:
                for prediction in predictions:
                    out.write(
                        json.dumps(
                            {**prediction, "model": provider.model, "condition": job.condition},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            summary["predictions"] += len(predictions)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(work, job) for job in pending]):
            future.result()
    return summary


def _jpeg_size(data: bytes) -> tuple[int, int]:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return image.size
