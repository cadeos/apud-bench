import io
import json
import re
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

pytest.importorskip("apud")
pytest.importorskip("anthropic")

from apud.client import IIIFClient  # noqa: E402

from apud_bench import evaluate, iiif, tasks  # noqa: E402
from apud_bench.runners import run as runner  # noqa: E402
from apud_bench.runners.base import Request, Response  # noqa: E402
from apud_bench.runners.claude import ClaudeProvider  # noqa: E402

BASE = "http://bench.test"
LINES = ["In principio erat verbum", "et verbum erat apud Deum", "et Deus erat verbum"]


@pytest.fixture
def bench(tmp_path):
    """A published one-page subset served from disk through a mock HTTP transport."""
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (2000, 3000), "white").save(src / "p1.jpg")
    page = {
        "subset": "pilot",
        "xml_path": "data/p1.xml",
        "image_path": "p1.jpg",
        "width": 2000,
        "height": 3000,
        "lines": [
            {"text": t, "bbox": [200, 300 + 200 * i, 1400, 120], "polygon": [[0, 0]]}
            for i, t in enumerate(LINES)
        ],
    }
    pages_file = tmp_path / "pages.jsonl"
    pages_file.write_text(json.dumps(page) + "\n")
    root = tmp_path / "iiif"
    iiif.publish("pilot", "Pilot", "CC-BY-4.0", pages_file, src, root / "pilot", BASE)

    def handler(request):
        path = root / request.url.path.lstrip("/")
        return (
            httpx.Response(200, content=path.read_bytes())
            if path.is_file()
            else httpx.Response(404)
        )

    client = IIIFClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    items = tasks.items("pilot", [page], BASE, ("transcription", "grounding"))
    return SimpleNamespace(items=items, source=runner.ImageSource(client), out=tmp_path / "run")


class OracleProvider:
    """Answers from the ground truth, after mapping canvas boxes to image pixels."""

    name, model, max_side = "fake", "fake-model", 1000

    def __init__(self, items, fail_first=False):
        self.items = items
        self.calls = 0
        self.fail_first = fail_first

    def complete(self, request: Request) -> Response:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("transient")
        width, _ = Image.open(io.BytesIO(request.images[0])).size
        if "boxes" in request.schema["properties"]:
            scale = width / 2000
            queries = re.findall(r"^(\d+): (.+)$", request.prompt, re.M)
            by_text = {i["query"]: i["reference"] for i in self.items if i["task"] == "grounding"}
            boxes = []
            for index, text in queries:
                x, y, w, h = by_text[text]
                boxes.append(
                    {
                        "index": int(index),
                        "found": True,
                        "box": [(x + w) * scale, y * scale, x * scale, (y + h) * scale],
                    }
                )
            return Response({"boxes": boxes}, "{}")
        return Response({"text": "In principio erat verbum"}, "{}")


def test_plan_groups_lines_per_canvas(bench):
    page_jobs = runner.plan(bench.items, "grounding:page", group_size=2)
    assert [len(j.items) for j in page_jobs] == [2, 1]
    region_jobs = runner.plan(bench.items, "transcription:region")
    assert len(region_jobs) == 3 and all(len(j.items) == 1 for j in region_jobs)
    assert runner.plan(bench.items, "grounding:page", pages_per_subset=0) == []
    with pytest.raises(ValueError):
        runner.plan(bench.items, "grounding:region")


def test_parse_rejects_bad_entries():
    job = runner.Job("j", "grounding:page", "pilot", "m", "c", [{"id": "a"}, {"id": "b"}])
    answer = {
        "boxes": [
            {"index": 0, "found": True, "box": [10, 20, 30, 60]},
            {"index": 1, "found": False, "box": []},
            {"index": 7, "found": True, "box": [1, 2, 3, 4]},
        ]
    }
    assert runner.parse(job, Response(answer, ""), 2.0) == [{"id": "a", "bbox": [20, 40, 40, 80]}]
    lines_job = runner.Job("j", "transcription:page", "pilot", "m", "c", [{"id": "a"}, {"id": "b"}])
    lines = {"lines": [{"index": 1, "text": "verbum"}, {"index": 0, "text": 3}]}
    assert runner.parse(lines_job, Response(lines, ""), 1.0) == [{"id": "b", "text": "verbum"}]
    assert runner.parse(lines_job, Response(None, "refused"), 1.0) == []


def test_grounding_run_maps_boxes_back_to_canvas_and_resumes(bench):
    provider = OracleProvider(bench.items)
    jobs = runner.plan(bench.items, "grounding:page", group_size=2)
    summary = runner.run(provider, jobs, bench.source, bench.out, {"pilot": ["lat"]}, workers=2)
    assert (summary["calls"], summary["errors"], summary["predictions"]) == (2, 0, 3)

    predictions = [json.loads(line) for line in (bench.out / "predictions.jsonl").open()]
    grounding = [i for i in bench.items if i["task"] == "grounding"]
    score = evaluate.score(grounding, {p["id"]: p for p in predictions})
    assert score["overall"]["grounding"]["acc@0.75"] == 1  # 2000 px canvas shown at 1000 px

    again = runner.run(provider, jobs, bench.source, bench.out, {"pilot": ["lat"]})
    assert again["jobs"] == 0 and provider.calls == 2


def test_failed_calls_are_logged_and_retried(bench):
    provider = OracleProvider(bench.items, fail_first=True)
    jobs = runner.plan(bench.items, "transcription:region")[:1]
    first = runner.run(provider, jobs, bench.source, bench.out, {"pilot": ["lat"]})
    assert first["errors"] == 1 and first["predictions"] == 0
    second = runner.run(provider, jobs, bench.source, bench.out, {"pilot": ["lat"]})
    assert second["errors"] == 0 and second["predictions"] == 1


def test_dry_run_estimates_without_calling(bench):
    provider = OracleProvider(bench.items)
    jobs = runner.plan(bench.items, "transcription:page")
    summary = runner.run(provider, jobs, bench.source, bench.out, {"pilot": ["lat"]}, dry_run=True)
    assert provider.calls == 0 and not (bench.out / "predictions.jsonl").exists()
    # one page image at 667x1000 -> 24 x 36 patches of 28 px
    assert summary["estimated_image_tokens"] == 24 * 36


def test_claude_provider_request_shape(monkeypatch):
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking"),
                SimpleNamespace(type="text", text='{"text": "x"}'),
            ],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            model="claude-opus-5-5",
        )

    provider = ClaudeProvider(effort="low")
    monkeypatch.setattr(
        provider, "_client", SimpleNamespace(messages=SimpleNamespace(create=create))
    )
    response = provider.complete(Request((b"\xff\xd8jpeg",), "Read it.", {"type": "object"}))
    assert response.data == {"text": "x"} and response.usage == {
        "input_tokens": 10,
        "output_tokens": 5,
    }
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    image, text = sent["messages"][0]["content"]
    assert image["source"]["media_type"] == "image/jpeg" and text == {
        "type": "text",
        "text": "Read it.",
    }
    assert "fallbacks" not in sent and "thinking" not in sent


def test_claude_provider_records_refusals(monkeypatch):
    def create(**kwargs):
        return SimpleNamespace(
            content=[],
            stop_reason="refusal",
            usage=SimpleNamespace(input_tokens=1, output_tokens=0),
            model="m",
        )

    provider = ClaudeProvider()
    monkeypatch.setattr(
        provider, "_client", SimpleNamespace(messages=SimpleNamespace(create=create))
    )
    response = provider.complete(Request((b"x",), "p", {}))
    assert response.data is None and response.stop_reason == "refusal"
