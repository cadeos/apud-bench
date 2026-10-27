import hashlib
import io
import json
import tarfile
import zipfile

import httpx
import pytest
from PIL import Image

from apud_bench import sources
from apud_bench.build import build
from apud_bench.fetch import extract, fetch
from apud_bench.registry import Subset
from apud_bench.sources import SourceError


def subset(url="https://github.com/org/repo", **kwargs):
    fields = dict(
        id="pilot",
        name="Pilot",
        url=url,
        license="CC-BY-4.0",
        tier="core",
        stratum="medieval",
        script="Latn",
        langs=["lat"],
        centuries="12",
        tasks=["T", "G"],
    )
    return Subset(**{**fields, **kwargs})


def mock_client(routes, calls=None):
    def handler(request):
        url = str(request.url)
        if calls is not None:
            calls.append(url)
        if url not in routes:
            return httpx.Response(404)
        body = routes[url]
        if isinstance(body, bytes):
            return httpx.Response(200, content=body)
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


# ── sources ──────────────────────────────────────────────────────────────


def test_resolve_github_pins_commit():
    http = mock_client(
        {
            "https://api.github.com/repos/org/repo": {"default_branch": "main"},
            "https://api.github.com/repos/org/repo/commits/main": {"sha": "abc123"},
        }
    )
    pin = sources.resolve(subset("https://github.com/org/repo/tree/main"), http)
    assert pin == {
        "kind": "github",
        "repo": "org/repo",
        "commit": "abc123",
        "url": "https://codeload.github.com/org/repo/tar.gz/abc123",
    }


ZENODO_RECORD = {
    "id": 42,
    "files": [
        {"key": f, "size": 1, "checksum": "md5:x", "links": {"self": f"https://z/{f}"}}
        for f in ("page.zip", "alto.zip", "images.zip", "CITATION.cff", "htr-united.yml")
    ],
}


@pytest.mark.parametrize(
    "url",
    [
        "https://zenodo.org/records/41",
        "https://zenodo.org/record/41#.Y6LQ",
        "https://doi.org/10.5281/zenodo.41",
        "10.5281/zenodo.41",
    ],
)
def test_resolve_zenodo_url_forms_and_metadata_filter(url):
    http = mock_client({"https://zenodo.org/api/records/41": ZENODO_RECORD})
    pin = sources.resolve(subset(url), http)
    assert pin["record"] == 42  # concept id resolved to the pinned version
    assert [f["key"] for f in pin["files"]] == ["page.zip", "alto.zip", "images.zip"]


def test_resolve_zenodo_with_fetch_hint():
    http = mock_client({"https://zenodo.org/api/records/41": ZENODO_RECORD})
    hinted = subset("https://zenodo.org/records/41", fetch={"files": ["page.zip", "images.zip"]})
    assert [f["key"] for f in sources.resolve(hinted, http)["files"]] == ["page.zip", "images.zip"]
    with pytest.raises(SourceError, match="not in Zenodo record"):
        sources.resolve(subset("https://zenodo.org/records/41", fetch={"files": ["x.zip"]}), http)


def test_resolve_unsupported_host():
    with pytest.raises(SourceError, match="unsupported source"):
        sources.resolve(subset("https://nakala.fr/10.34847/x"), mock_client({}))


def test_lock_round_trip(tmp_path):
    path = tmp_path / "lock.json"
    sources.write_lock({"b": {"kind": "github"}, "a": {"kind": "zenodo"}}, path)
    assert list(sources.read_lock(path)) == ["a", "b"]
    assert sources.read_lock(tmp_path / "missing.json") == {}


# ── fetch ────────────────────────────────────────────────────────────────


def zip_bytes(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def test_fetch_zenodo_verifies_checksum_and_is_idempotent(tmp_path):
    archive = zip_bytes({"data/page.xml": "<x/>"})
    pin = {
        "kind": "zenodo",
        "record": 1,
        "files": [
            {
                "key": "page.zip",
                "size": len(archive),
                "checksum": "md5:" + hashlib.md5(archive).hexdigest(),
                "url": "https://z/page.zip",
            }
        ],
    }
    calls = []
    http = mock_client({"https://z/page.zip": archive}, calls)
    target = fetch("pilot", pin, http, tmp_path)
    assert (target / "data" / "page.xml").read_text() == "<x/>"
    fetch("pilot", pin, http, tmp_path)
    assert calls == ["https://z/page.zip"]


def test_fetch_rejects_checksum_mismatch(tmp_path):
    pin = {
        "kind": "zenodo",
        "record": 1,
        "files": [{"key": "a.zip", "size": 1, "checksum": "md5:00", "url": "https://z/a.zip"}],
    }
    with pytest.raises(SourceError, match="checksum mismatch"):
        fetch("pilot", pin, mock_client({"https://z/a.zip": zip_bytes({"a": "b"})}), tmp_path)


def test_fetch_github_tarball(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        data = b"<alto/>"
        info = tarfile.TarInfo("repo-abc/data/f1.xml")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    pin = {"kind": "github", "repo": "o/r", "commit": "abc", "url": "https://gh/abc"}
    target = fetch("pilot", pin, mock_client({"https://gh/abc": buffer.getvalue()}), tmp_path)
    assert (target / "repo-abc" / "data" / "f1.xml").read_bytes() == b"<alto/>"


def test_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("ok/../../escape.txt")
        zf.writestr(info, "x")
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(SourceError, match="unsafe path"):
        extract(archive, target)
    assert not (tmp_path / "escape.txt").exists()


# ── build ────────────────────────────────────────────────────────────────

PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="{image}" imageWidth="{w}" imageHeight="{h}">
    <TextRegion id="r1">{lines}</TextRegion>
  </Page>
</PcGts>"""

LINE = """<TextLine id="l1"><Coords points="10,10 90,10 90,30 10,30"/>
  <TextEquiv><Unicode>In principio</Unicode></TextEquiv></TextLine>"""


def make_image(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 255).save(path)


def make_xml(path, image="", w=100, h=200, lines=LINE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PAGE_XML.format(image=image, w=w, h=h, lines=lines), encoding="utf-8")


def test_build_pairs_images_and_reports_skips(tmp_path):
    src = tmp_path / "src"
    # declared name, image in another folder, at twice the annotated resolution
    make_xml(src / "page" / "a.xml", image="scans/A.JPG")
    make_image(src / "images" / "a.jpg", (200, 400))
    # no declared name: matched by stem
    make_xml(src / "page" / "b.xml")
    make_image(src / "images" / "b.png", (100, 200))
    # same image declared twice
    make_xml(src / "page" / "c.xml", image="b.png")
    # cropped image: aspect ratio differs
    make_xml(src / "page" / "d.xml")
    make_image(src / "images" / "d.jpg", (100, 100))
    # no image at all, and a page without lines
    make_xml(src / "page" / "e.xml")
    make_xml(src / "page" / "f.xml", image="f.jpg", lines="")
    make_image(src / "images" / "f.jpg", (100, 200))
    # not ground truth: ignored silently
    (src / "page" / "mets.xml").write_text("<mets/>", encoding="utf-8")

    out = tmp_path / "build"
    report = build(subset(), {"kind": "github"}, src, out)

    assert (report.pages, report.lines) == (2, 2)
    assert dict(report.skipped) == {
        "duplicate-image": 1,
        "aspect-mismatch": 1,
        "no-image": 1,
        "no-lines": 1,
    }
    pages = [json.loads(line) for line in (out / "pages.jsonl").read_text().splitlines()]
    a, b = pages
    assert a["image_path"] == "images/a.jpg" and (a["width"], a["height"]) == (200, 400)
    assert a["lines"][0]["bbox"] == [20, 20, 160, 40]  # rescaled to the real image
    assert b["image_path"] == "images/b.png" and b["lines"][0]["bbox"] == [10, 10, 80, 20]
    saved = json.loads((out / "report.json").read_text())
    assert saved["license"] == "CC-BY-4.0" and saved["skipped_examples"]["no-image"] == [
        "page/e.xml"
    ]


def test_build_xml_glob_selects_transcription_layer(tmp_path):
    src = tmp_path / "src"
    make_xml(src / "p1.xml")
    make_xml(src / "p1.chocomufin.xml", image="p1.jpg")
    make_image(src / "p1.jpg", (100, 200))
    hinted = subset(build={"xml_glob": "*.chocomufin.xml"})
    report = build(hinted, {"kind": "github"}, src, tmp_path / "out")
    assert (report.pages, dict(report.skipped)) == (1, {})
    page = json.loads((tmp_path / "out" / "pages.jsonl").read_text())
    assert page["xml_path"] == "p1.chocomufin.xml"
