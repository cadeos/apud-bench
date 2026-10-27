import json

from PIL import Image

from apud_bench import iiif


def test_page_key_is_stable_and_url_safe():
    key = iiif.page_key("repo-abc/data/Français 1593 (2).xml")
    assert key == iiif.page_key("repo-abc/data/Français 1593 (2).xml")
    assert key.startswith("Fran-ais-1593-2-") and len(key.rsplit("-", 1)[1]) == 8
    assert iiif.page_key("a/p1.xml") != iiif.page_key("b/p1.xml")


def test_scale_factors_stop_when_image_fits_one_tile():
    assert iiif.scale_factors(500, 400) == [1]
    assert iiif.scale_factors(1100, 700) == [1, 2, 4]


def test_tiles_cover_every_pixel_once_per_scale_factor():
    width, height = 1100, 700
    factors = iiif.scale_factors(width, height)
    boxes = [box for _, _, box in iiif.tile_requests(width, height)]
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes)
    assert area == width * height * len(factors)
    for region, size, (x0, y0, x1, y1) in iiif.tile_requests(width, height):
        sw, sh = (int(n) for n in size.split(","))
        assert sw <= iiif.TILE_SIZE and sh <= iiif.TILE_SIZE
        if region != "full":
            assert region == f"{x0},{y0},{x1 - x0},{y1 - y0}"


def test_sizes_match_pyramid_levels():
    assert len(iiif.level_sizes(1766, 2546)) == len(iiif.scale_factors(1766, 2546))
    assert iiif.level_sizes(1766, 2546)[0] == (221, 319)


def test_single_tile_level_uses_full_region():
    requests = list(iiif.tile_requests(1100, 700))
    assert requests[-1] == ("full", "275,175", (0, 0, 1100, 700))


def write_page(tmp_path, size=(1100, 700)):
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", size, "white").save(src / "p1.jpg")
    page = {
        "subset": "pilot",
        "xml_path": "data/p1.xml",
        "image_path": "p1.jpg",
        "id": "p1",
        "width": size[0],
        "height": size[1],
        "lines": [
            {
                "id": "l1",
                "text": "In principio",
                "bbox": [10, 20, 300, 40],
                "polygon": [[10, 20], [310, 20], [310, 60], [10, 60]],
                "baseline": None,
                "region_id": "r1",
            }
        ],
    }
    pages = tmp_path / "pages.jsonl"
    pages.write_text(json.dumps(page) + "\n", encoding="utf-8")
    return src, pages


def test_publish_writes_level0_service_manifest_and_hidden_ground_truth(tmp_path):
    src, pages = write_page(tmp_path)
    out = tmp_path / "iiif" / "pilot"
    count = iiif.publish("pilot", "Pilot", "CC-BY-4.0", pages, src, out, "http://host/iiif/")
    assert count == 1

    key = iiif.page_key("data/p1.xml")
    service = out / "images" / key
    info = json.loads((service / "info.json").read_text())
    assert info["id"] == f"http://host/iiif/pilot/images/{key}"
    assert (info["profile"], info["width"], info["height"]) == ("level0", 1100, 700)
    assert info["tiles"] == [{"width": 512, "scaleFactors": [1, 2, 4]}]
    assert [(s["width"], s["height"]) for s in info["sizes"]] == [
        (275, 175),
        (550, 350),
        (1100, 700),
    ]
    for region, size, _ in iiif.tile_requests(1100, 700):
        tile = service / region / size / "0" / "default.jpg"
        assert tile.exists(), tile
        with Image.open(tile) as im:
            assert im.size == tuple(int(n) for n in size.split(","))
    assert (service / "full" / "max" / "0" / "default.jpg").exists()
    assert (service / "full" / "550,350" / "0" / "default.jpg").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["id"] == "http://host/iiif/pilot/manifest.json"
    [canvas] = manifest["items"]
    assert (canvas["width"], canvas["height"]) == (1100, 700)
    body = canvas["items"][0]["items"][0]["body"]
    assert body["service"][0]["id"] == info["id"]
    assert "annotations" not in canvas and "annotations" not in manifest

    truth = json.loads((out / "annotations" / f"{key}.json").read_text())
    [line] = truth["items"]
    assert line["body"]["value"] == "In principio"
    assert line["target"]["source"]["id"] == canvas["id"]
    assert line["target"]["selector"][0]["value"] == "xywh=10,20,300,40"
    assert line["target"]["selector"][1]["value"].startswith("<svg")
