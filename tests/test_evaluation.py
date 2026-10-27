import pytest

from apud_bench import evaluate, metrics, splits, tasks

# ── metrics ──────────────────────────────────────────────────────────────


def test_normalize_levels():
    assert metrics.normalize("  In   principio\n", "strict") == "In principio"
    assert metrics.normalize("ſcriptũ Ꝛ", "strict") == "ſcriptũ Ꝛ"
    assert metrics.normalize("ſcriptũ", "loose") == "scriptu"
    assert metrics.normalize("ÉRAT ꝛ", "loose") == "erat r"
    with pytest.raises(ValueError):
        metrics.normalize("x", "other")


def test_edit_distance():
    assert metrics.edit_distance("kitten", "sitting") == 3
    assert metrics.edit_distance("", "abc") == 3
    assert metrics.edit_distance(["in", "principio"], ["in", "principium"]) == 1


def test_error_rate_is_micro_averaged():
    rate = metrics.ErrorRate()
    rate.add("abcd", "abcd")
    rate.add("ab", "")
    assert rate.value == pytest.approx(2 / 6)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),
        ([0, 0, 10, 10], [5, 0, 10, 10], 50 / 150),
        ([0, 0, 10, 10], [20, 20, 5, 5], 0.0),
        ([0, 0, 10, 10], [0, 0, 0, 10], 0.0),
    ],
)
def test_iou(a, b, expected):
    assert metrics.iou(a, b) == pytest.approx(expected)


# ── splits ───────────────────────────────────────────────────────────────

PATHS = [f"repo-sha/data/ms{m}/p{p}.xml" for m in range(3) for p in range(10)] + [
    "repo-sha/data/ms3/only.xml"
]


def test_draw_spreads_across_documents_and_is_deterministic():
    chosen = splits.draw(PATHS, 8, seed=0)
    assert chosen == splits.draw(PATHS, 8, seed=0) == sorted(chosen)
    documents = [splits.document_of(p) for p in chosen]
    assert set(documents) == {"data/ms0", "data/ms1", "data/ms2", "data/ms3"}
    assert (
        max(documents.count(d) for d in set(documents))
        - min(documents.count(d) for d in {"data/ms0", "data/ms1", "data/ms2"})
        <= 1
    )
    assert splits.draw(PATHS, 8, seed=1) != chosen
    assert len(splits.draw(PATHS, 100)) == len(PATHS)


def test_split_is_bound_to_its_source_pin(tmp_path):
    pin = {"kind": "github", "commit": "abc"}
    splits.write("pilot", pin, ["a.xml"], 0, tmp_path)
    assert splits.read("pilot", pin, tmp_path) == ["a.xml"]
    with pytest.raises(splits.SplitError, match="different source pin"):
        splits.read("pilot", {"kind": "github", "commit": "def"}, tmp_path)
    with pytest.raises(splits.SplitError, match="no v0 split"):
        splits.read("other", pin, tmp_path)


# ── tasks ────────────────────────────────────────────────────────────────


def page(lines):
    return {
        "xml_path": "repo/data/p1.xml",
        "width": 1000,
        "height": 2000,
        "lines": [{"text": t, "bbox": [0, i * 50, 500, 40]} for i, t in enumerate(lines)],
    }


def test_items_filter_grounding_to_unambiguous_lines():
    lines = ["In principio erat", "", "Amen", "et verbum", "et  verbum", "Deus meus"]
    out = tasks.items("pilot", [page(lines)], "http://host/", ("transcription", "grounding"))
    transcription = [i for i in out if i["task"] == "transcription"]
    grounding = [i for i in out if i["task"] == "grounding"]
    assert [i["reference"] for i in transcription] == [t for t in lines if t]
    # empty skipped everywhere; "Amen" too short; the two "et verbum" lines are ambiguous
    assert [i["query"] for i in grounding] == ["In principio erat", "Deus meus"]
    first = grounding[0]
    assert first["manifest"] == "http://host/pilot/manifest.json"
    assert first["canvas"].startswith("http://host/pilot/canvas/p1-")
    assert first["reference"] == [0, 0, 500, 40] and first["canvas_size"] == [1000, 2000]


# ── evaluate ─────────────────────────────────────────────────────────────


def sample_items():
    return tasks.items(
        "pilot",
        [page(["In principio erat", "Deus meus"])],
        "http://h",
        ("transcription", "grounding"),
    )


def test_oracle_and_trivial_bound_the_metrics():
    items = sample_items()
    oracle = evaluate.score(items, {p["id"]: p for p in evaluate.baseline(items, "oracle")})
    trivial = evaluate.score(items, {p["id"]: p for p in evaluate.baseline(items, "trivial")})
    assert oracle["overall"]["transcription"]["cer_strict"] == 0
    assert oracle["overall"]["grounding"]["mean_iou"] == 1
    assert trivial["overall"]["transcription"]["cer_strict"] == 1
    assert trivial["overall"]["grounding"]["acc@0.5"] == 0
    assert set(oracle["subsets"]) == {"pilot"}


def test_missing_and_malformed_predictions_score_worst():
    items = sample_items()
    grounding_id = next(i["id"] for i in items if i["task"] == "grounding")
    result = evaluate.score(items, {grounding_id: {"id": grounding_id, "bbox": [1, 2, "x", 4]}})
    assert result["overall"]["transcription"]["answered"] == 0
    assert result["overall"]["transcription"]["cer_loose"] == 1
    assert result["overall"]["grounding"]["answered"] == 0
    assert result["overall"]["grounding"]["mean_iou"] == 0


def test_macro_average_weights_subsets_equally():
    big = tasks.items(
        "big", [page(["aaaa aaaa"] * 1 + [f"line {i}" for i in range(9)])], "h", ("transcription",)
    )
    small = tasks.items("small", [page(["bbbb bbbb"])], "h", ("transcription",))
    predictions = {i["id"]: {"id": i["id"], "text": i["reference"]} for i in big}
    result = evaluate.score(big + small, predictions)
    assert result["subsets"]["big"]["transcription"]["cer_strict"] == 0
    assert result["subsets"]["small"]["transcription"]["cer_strict"] == 1
    assert result["macro"]["transcription"]["cer_strict"] == 0.5
    assert result["overall"]["transcription"]["cer_strict"] < 0.2
