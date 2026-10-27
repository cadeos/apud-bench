"""Score predictions against task items.

Results are reported three ways: ``overall`` (micro-average over all items), ``macro``
(unweighted mean over subsets) and per subset.

Predictions are JSON lines, one per item: ``{"id": ..., "text": ...}`` for transcription and
``{"id": ..., "bbox": [x, y, w, h]}`` (canvas coordinates) for grounding. A missing or malformed
prediction scores as the worst answer (empty text, IoU 0), so coverage cannot be gamed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from apud_bench.metrics import LEVELS, ErrorRate, iou, normalize

IOU_THRESHOLDS = (0.25, 0.5, 0.75)


def score(items: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_subset[item["subset"]].append(item)
    subsets = {s: _score_group(group, predictions) for s, group in sorted(by_subset.items())}
    return {
        "overall": _score_group(items, predictions),
        "macro": _macro(subsets),
        "subsets": subsets,
    }


def _macro(subsets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Unweighted mean over subsets, so large subsets do not drown small scripts."""
    out: dict[str, Any] = {}
    for task in ("transcription", "grounding"):
        scored = [r[task] for r in subsets.values() if task in r]
        if not scored:
            continue
        metrics = [k for k in scored[0] if k not in ("items", "answered")]
        out[task] = {
            "subsets": len(scored),
            **{k: round(sum(r[k] for r in scored) / len(scored), 4) for k in metrics},
        }
    return out


def _score_group(items: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict:
    out: dict[str, Any] = {}
    transcription = [i for i in items if i["task"] == "transcription"]
    grounding = [i for i in items if i["task"] == "grounding"]
    if transcription:
        cer = {level: ErrorRate() for level in LEVELS}
        wer = {level: ErrorRate() for level in LEVELS}
        answered = 0
        for item in transcription:
            prediction = predictions.get(item["id"], {})
            text = prediction.get("text")
            if isinstance(text, str):
                answered += 1
            else:
                text = ""
            for level in LEVELS:
                reference, hypothesis = normalize(item["reference"], level), normalize(text, level)
                cer[level].add(reference, hypothesis)
                wer[level].add(reference.split(), hypothesis.split())
        out["transcription"] = {
            "items": len(transcription),
            "answered": answered,
            **{f"cer_{level}": round(cer[level].value, 4) for level in LEVELS},
            **{f"wer_{level}": round(wer[level].value, 4) for level in LEVELS},
        }
    if grounding:
        ious = [_iou(item, predictions.get(item["id"], {})) for item in grounding]
        out["grounding"] = {
            "items": len(grounding),
            "answered": sum(
                _valid_box(predictions.get(i["id"], {}).get("bbox")) for i in grounding
            ),
            "mean_iou": round(sum(ious) / len(ious), 4),
            **{f"acc@{t}": round(sum(v >= t for v in ious) / len(ious), 4) for t in IOU_THRESHOLDS},
        }
    return out


def _valid_box(box: Any) -> bool:
    return (
        isinstance(box, list | tuple)
        and len(box) == 4
        and all(isinstance(v, int | float) for v in box)
    )


def _iou(item: dict[str, Any], prediction: dict[str, Any]) -> float:
    box = prediction.get("bbox")
    return iou(item["reference"], box) if _valid_box(box) else 0.0


def baseline(items: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Reference predictions that bound the metrics: ``oracle`` (perfect) and ``trivial``
    (empty text, whole-canvas box)."""
    predictions = []
    for item in items:
        if item["task"] == "transcription":
            text = item["reference"] if kind == "oracle" else ""
            predictions.append({"id": item["id"], "text": text})
        else:
            w, h = item["canvas_size"]
            box = item["reference"] if kind == "oracle" else [0, 0, w, h]
            predictions.append({"id": item["id"], "bbox": box})
    return predictions
