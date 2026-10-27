# apud-bench

**Can a vision-language model say where it read what it reports?**

`apud-bench` evaluates vision-language models on historical handwritten documents with three
tasks:

| Task | Question | Metric |
|---|---|---|
| Transcription | Read a line, full page vs IIIF region crop | CER / WER |
| Grounding | Given a line of text, return its box on the page | IoU against ground-truth line geometry |
| Citation faithfulness | In a long transcription, is each line attributed to the right region? | correct / wrong / hallucinated citations |

Existing HTR benchmarks for VLMs score reading only. Document-grounding benchmarks use modern
PDFs. `apud-bench` brings grounding and citation to manuscripts, across scripts and centuries.

Status: pre-alpha. The pipeline runs end to end on four pilot subsets; model runners and the
citation-faithfulness task are next.

## Data

The benchmark is built from **openly downloadable ground truth produced by the HTR community**
(HTR-United, Zenodo, GitHub). Nothing is redistributed: a build script fetches each subset from
its original source, normalises its geometry and text, and generates static IIIF manifests and
tiles locally.

- **core**: open licences only (CC-BY, CC-BY-SA, CC0, Apache-2.0, MIT, Etalab 2.0)
- **nc-extension**: NonCommercial subsets, kept separate
- **hold**: candidates whose licence or access still needs confirming

Every derived artefact carries the licence of the subset it comes from. See
[DATA_LICENSES.md](DATA_LICENSES.md) and [`subsets/subsets.yaml`](subsets/subsets.yaml).

```bash
uv run apud-bench stats          # registry summary
uv run apud-bench list --tier core
uv run apud-bench check          # enforce licence/tier rules
uv run apud-bench inspect page.xml --image-size 4000x6000   # PAGE/ALTO -> canvas JSON
```

### Ground-truth formats

`apud_bench.formats.load()` reads PAGE (2010-03-19 to 2019-07-15) and ALTO (v2 to v4) into one
model, `Page` -> `Line` (text, polygon, baseline, bbox), in **canvas coordinates**: pixels of the
full-resolution page image. It handles `points` attributes and 2010 `<Point>` children, ALTO
polygons written as `x,y` or `x y`, rectangle-only lines, scalar and point baselines, and
Transkribus `mm10` units. Passing the real image size rescales ground truth produced on resized
images.

### Building subsets

```bash
uv run apud-bench lock htromance-it eparchos      # pin sources in subsets/sources.lock.json
uv run apud-bench build htromance-it eparchos     # download, verify, extract, normalise
```

- **lock** pins each source: the GitHub commit SHA, or the exact Zenodo record version with
  per-file checksums. Builds only use what the lock file says, so they are reproducible.
- **fetch/build** download into `$APUD_BENCH_DATA` (default `~/.cache/apud-bench`), verify
  checksums and extract safely (no path traversal, no links, size cap).
- **build** pairs every PAGE/ALTO file with its image (declared file name first, then file
  stem, nearest folder wins), reads the real image size, rescales the ground truth to it, and
  writes `build/<subset>/pages.jsonl` plus `report.json` with counts and every skipped file by
  reason (`no-image`, `aspect-mismatch`, `duplicate-image`, `no-lines`, ...).

### Static IIIF

```bash
uv run apud-bench iiif htromance-it eparchos      # Level 0 tiles + manifests
uv run apud-bench serve                           # http://localhost:8000, with CORS
```

Each built subset is published under `<data-dir>/iiif/<subset>/` as a IIIF Image API 3
**Level 0** service per page (512 px tiles, `sizes` matching the pyramid levels, `full/max`)
and one Presentation 3 `manifest.json`, so any IIIF viewer or client can open it. Canvas size
equals the image pixel size, so ground-truth coordinates are canvas coordinates.

Ground truth is written to `annotations/<page>.json` as W3C annotations (`xywh` and SVG
polygon selectors). These pages are **not linked from the manifest**: a model browsing the
manifest cannot read the answers. Use `--base-url` when the data is hosted elsewhere.

### Splits, tasks and scoring

```bash
uv run apud-bench split htromance-it eparchos --pages 20      # freeze splits/v0/<subset>.json
uv run apud-bench tasks htromance-it eparchos --out items.jsonl
uv run apud-bench baseline --items items.jsonl --kind oracle --out oracle.jsonl
uv run apud-bench evaluate --items items.jsonl --predictions oracle.jsonl
```

- **Splits** are evaluation-only page lists, drawn round-robin across manuscripts with a fixed
  seed and committed in `splits/v0/`. Each split records the source pin it was drawn from and
  is refused against any other version of the data.
- **Items** are addressed by IIIF identifiers (manifest, canvas, `xywh`), so a model or an
  agent using `apud-mcp` works from the IIIF data, never from local files.
  - `transcription`: read the text of a canvas region;
  - `grounding`: given a line of text, return its box on the canvas. Lines shorter than five
    characters and lines whose text repeats on the same page are excluded as ambiguous.
- **Predictions** are JSON lines: `{"id", "text"}` or `{"id", "bbox": [x, y, w, h]}`. Missing
  or malformed predictions score as the worst answer.
- **Metrics**: CER and WER at two normalisation levels, `strict` (NFC, whitespace) and `loose`
  (also no combining marks, case folded, long s and r rotunda modernised); mean IoU and
  accuracy at IoU 0.25 / 0.5 / 0.75. Reported micro-averaged (`overall`), as an unweighted
  mean over subsets (`macro`), and per subset.

`baseline --kind oracle` (perfect) and `--kind trivial` (empty text, whole-canvas box) bound the
metrics: on the four pilot subsets they score CER 0 / 1 and mean IoU 1 / 0.013.

### Running models

```bash
pip install "apud-bench[runners]"      # adds apud, anthropic and openai SDKs
uv run apud-bench serve &              # the IIIF data the models will read

# estimate first: fetches every image through IIIF, calls no model
uv run apud-bench run --items items.jsonl --condition grounding:page --out runs/opus --dry-run

uv run apud-bench run --items items.jsonl --condition grounding:page --out runs/opus \
    --provider anthropic --model claude-opus-5-5 --effort medium --pages-per-subset 2
uv run apud-bench evaluate --items items.jsonl --task grounding --predictions runs/opus/predictions.jsonl
```

Three conditions isolate what the model is good at:

| Condition | The model sees | Calls |
|---|---|---|
| `transcription:region` | only the IIIF crop of one line | one per line |
| `transcription:page` | the whole page and numbered line boxes | one per up to 10 lines |
| `grounding:page` | the whole page and numbered line texts | one per up to 10 lines |

Comparing `transcription:region` with `transcription:page` measures what region access through
IIIF is worth. Images are always fetched through IIIF with `apud`, at the model's own maximum
resolution; boxes are converted between image pixels and canvas coordinates by the runner.

Runs are resumable: `predictions.jsonl` and `calls.jsonl` are appended per call, and finished
items are never paid for twice. Failed calls are logged and retried on the next run; refusals are
recorded and score as missing. Providers: `anthropic` (Claude, official SDK; no refusal
fallback, so the named model is always the one scored) and `openai` (OpenAI and any
OpenAI-compatible endpoint via `--base-url`). Credentials come from the environment
(`ANTHROPIC_API_KEY`, or the variable named by `--api-key-env`).

## Credits

All ground truth belongs to its creators; each subset entry links to its source. Please cite
the original datasets alongside this benchmark.

## License

Code: Apache-2.0. Data: per subset, see [DATA_LICENSES.md](DATA_LICENSES.md).
