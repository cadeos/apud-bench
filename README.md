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

Status: pre-alpha. Subset registry and PAGE/ALTO readers are in place; download, build and
evaluation are next.

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

## Credits

All ground truth belongs to its creators; each subset entry links to its source. Please cite
the original datasets alongside this benchmark.

## License

Code: Apache-2.0. Data: per subset, see [DATA_LICENSES.md](DATA_LICENSES.md).
