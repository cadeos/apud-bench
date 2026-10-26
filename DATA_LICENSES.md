# Data licences

`apud-bench` does not redistribute images or transcriptions. It ships code, the subset registry
and evaluation splits; the data is fetched from each original source by the build script.

## Rules

1. **Each subset keeps its own licence.** Anything derived from a subset (normalised geometry,
   split files, generated IIIF manifests and tiles, model outputs on it) is released under that
   subset's licence, with attribution to its creators.
2. **The most restrictive licence wins.** When the HTR-United catalog and the source (Zenodo
   record or repository LICENSE) disagree, the registry records the more restrictive one.
3. **core** accepts only CC-BY-4.0, CC-BY-SA-4.0, CC0-1.0, Apache-2.0, MIT and Etalab-2.0.
   CC-BY-SA subsets are flagged `share-alike`: their derived artefacts must be CC-BY-SA too.
4. **nc-extension** holds every NonCommercial subset. Results on it are reported separately.
5. **hold** subsets are not used until their licence or access is confirmed.

`uv run apud-bench check` enforces rules 3 and 4 on the registry.

## Notes on the registry

- `licence_note: catalog-only` means the source repository has no LICENSE file and the licence
  is taken from its HTR-United catalog entry.
- Image rights held by the owning library may add conditions beyond the dataset licence; check
  the source before reusing images outside research.
