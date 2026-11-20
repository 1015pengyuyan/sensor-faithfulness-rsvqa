# Eleven-model CPU reproduction

This entry point audits the immutable 2026-11-19 editorial-v6 normalized matrix.
Python 3.10+ and the repository CPU dependencies are sufficient. No GPU, service
credentials, image files, model downloads or network calls are used.

## Evidence layout

Extract the separately published v3 normalized-evidence attachment into a fresh
directory such that `MANIFEST.json` is directly inside it. The expected payload is:

```text
reproduction_data/v3/
  MANIFEST.json
  NORMALIZED_SELECTED.jsonl
  ENDPOINTS.csv
  SUPPLEMENT_ENDPOINTS.csv
  COHORTS.csv
  OUTCOMES.csv
  COMPLETION_AUDIT.json
  MODELS.json
  CHECKPOINT_IDENTITIES.json
  SOURCE_EXCLUSION.json
  LICENSE_SCOPE.json
  README.md
  ... preserved license and attribution notices
```

`configs/v3/evidence.json` independently pins the nine scientific payloads by
SHA-256 and size. `MANIFEST.json` covers the complete attachment payload. Archive
checksums belong to the release attachment record, not to a self-hashing manifest.
Do not substitute the v2 attachments or an older partially completed matrix.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python reproduction/v3/reproduce.py verify --data reproduction_data/v3
python reproduction/v3/reproduce.py all --data reproduction_data/v3 --out reproduced/v3
```

Use a new output path for each replay. Existing output paths are refused before
writing. Missing evidence, modified bytes, unsafe manifest paths, duplicate
scientific inputs, replay contamination, score/parser disagreement and incomplete
endpoint coverage cause an explicit failure.

## What is recomputed

The command validates 195,618 normalized scientific records across eleven models
and two datasets. It checks binary target scoring, abstention and parser-failure
annotations, and reparses the 113,052 records containing `raw_output` with the
repository's frozen parser. Original five-model normalized records have no raw
text in this package; they receive annotation/accounting checks.

The preserved `reproduction/v2/frozen_metrics.py` definitions reconstruct all
6,888 main archived point estimates and their eligible denominators: capability,
conditional CIFR/RSR, control contrasts and the originally specified position and
interaction summaries. Undefined ratios remain non-estimable rather than zero.
All 33 model/dataset/cohort entries are reconstructed, including the seven-source
BEN exclusion. No model or nonpositive endpoint is dropped.

Outputs are `RECOMPUTED_MAIN_POINTS.csv`, `RECOMPUTED_COHORTS.csv`, and
`REPRODUCTION_RECEIPT.json`. These are local verification products, not additional
scientific inputs. The receipt records Python version, the input manifest hash,
counts, the closed-model completion summary and explicit scope limitations.

## What remains archived

Confidence limits, bootstrap-validity counts, `SUPPLEMENT_ENDPOINTS.csv`,
`OUTCOMES.csv` and completion observations retain the exact pinned source bytes.
The current command does not regenerate interval arrays, independently replay
original client calls, validate third-party image bytes or reconstruct the separate
2026-11-19 review-supplement package. A matching hash demonstrates artifact
identity; it does not independently validate a statistical assumption.

Archived intervals condition on fixed donor assignments and observed target
sources. They do not establish independent geography/parent scenes and do not
include the closed services' observed generation variation. No saved draw array is
replaced, extended or newly sampled. The existing v1/v2 full cached-draw replay
remains available under its historical instructions and assets.
