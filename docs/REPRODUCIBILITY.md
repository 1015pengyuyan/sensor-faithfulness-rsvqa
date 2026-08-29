# Reproducibility record

## Scientific freeze

The release does not revise prompts, parser behavior, image ordering,
distractor matching, model revisions, rendering, predictions, sample sets, or
reported statistics. Official test assets remain excluded.

## Historical GPU environment recorded by run manifests

- Python: 3.10 environment
- PyTorch: 2.3.1
- Transformers: 4.49.0
- Deterministic decoding: `do_sample=False`, `max_new_tokens=16`

Exact historical versions of Accelerate, NumPy, Pandas, Pillow, and Matplotlib
were not embedded in the frozen GPU manifests. They are therefore not invented
here. The release reference environment pins the locally verified CPU artifact
stack below and compatible GPU dependencies in `environment.yml`.

## Locally verified release CPU environment

- Python 3.11.9
- NumPy 1.26.4
- Pandas 2.3.3
- Pillow 11.3.0
- Matplotlib 3.9.2

Rasterio/GDAL were not used by the released rendering implementation; TIFF
bands are read through Pillow. They are intentionally absent from the minimal
environment rather than assigned an unverified version.

## Model revisions

See `configs/models.yaml`. Revisions are immutable commit hashes; `latest`,
`main`, and `master` are not accepted reproduction values.

## Integrity gates

Run:

```bash
python scripts/check_paper_consistency.py
python scripts/verify_fixture.py fixtures/m4sar/COUNTERFACTUAL_CONFIRMATORY_FIXTURE.jsonl --expected-sha256 94b71c918476a266a901ee7d3430f16d3c104369c6ce29c1cce1a4e50555baef
python scripts/verify_fixture.py fixtures/bigearthnet/BIGEARTHNET_EXTERNAL_FULL_FIXTURE.jsonl --expected-sha256 262ebdcc3b09da82ed4508cae7edce9235857b864acd81b2130078be51fda6cb
```

Generated predictions and raw outputs belong under ignored `outputs/raw/`.
Never commit model caches, images, or credentials.
