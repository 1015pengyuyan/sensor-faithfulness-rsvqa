# Datasets and redistribution boundaries

## M4-SAR

- Official repository: `wchao0601/m4-sar`
- Fixed revision: `e1609a2c7d2cacda10d4ad1e895df9627f37c236`
- Recorded dataset-card license: CC BY 4.0
- Released here: metadata-only confirmatory fixture and source-freeze tables
- Not released here: images, archives, labels outside the fixture, or test assets

The complete source archive was detected as containing a test prefix and was
quarantined before extraction, model inference, training, tuning, or reporting.
Only sanitized allowed train/validation assets were used. The official test
split is outside this paper and outside this repository.

The public fixture uses relative identifiers such as source IDs, image-relative
paths, and image hashes. Users must obtain permitted source data independently
and reproduce the sanitized layout under `${DATA_ROOT}`.

## BigEarthNet.txt and BigEarthNet v2

- Annotation repository: `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`
- Annotation revision: `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316`
- BigEarthNet v2 source: `https://doi.org/10.5281/zenodo.10891137`
- Version: 2.0.0
- Recorded license: CDLA-Permissive-1.0
- Released here: metadata-only fixture, patch/S1 identifiers, source-freeze
  tables, rendering configuration, and aggregate results
- Not released here: S1/S2 imagery, archives, or community LMDB files

The primary external evaluation is the 321-source smoke-naive subset. The 20
pre-exposed smoke source IDs are included only to make this exclusion
reproducible. The full 341-source fixture is retained so the construction and
exclusion can be audited.

## Expected BigEarthNet rendered layout

```text
${DATA_ROOT}/s2/<patch_id>.png
${DATA_ROOT}/s1/<s1_name>.png
```

Sentinel-2 is rendered as B04/B03/B02 with clip `[0, 3000]`. Sentinel-1 is
rendered from VV/VH dB using clips `[-25, 0]` and `[-32, -5]`; the third channel
is the deterministic rounded average of the two scaled channels. All output is
512x512 uint8 RGB. See `configs/bigearthnet_rendering.json`.

## License responsibility

Fixture metadata is provided to reproduce the paper protocol. Users remain
responsible for complying with the official dataset terms. Dataset licenses do
not automatically become the repository code license.
