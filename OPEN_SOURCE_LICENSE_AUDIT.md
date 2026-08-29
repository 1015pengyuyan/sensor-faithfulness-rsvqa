# Open-source license audit

## Decision

**Repository code license candidate: Apache-2.0.**

Apache-2.0 is preferred over MIT for this release because it supplies an
explicit patent grant while remaining permissive. This decision applies only to
repository-authored source and documentation.

## Third-party implementation audit

- The parser, fixture construction, validation, metric, bootstrap, rendering,
  and paper-artifact code were extracted and reorganized from the paper's own
  frozen research implementation.
- No third-party model source file was copied into this repository.
- Qwen, EarthMind, and LLaVA-OneVision modules are thin API wrappers. Model
  custom code and checkpoints are fetched separately by users from the pinned
  official repositories.
- EarthMind's pinned repository includes custom code with upstream notices; that
  code is not redistributed here and is not relicensed.
- Python dependencies are referenced, not vendored.

## External asset licenses

| Asset | Fixed source | Recorded license | Redistributed here |
|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | pinned Hugging Face revision | Qwen Research License | no checkpoint/code |
| EarthMind-4B | pinned Hugging Face revision | Apache-2.0 model repository | no checkpoint/custom code |
| LLaVA-OneVision-7B | pinned Hugging Face revision | Apache-2.0 model repository | no checkpoint |
| M4-SAR | pinned Hugging Face dataset revision | CC BY 4.0 | metadata-only fixture |
| BigEarthNet.txt | pinned Hugging Face dataset revision | CDLA-Permissive-1.0 | metadata-only fixture |
| BigEarthNet v2 | DOI 10.5281/zenodo.10891137, v2.0.0 | CDLA-Permissive-1.0 | no images/archive |

Official source and attribution details are recorded in `configs/`,
`metadata/`, `docs/DATASETS.md`, and `THIRD_PARTY_NOTICES.md`.

## Gate

`PASS_LICENSE_CANDIDATE_WITH_AUTHOR_METADATA_TODO`

Before public release, the human maintainer must confirm the final copyright
holder/author order and replace the explicit CITATION TODO. No dataset, model,
or dependency is relicensed by the repository license.
