# Eight-checkpoint reproduction guide

## Status and layout

This branch prepares the V14.1 evidence for review. `v2.0.0-eight-model` is a
**draft release**, not an assertion that extension assets are publicly accessible.
The historical [v1.0.1-paper](https://github.com/1015pengyuyan/sensor-faithfulness-rsvqa/releases/tag/v1.0.1-paper)
reproduces the earlier study only; it does not cover the eight-model extension.
The unversioned legacy `scripts/`, `configs/`, `fixtures/` and `results/paper_summary/`
are preserved historical assets. Use `reproduction/v2/` and `configs/v2/` for this update.

Unzip the four draft attachments into the **same**, initially empty directory:

```text
reproduction_data/
  main/                  # 16 stage exports, original-prompt unified 128-token outputs
  fixtures/              # prompts, labels, relative image identities and allocations
  analysis/original/     # all accepted endpoints, saved source draws and endpoint arrays
  robustness/            # separately labelled post-hoc checks, all eight models
  position/              # corrected stages and independent acceptance receipts
  position_analysis/     # initial corrected-position analysis (original FAIL retained)
  position_closeout/     # final common-set rates and answer redistribution
  figures/               # accepted quantitative plots and numerical source data
  manifests/             # public file hashes and original-source hashes
```

No weights, third-party image bytes, credentials, server connection files,
protected allocation contents or author-placeholder manuscript PDFs are supplied.
Original archives are unchanged. The public schemas remove operational locations;
raw responses, parsed answers, parser status, scientific values and row order are
not rewritten. `EXPORT_PROVENANCE.json` connects each export to its original SHA256.

## Clean CPU reproduction

Python 3.10+; install the repository's CPU package in a separate environment:

```bash
python -m venv .venv
# Activate .venv using your operating system's standard command.
python -m pip install -e .
python -m unittest discover -s tests -p 'test_public_v2.py' -v
python reproduction/v2/reproduce.py demo --out reproduced
python reproduction/v2/reproduce.py all --data reproduction_data --out reproduced
python reproduction/v2/rebuild_article_figures.py --data reproduction_data --out reproduced/article
```

`all` verifies every attachment file, validates fixture metadata, re-parses every
saved main-matrix call with the frozen parser, reproduces A0 scoring, verifies
accepted percentile intervals against **saved** endpoint arrays, and rebuilds
tables and vector capability plots. It never loads a model or produces random
draws. Missing files/dependencies or mismatched hashes fail explicitly.
The fixture check is metadata-only without locally acquired images; it does not
claim that absent image bytes were checked. Archived route/processor evidence is
provided separately. Quantitative figure source data and accepted vector figures
are in the figure attachment. The task illustration containing dataset pixels is
not redistributed.

`frozen_metrics.py` mechanically preserves the accepted source-contribution,
replicate aggregation, eligibility and percentile definitions. The original
source hash is recorded in its header. Saved source multiplicities are shared
across models; zero-denominator draws remain invalid and are counted, not redrawn.
No tail precision is created by replaying these arrays.

## Data acquisition and fixture reconstruction

Obtain data under the original terms from the pinned official sources in
`configs/datasets.yaml`; obtain checkpoints at the **full revisions** in
`configs/v2/models.json`. The release supplies all fixed selections and donor
assignments, so reconstruction means resolving the supplied relative paths under
a user-configured image root, not reselecting samples from model outputs.
Use the rendering specification in `reproduction/v2/BIGEARTHNET_RENDERING.json`.
Image-file SHA256 in each fixture is the acceptance target. Different decoder or
renderer output must not be silently accepted as the frozen pixels.

M4 sanitized paired IDs do not by themselves recover parent products, scenes,
times or footprints. BEN benchmark split lineage and historical access do not
authorize fetching new official-test or protected assets. If an exact permitted
asset cannot be obtained, report that limitation; do not substitute another image.
See `docs/V2_DATA_AND_LICENSES.md` for source and redistribution boundaries.

## Native inference interfaces

This CPU reproduction does not run inference. First-party native adapter sources
and their provenance are provided under `reproduction/v2/native/`; obtain any
required third-party packages/code from their official pinned sources, not from
this repository. Independent model environments must not be mixed. Observed
environment versions are recorded per model; historical environment acceptance is
not a guarantee that an arbitrary new host is technically admitted.

Frozen tasks and output instruction are supplied verbatim in public fixtures.
The prompt's `yes|no|abstain` schema example is **not** repaired for the main study.
LLaVA's separate revised-format compatibility test is not mixed into these outputs.

## Counting and scientific interpretation

Main matrix: **139,092 scientific condition-input records + 416 replays** across
16 stages. These are not independent sample sizes. M4 has 420 questions/420 target
sources. BEN has 321 questions/282 sources; excluding seven whole sources removes
eight more questions, giving 313/275. The C1 control subset is 215/195 (209/189 after
exclusion). Replays are excluded from all scientific denominators.

Corrected position collection: v8/v9 **29,980 scientific records + 260 replays**;
the earlier LLaVA alternative-position stage is **2,998 + 26**, counted separately.
Bridge calls are technical evidence, not additional scientific samples. Main-stage
predictions referenced by more than one analysis are not duplicate observations.

Four same-round complete-position models: Idefics2, Qwen2.5-VL-7B, Qwen3-VL-8B,
InternVL3.5-8B. Qwen3B and Mantis cross-run reuse is supplementary and discloses its
limited bridge evidence. LLaVA's unarchived bridge is not authenticated for paired
position inference. EarthMind has no generic position factor.

Position eligibility is **question × requested sensor**, requiring correct A0 at
both positions. Interaction requires the same question correct in all four cells.
Source bootstrap carries only eligible questions, not every question from an
eligible source. Direction is actual **slot 2 − slot 1**; BEN Main is optical slot 1
but SAR slot 2, so Alternative − Main is not a common direction.

## Frozen and post-hoc analyses

Original inference uses 10,000 source-cluster draws, seed 20261016, with all
questions/directions/conditions/positions of a source travelling together.
Replicates are averaged before question-weighted pooling. Model differences share
draws. Conditional CIFR/RSR denominators differ by model and are always reported.
Failure and abstention count as wrong for fixed-population accuracy; they are not
directed reversals. Non-requested-only correctness uses the requested-image gold,
not distractor recognition ability.

The robustness and position packages are post-hoc, not independent confirmation.
Original BEN alternate-position endpoints retained in the original archive are
pre-correction historical outputs; they are superseded for position inference by
the corrected-position/final-closeout packages, not combined with them.
The 32-member multiplicity family includes all 8 models × 2 datasets × 2 sensors.
Approximate Bonferroni-expanded intervals have limited empirical tail precision;
they are not strict simultaneous-coverage guarantees. Requested-only capability
subsets have unadjusted intervals and do not identify evidence use inside dual-image
inference. EarthMind single-image/text-only controls are unavailable.

Low reverse rates are not alone faithfulness evidence: retain capability,
target-label retention, opposite-label responses, abstentions and parse failures.
Point-order A0 balanced accuracy is capability description, not a faithfulness
ranking. Do not infer interactions from differences in interval significance.
Small-sample empirical `[0,0]` intervals do not prove a population effect is zero.
All inference is conditional on fixed donors and historically exposed sources;
parent-scene/geographic independence remains unestablished.
