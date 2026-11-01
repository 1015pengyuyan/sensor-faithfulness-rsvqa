# Cross-sensor conflict diagnosis in remote-sensing VQA

Reproduction materials for the **V14.1 working manuscript**: a retrospective
behavioral diagnosis on two remote-sensing datasets and eight fixed checkpoints.
Controlled replacement of the **non-requested** image separates target-label
retention, reverse-label answers, abstention and parser failure. A second
same-label (congruent) replacement tests additional reversals beyond ordinary
replacement sensitivity. Low reversal rates alone are not faithfulness evidence.

## Versions and access

This review branch prepares **v2.0.0-eight-model**, a draft release pending owner
review—not yet a public extension record. The public
[v1.0.1-paper](https://github.com/1015pengyuyan/sensor-faithfulness-rsvqa/releases/tag/v1.0.1-paper)
is the unchanged historical reproduction version, not the eight-model extension.
No final-author manuscript PDF or invented citation metadata are supplied.

Start with [v2 reproduction](docs/V2_REPRODUCTION.md),
[data and licensing](docs/V2_DATA_AND_LICENSES.md), and
[full revisions and environments](configs/v2/models.json).

## Eight checkpoints

| Checkpoint | Official source | Native interface |
|---|---|---|
| Qwen2.5-VL-3B-Instruct | [Qwen](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Ordered multi-/single-/no-image |
| Qwen2.5-VL-7B-Instruct | [Qwen](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Ordered multi-/single-/no-image |
| LLaVA-OneVision-Qwen2-7B-ov | [LLaVA](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) | Native ordered images; inclusion policy below |
| EarthMind-4B | [EarthMind](https://huggingface.co/sy1998/EarthMind-4B) | Named SAR `image` / optical `rgb_image` |
| Qwen3-VL-8B-Instruct | [Qwen](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Native ordered multi-/single-/no-image |
| InternVL3.5-8B | [OpenGVLab](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | Native chat and image-context tokens |
| Mantis-8B-SigLIP-Llama3 | [TIGER-Lab](https://huggingface.co/TIGER-Lab/Mantis-8B-siglip-llama3) | Native Mantis image markers |
| Idefics2-8B | [Hugging Face](https://huggingface.co/HuggingFaceM4/idefics2-8b) | Native typed chat content |

Eight checkpoints span seven named families, with shared components; model count
is not independent architecture diversity. Full 40-character revisions, separate
licenses and observed environments are in `configs/v2/models.json`. Native
templates/EOS/processors differ: visual-token budgets and tensors are not identical.

## Cohorts and interfaces

| Analysis | Questions | Physical target sources |
|---|---:|---:|
| M4-SAR | 420 | 420 |
| BigEarthNet.txt primary | 321 | 282 |
| BEN whole-source exclusion | 313 | 275 |
| BEN second-congruent subset | 215 | 195 |
| BEN second-congruent after exclusion | 209 | 189 |

BEN historical allocation was 341 questions/295 sources. Removing 20 technical
smoke questions gives 321/282. Excluding seven overlapping **whole sources removes
eight additional questions**, giving 313/275; 28 is cumulative relative to 341.

Generic M4 primary input is **requested-first**. BEN primary is
**optical-first/SAR-second**: requested optical occupies slot 1 and requested SAR
slot 2. EarthMind has named branches, not generic position/single-/no-image controls.

## Evidence layers

- **Unified original-prompt 128-token main matrix:** eight models, two datasets,
  deterministic decoding and frozen parser. **139,092 scientific condition-input
  records + 416 replays**. Calls are not independent samples. Full-population A0
  balanced accuracy describes capability, separately from conditional CIFR/RSR.
- **Post-hoc robustness:** complete eight-model multiplicity family, applicable
  requested-only capability subsets and BEN source exclusion. Subset intervals are
  unadjusted; approximate Bonferroni tail precision does not guarantee simultaneous coverage.
- **Post-hoc BEN position correction:** request/ignore references repaired without
  changing task, labels, images or decoding. Four models have complete same-round
  Main/Alternative data. Qwen3B/Mantis cross-run reuse is supplementary. LLaVA's
  unarchived bridge is not authenticated for paired position inference; EarthMind is N/A.

All models and zero, reverse, uncertain and non-estimable results are retained.
Effects depend on model, sensor and position—not a universal eight-model effect.
Different model-specific eligible populations do not establish strict faithfulness
ranks. Intervals condition on fixed distractors and exposed sources; independent
parent-scene/geographic validation remains unestablished.

### LLaVA scientific inclusion is not technical re-admission

The original 99% semantic-parse gate remains unchanged: 380/384 (98.9583%) was
**smoke FAIL / NOT_ADMITTED**. A dated scientific-inclusion supplement retained
format failures as outcomes for **all eight models**, without waiving engineering
gates. Original-prompt LLaVA predictions are included under that policy. Revised-
format compatibility is separate and does not change the old decision.
Failures/abstentions count as wrong for capability, not as directed reversals.

## CPU reproduction—no inference or new random draws

```bash
python -m pip install -e .
python -m unittest discover -s tests -p 'test_public_v2.py' -v
python reproduction/v2/reproduce.py demo --out reproduced
# Extract draft attachments to reproduction_data/, then:
python reproduction/v2/reproduce.py all --data reproduction_data --out reproduced
```

See [full instructions](docs/V2_REPRODUCTION.md). This validates exports, re-scores
saved predictions, verifies intervals against saved endpoint arrays, and rebuilds
tables/vector plots. Missing assets fail explicitly. Frozen task text and internal
A0/A2/A3 IDs remain in metadata; display terminology does not rename artifacts.

## Terms and historical materials

Repository-authored code: [Apache-2.0](LICENSE). Checkpoint/dataset/third-party terms
are separate; see [notices](THIRD_PARTY_NOTICES.md) and
[v2 terms](docs/V2_DATA_AND_LICENSES.md). Obtain images/weights officially.
No protected or new official-test access is granted.

Legacy scripts/configs/fixtures and `results/paper_summary/` remain historical,
not current eight-model tables. [Citation metadata](CITATION.md) remain pending.
