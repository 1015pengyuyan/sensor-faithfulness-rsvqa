# Cross-sensor conflict diagnosis in remote-sensing VQA

Code and reproduction materials for the **eleven-model manuscript snapshot of
2026-11-19**: nine fixed open checkpoints and two official-client service
configurations. Replacing the non-requested image separates target-label
retention, reverse-label answers, abstention and parser failure. A same-label
(congruent) replacement measures additional reversals beyond ordinary replacement
sensitivity. Low reversal alone does not establish faithfulness.

## Current and historical versions

The v3 source adds portable CPU replay of the current normalized scientific
records, exact checkpoint/configuration identities, evidence hashes and current
methodological boundaries. Begin with [v3 reproduction](docs/V3_REPRODUCTION.md),
[evidence and limitations](docs/V3_EVIDENCE_AND_LIMITS.md), and
[data and licensing](docs/V3_DATA_AND_LICENSES.md).

The normalized evidence is distributed separately from Git through
[v3.0.0-eleven-model](https://github.com/1015pengyuyan/sensor-faithfulness-rsvqa/releases/tag/v3.0.0-eleven-model).
Verify downloads against the [attachment manifest](reproduction/v3/RELEASE_ATTACHMENTS.json)
before CPU replay. See [release verification](docs/V3_RELEASE_READINESS.md)
for the scope of that verification. No author-placeholder
manuscript PDF or unselected figure variant is part of this update.

The published
[v2.0.0-eight-model](https://github.com/1015pengyuyan/sensor-faithfulness-rsvqa/releases/tag/v2.0.0-eight-model)
and
[v1.0.1-paper](https://github.com/1015pengyuyan/sensor-faithfulness-rsvqa/releases/tag/v1.0.1-paper)
remain historical records. Their code, figures and scientific results are retained.
[V2 instructions](docs/V2_REPRODUCTION.md), `configs/v2/`, legacy scripts and
`results/paper_summary/` describe those versions rather than the current model set.

## Eleven evaluated configurations

| Model | Access/identity |
|---|---|
| Qwen2.5-VL-3B-Instruct | Fixed open checkpoint |
| Mantis-8B-SigLIP-Llama3 | Fixed open checkpoint |
| Idefics2-8B | Fixed open checkpoint |
| EarthMind-4B | Fixed open checkpoint; named SAR/optical branches |
| LLaVA-OneVision-Qwen2-7B-ov | Fixed open checkpoint; scientific inclusion policy below |
| Gemma 4 E4B-it | Fixed open checkpoint |
| MiniCPM-V 4.5 | Fixed open checkpoint |
| GLM-4.1V-9B-Base | Fixed open checkpoint; distinct from GLM-4.6V-Flash |
| Phi-4-multimodal-instruct | Fixed open checkpoint |
| GPT-6 Luna | Official Codex CLI, requested `gpt-6-luna`, effort `none` |
| Gemini 3.8 Flash | Official Antigravity CLI, requested `gemini-3.8-flash-low` |

[Full identities](configs/v3/models.json) retain nine 40-character checkpoint
revisions. Open inference uses native processors/templates, greedy decoding,
one beam and at most 128 new tokens. The two CLI runs have neither an
independently pinned server checkpoint nor a verified 128-token hard ceiling.
Their results are access-specific observations. Model count does not imply
independent architecture diversity or identical visual-token budgets.

The three other v2 checkpoints (Qwen2.5-VL-7B, Qwen3-VL-8B and InternVL3.5-8B)
remain in history and are not silently added to the current eleven-model matrix.

## Cohorts and evidence

| Cohort | Questions | Physical target sources |
|---|---:|---:|
| M4-SAR | 420 | 420 |
| BigEarthNet.txt primary | 321 | 282 |
| BEN whole-source exclusion | 313 | 275 |
| BEN second-congruent subset | 215 | 195 |
| BEN second-congruent after source exclusion | 209 | 189 |

The current normalized matrix contains **195,618 scientific condition-input
records**, excluding replays. Ten generic interfaces contribute 18,842 records
each; EarthMind contributes 7,198 named-interface records. These calls are not
independent samples. Each closed evaluation has 18,894 planned and available calls
including 52 replay pairs; parsed answers agree in 43/52 pairs for each service.
Source-cluster intervals do not include that generation variability.

M4 generic primary input is requested-first. BEN primary input is
optical-first/SAR-second, so requested SAR occupies slot 2. EarthMind's named
branches have no generic position or single-/no-image controls. Questions, sources
and model-specific A0-correct eligible sets are different denominators.

The original LLaVA smoke decision remains **380/384, FAIL / NOT_ADMITTED** under
the unchanged 99% semantic-parse gate. Subsequent scientific inclusion treats
format failures as observed outcomes; it does not reclassify technical admission.
Abstentions and failures count as wrong for capability and do not count as
directed reversals.

The sealed 2026-11-19 post-hoc CPU supplement covers single-image capability,
all-C1 contrasts, M4 common-question position differences, multiplicity
sensitivity, and a three-model BEN position audit. It is separate from the v3 main
evidence attachment. No new matched-modality GPU experiment was completed;
parent-scene dependency and precision qualification remain unresolved. See the
[evidence boundary](docs/V3_EVIDENCE_AND_LIMITS.md).

## CPU replay

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
# Obtain the corresponding evidence attachment and extract it to reproduction_data/v3/.
python reproduction/v3/reproduce.py verify --data reproduction_data/v3
python reproduction/v3/reproduce.py all --data reproduction_data/v3 --out reproduced/v3
```

The command verifies immutable evidence hashes, reconstructs **6,888 main point
estimates** and eligible denominators, checks **33 cohorts**, and reparses all
**113,052 available raw responses** with the frozen parser. The original five
normalized groups contain parsed annotations without raw text in this attachment.
Supplement endpoints and conditional confidence limits retain their archived
bytes; this command does not regenerate them. It performs no model inference or
new random draws and refuses to overwrite an existing output directory.

## Terms

Project-authored code retains [Apache-2.0](LICENSE). The authorized project-owned
contributions retain the [CC BY 4.0 field-specific scope](RESULTS_LICENSE_SCOPE.md).
Dataset fields retain [upstream terms](UPSTREAM_DATA_NOTICES.md). Parsed/raw model
answers have [separate notices](MODEL_OUTPUT_NOTICES.md), without a blanket reuse
grant. Obtain weights and imagery from their official providers; they are not
bundled. [Citation metadata](CITATION.md) remain pending a final author record.
