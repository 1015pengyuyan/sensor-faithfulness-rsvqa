# Release result consistency audit

## Frozen input hashes

| Artifact | SHA256 | Status |
|---|---|---|
| M4-SAR confirmatory fixture | `94b71c918476a266a901ee7d3430f16d3c104369c6ce29c1cce1a4e50555baef` | exact match |
| BigEarthNet full external fixture | `262ebdcc3b09da82ed4508cae7edce9235857b864acd81b2130078be51fda6cb` | exact match |
| BigEarthNet rendering configuration | `deb9c5e958fb2c0578603000854417444f57c35b0341ca93bcdcbac7284f0b08` | exact match |
| Final cross-dataset summary | `04752168a58831a5200dbdb1c4d7078b74950e8f359e293e3722e0906ac437a5` | exact match |
| Final robust-flip results | `359cece0fb9fd0ae240af8293c2bd448ef4c9aa7223161b2507fc10a2eab4401` | exact match |

## Fixture gates

| Fixture | Rows | Sources | Duplicate IDs | Target/distractor overlap | Label violations | Missing requested evidence | Test refs |
|---|---:|---:|---:|---:|---:|---:|---:|
| M4-SAR | 3,360 | 420 | 0 | 0 | 0 | 0 | 0 |
| BigEarthNet | 2,728 | 341 | 0 | 0 | 0 | 0 | 0 |

## Automated checks

- Standard-library core tests: 7/7 passed, including deterministic PDF reproduction.
- CPU-only demo produces CIFR `0.8888888889`, RSR `0.1111111111`, and ASYM
  `0.3333333333`.
- Paper consistency checker: five frozen hashes matched; six unique
  dataset/model rows were present.
- Table 3 was rebuilt from the frozen six-row summary without recomputing
  results.
- Figure 2 and Figure 3 were rebuilt from the same frozen summary.

## Generated artifact hashes

| Artifact | SHA256 |
|---|---|
| `m4sar_summary.csv` | `deff96a798e93321490e56b3b26f5eeb3be460cb69c19c3aa628e299afc4dc4f` |
| `bigearthnet_summary.csv` | `5efca6d293e7d348de2d042384f0b7316d02b21569c096bae4e5822840ea5347` |
| `table3_reproduced.tex` | `a25d637f97cc487c2699fc82360b50ed73fc5fcb1f3635d33775ae266a364814` |
| Figure 2 PDF | `492f415c14904eb106d07840c0d3b5d8a7a8bb87cbba7016a82ab732ef3f610a` |
| Figure 3 PDF | `2b9ce4919ca170953a81e7559fb755ad4c383506f81e55bd279665eb6bad8d60` |

PDF artifacts are byte-stable under the pinned release environment because volatile
CreationDate/ModDate metadata are omitted. Cross-environment font differences may
still change rendering bytes; frozen numeric inputs and table values remain binding.

## Decision

`PASS_FROZEN_RESULT_CONSISTENCY`
