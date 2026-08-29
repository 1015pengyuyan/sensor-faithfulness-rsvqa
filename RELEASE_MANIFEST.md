# Release manifest

- Repository name: `sensor-faithfulness-rsvqa`
- Candidate version: `1.0.0-paper`
- Candidate tag: `v1.0.0-paper`
- Candidate date: `2026-08-29`
- Public push: not performed
- Git content commit SHA: `c2101ba60cee1e90f93963c97142eca2e46db691`
- Author order: `TODO` pending human confirmation
- Repository URL: intentionally unset pending account/organization selection

## Fixed model revisions

| Model | Revision | Role |
|---|---|---|
| Qwen2.5-VL-3B-Instruct | `66285546d2b821cf421d4f5eb2576359d3770cd3` | general VLM control |
| EarthMind-4B | `f17c66b01e540268f3862a540e84a3f7ee329cd3` | multi-sensor EO model |
| LLaVA-OneVision-7B | `0d50680527681998e456c7b78950205bedd8a068` | independent general multi-image architecture control |

## Fixed dataset revisions

| Dataset | Revision/version | License |
|---|---|---|
| M4-SAR | `e1609a2c7d2cacda10d4ad1e895df9627f37c236` | CC BY 4.0 |
| BigEarthNet.txt | `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316` | CDLA-Permissive-1.0 |
| BigEarthNet v2 | `2.0.0`, DOI `10.5281/zenodo.10891137` | CDLA-Permissive-1.0 |

## Binding hashes

| Binding artifact | SHA256 |
|---|---|
| M4-SAR fixture | `94b71c918476a266a901ee7d3430f16d3c104369c6ce29c1cce1a4e50555baef` |
| BigEarthNet fixture | `262ebdcc3b09da82ed4508cae7edce9235857b864acd81b2130078be51fda6cb` |
| BigEarthNet rendering config | `deb9c5e958fb2c0578603000854417444f57c35b0341ca93bcdcbac7284f0b08` |
| Cross-dataset paper summary | `04752168a58831a5200dbdb1c4d7078b74950e8f359e293e3722e0906ac437a5` |
| Robust-flip paper summary | `359cece0fb9fd0ae240af8293c2bd448ef4c9aa7223161b2507fc10a2eab4401` |

The tag will be created locally after all checks pass. Public repository creation
and push require separate human approval.
