# GitHub release readiness

## Required answers

| Question | Answer | Evidence |
|---|---|---|
| Does the repository contain third-party raw data? | No | Only metadata-only fixtures and identifiers are present. |
| Does it contain model weights? | No | Weight/archive extensions are ignored and scan count is zero. |
| Does it contain official test assets? | No | Both fixture scans report zero test references; no test files are present. |
| Does it contain sensitive information? | No detected sensitive information | Credential/private-infrastructure scan passed. |
| Does it contain personal absolute paths? | No | Only parameterized roots and relative paths are used. |
| Are fixtures consistent with the paper? | Yes | Both byte hashes match the frozen paper fixtures. |
| Are CIFR/RSR/ASYM reproducible? | Yes | CPU demo and unit tests pass; definitions match source-level pooling. |
| Can Table 3 be rebuilt? | Yes | `scripts/reproduce_table3.py` succeeds from the frozen summary. |
| Can Figures 2 and 3 be rebuilt? | Yes | Both PDF/PNG pairs are generated from the frozen summary. |
| Are model revisions fixed? | Yes | Three immutable commit hashes are recorded. |
| Are third-party licenses recorded? | Yes | Model and dataset notices are documented without relicensing. |

## Decision

- Technical release state: `PUBLIC_RELEASE_TECHNICALLY_READY`
- Local candidate tag: `v1.0.0-paper` (not pushed)
- Current public-release state: `WAITING_FOR_HUMAN_METADATA_AND_PUSH_APPROVAL`

The code and reproduction gates are ready. Before public publication, the human
maintainer must replace the explicit CITATION author TODO, choose the real
GitHub account/organization, review the Apache-2.0 candidate, and approve the
local release commit/tag for push. No GitHub repository has been created or
pushed by this preparation step.
