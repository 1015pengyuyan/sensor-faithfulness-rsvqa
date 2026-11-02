# Third-party notices

This repository does not redistribute third-party model weights, model custom
code, raw dataset images, or data archives.

The inference modules call public APIs from PyTorch, Transformers, and the
official model repositories. EarthMind's remote/custom model code is loaded
from its pinned official checkpoint by the user and remains subject to its own
license. No EarthMind, Qwen, or LLaVA source file is copied into this repository.

Metadata-only fixtures reference:

- M4-SAR (`wchao0601/m4-sar`), recorded as CC BY 4.0.
- BigEarthNet.txt (`BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`) and BigEarthNet v2
  (DOI `10.5281/zenodo.10891137`), recorded as CDLA-Permissive-1.0.

The repository's existing Apache-2.0 license applies only to repository-authored code
and documentation. It does not relicense fixture source metadata, datasets,
model checkpoints, or Python dependencies.

## Eight-checkpoint extension

See `configs/v2/models.json` for full official revisions, observed dependency
versions and license source checks. Qwen2.5-VL-3B uses the Qwen Research License;
the other Qwen checkpoints have separate upstream declarations. Mantis checkpoint
use remains subject to Meta Llama 3 Community License and applicable linked
acceptable-use/attribution terms (including the upstream “Built with Meta Llama 3”
requirements where applicable). Its upstream Mantis implementation is obtained
separately at the recorded commit, not vendored here. Other checkpoint-specific
terms and bundled upstream notices still apply.

Several pinned snapshots do not contain a standalone LICENSE file. The source
check records this 404 explicitly and links their pinned model-card declarations;
it is not represented as a successfully inspected license text. In particular,
Qwen2.5-VL-7B's pinned card declares Apache-2.0 but supplies no LICENSE file in
that revision. Check applicable upstream terms before acquisition or redistribution.
This update distributes none of those checkpoints or their third-party code.

`reproduction/v2/native/` contains project-authored calling/telemetry wrappers,
not the upstream EarthMind, Qwen, InternVL, LLaVA, Mantis or Idefics modeling source.
The `pinned_internvl_128` wrappers are likewise project-authored. Their origins,
original hashes and limited module-location changes are recorded.

The result attachments have no blanket Apache-2.0 grant. See
`docs/V2_ATTACHMENT_LICENSE_MATRIX.md` for per-material terms and pending owner
decisions, including model outputs and the document-embedded Arial font subsets
in three quantitative PDFs. No standalone font software is supplied.
