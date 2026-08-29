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

The repository's Apache-2.0 candidate applies only to repository-authored code
and documentation. It does not relicense fixture source metadata, datasets,
model checkpoints, or Python dependencies.
