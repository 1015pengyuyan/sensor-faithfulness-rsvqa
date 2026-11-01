# Data, code and checkpoint terms

Repository-authored code is covered by the repository Apache-2.0 license.
This does **not** relicense checkpoints, datasets or third-party model code.
No model weights, third-party images or third-party implementation source are
redistributed. Official access and license links are in `configs/v2/models.json`.
Keep applicable Llama 3 license/attribution and acceptable-use requirements when
obtaining Mantis; the absence of a Hub access gate is not absence of license terms.
Qwen2.5-VL-3B and Qwen2.5-VL-7B do not share identical checkpoint license terms.

| Material | Official source | Terms / public content here |
|---|---|---|
| M4-SAR | [pinned dataset](https://huggingface.co/datasets/wchao0601/m4-sar/tree/e1609a2c7d2cacda10d4ad1e895df9627f37c236) | CC-BY-4.0; selected IDs, questions/labels and allocations only; attribution retained |
| BigEarthNet.txt | [pinned dataset](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/tree/72d865f2146f0a85b720f7f3ca1cdbaeafc3d316) | CDLA-Permissive-1.0; selected metadata only |
| BigEarthNet v2 | [official data record](https://doi.org/10.5281/zenodo.10891137) | CDLA-Permissive-1.0; no image distribution or new test access |
| Model outputs and derived statistics | Draft v2 attachments | Repository-generated research evidence; retain model/data source attribution and applicable upstream terms |
| Manuscript PDF | Not included | Author metadata pending; no guessed authors, DOI, journal or publication claim |

The draft record is prepared for owner review only. It is not yet a public
extension release. Historical public assets remain at `v1.0.1-paper` unchanged.
Author information and any final dataset-output licensing choice should be
confirmed before publication; do not treat the code license as a blanket grant
over third-party-origin metadata or checkpoint-generated outputs.

## Public export rules

- Explicit field lists preserve scientific IDs, labels, task strings, image
  identities/order, original responses, parsed answers and format/answer statuses.
- Operational absolute paths, accounts, host addresses, credentials and environment
  locations are omitted. Original-source SHA256 and public-file SHA256 are retained.
- Saved numerical CSV/JSON/NPZ files are complete across models, including null,
  zero, negative and uncertain results. No filtering by effect or answer quality.
- Replay and scientific rows remain distinguishable; old controller FAIL records
  and supplemental acceptance are not recast as the same status.
- Privacy checks reject suspect raw answers instead of rewriting them. No source
  archive, existing local connection file or historical tag is edited.

Independent-source crosswalks and protected-allocation membership have not been
provided for independent geographic validation. Internal requests and candidate
allocation files are not part of the release, and no requests are sent by this update.
