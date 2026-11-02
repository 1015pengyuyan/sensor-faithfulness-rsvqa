# Per-material license matrix

Owner authorization: 2026-11-02, limited to contributions the project can license.
This table does not grant third-party rights or give any mixed ZIP one license.
Publication remains a separate, checksum-verified step; code integration alone
does not publish draft attachments.

| Actual distributed material | Source / scope | Applicable terms and basis |
|---|---|---|
| Project code and code documentation | Repository-authored wrappers, CPU helpers, configuration documentation | Existing [Apache-2.0](../LICENSE), unchanged. Third-party implementations are not bundled. |
| Project-owned statistics, caches, chart geometry and validation reports | Own portions of attachments 02–05 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en), only rights held; attribute project identifier, version, source link and changes. See [scope](../RESULTS_LICENSE_SCOPE.md). Embedded inherited/model content retains specific rules below. |
| Added task wording, selection, donor allocation and arrangement | Own contributions in fixtures and provenance, attachments 01–04 | Project contribution CC BY 4.0; incorporated questions, labels and IDs retain source terms. Mixed user prompts are not wholly relicensed. |
| Inherited M4 questions, labels and identifiers | [Fixed M4 card](https://huggingface.co/datasets/wchao0601/m4-sar/blob/e1609a2c7d2cacda10d4ad1e895df9627f37c236/README.md) | CC BY 4.0; retain provider credit, license link and changes under section 3(a). [Source notices](../UPSTREAM_DATA_NOTICES.md). Raw image rights are not cleared here. |
| Inherited BEN questions, labels, IDs and pairing fields | [Fixed BEN.txt card](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/blob/72d865f2146f0a85b720f7f3ca1cdbaeafc3d316/README.md), [BEN v2](https://doi.org/10.5281/zenodo.10891137) | CDLA-Permissive-1.0; provider credit, agreement reference and change notices retained. Sections 3.1–3.2 govern Data/Enhanced Data; Results under sections 1.10/3.4 do not make whole prediction files pure statistics. |
| Answers, parsed answer content and generated tokens | All eight full revisions in attachments 01/03 | Independent [MODEL_OUTPUT_NOTICES](../MODEL_OUTPUT_NOTICES.md). No project CC BY grant over generated content, universal output-use ban or unlimited training/commercial promise. Qwen3B and Mantis/Llama 3 conditions remain model-specific. |
| Font software in public PDFs | DejaVu Sans/Bold subsets verified against official Matplotlib v3.9.2 | [DejaVu/Bitstream/Arev notice](../licenses/DEJAVU_FONT_LICENSE.txt), not project CC BY. No standalone font binaries. |
| License texts and attributions | Notice files | Their own terms; copying notices does not relicense them under project CC BY. |

## Five mixed packages

| ZIP | Material classes |
|---|---|
| 01 main predictions/fixtures | Model answers, inherited M4/BEN fields, own task/allocation/provenance contributions |
| 02 statistics/cached draws | Own numerical results, inherited source/per-question fields where present |
| 03 position correction | Answers, inherited fields, own pairing/correction/validation contributions |
| 04 figure source data | Own chart content, inherited fields, separately licensed PDF font subsets |
| 05 CPU checks | Own validation reports; historical receipts labelled historical, not new analysis |

Each ZIP has a common LICENSE_SCOPE.json and source/model notices, plus its own
licenses/<package>/MATERIAL_LICENSES.json with file SHA256 and field overrides.
Packages extract together without replacing those specific maps.
[RELEASE_ATTACHMENTS.json](../reproduction/v2/RELEASE_ATTACHMENTS.json) identifies
current archives; previous draft archives remain in the local original archive.

## Font decision and non-distributed assets

Arial resolves from a Windows font directory. That observation and embedding flags
do not establish source/authoring entitlement. Original PDFs stay in local V14.1
and original archives. Prepared public PDFs/SVGs/PNGs are independent DejaVu
display copies: scientific values unchanged, not image-byte identity.

No weights, third-party implementations, raw sensor images, wheels, standalone
fonts, connection files or author-placeholder paper PDF are distributed.
Acquisition requirements for upstream assets are separate from actual distributed
materials. Full GPU environment locks remain unavailable; observed versions and
this limitation are retained, not replaced by invented locks. Publication did not
create environments or rerun inference.

No owner-selectable license item remains pending for the listed own contributions.
This does not warrant all third-party rights, grant new protected/test access or
promise unrestricted generated-answer reuse.
[Source and font evidence](V2_LICENSE_EVIDENCE.json).
