# Result-attachment license matrix

Checked: 2026-11-02. **This inventory is not a new license grant.** Code and
documentation are public on `main`; all five result attachments remain draft.
The repository's existing Apache-2.0 license does not automatically license their
datasets, model answers, numerical results, figures or embedded font software.
Only the repository owner and relevant rights holders may authorize additional
grants. This is a source-grounded inventory, not a determination of legal title.

## Actual distributed material

| Material | Source and location | Applicable terms and concrete basis | Still for the owner to decide |
|---|---|---|---|
| Project-authored code and documentation | Git `main`: first-party adapters, CPU/statistical/plotting helpers, configuration documentation | Existing [repository LICENSE](../LICENSE), Apache-2.0; preserve applicable attribution and change notices under section 4. External packages and source-data fields are outside this grant. | No replacement code license proposed. Confirm rights-holder authority if not already established; author names are not invented. |
| Project-created statistics, numerical caches, figures and validation reports | Pure-result portions of attachments 02–05, including estimates, cached draws and chart geometry | No separate result/figure license has yet been adopted. CDLA-Permissive-1.0 sections 1.10 and 3.4 distinguish computational Results from reproduced Data; this does not make whole mixed archives license-free. Facts need not all be copyrightable, but that does not establish a blanket public-domain status. | Choose the grant, if any, for project-owned result/figure/report contributions. CC BY 4.0 or CC0 are possible choices for rights actually held, **not selected here**. Exclude third-party material from any such grant. |
| M4-derived questions, labels, IDs and allocation metadata | M4 portions of attachments 01–03 and source-data portions of 04; pinned `wchao0601/m4-sar` card | The [fixed M4 card](https://huggingface.co/datasets/wchao0601/m4-sar/blob/e1609a2c7d2cacda10d4ad1e895df9627f37c236/README.md) declares CC BY 4.0. Preserve provided source attribution, license link and change indications under [CC BY 4.0 section 3(a)](https://creativecommons.org/licenses/by/4.0/legalcode.en). Selecting IDs or deriving questions does not make inherited fields project-owned. | Identify the license scope for the project's added task wording/selection/arrangement, if separately licensed; do not replace the upstream grant or remove its attribution. No claim of cleared image redistribution. |
| BEN-derived questions, labels, IDs, pairings and allocation metadata | BEN portions of attachments 01–04; fixed BigEarthNet.txt and BigEarthNet v2 sources | The [fixed BEN.txt card](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/blob/72d865f2146f0a85b720f7f3ca1cdbaeafc3d316/README.md) and [BEN v2 record](https://doi.org/10.5281/zenodo.10891137) declare CDLA-Permissive-1.0. [Sections 3.1–3.2](https://cdla.dev/permissive-1-0/) require the agreement reference, retained provider credit and change notices for Enhanced Data. Full question/label records are not reclassified as pure statistics. | Decide any additional license for project-authored arrangement/contributions, subject to those upstream obligations. Do not apply a pure-results exemption to entire prediction/fixture files. |
| Original model answers and parsed-answer records | `raw_output`, parsed/status fields and replay-linked answers in attachments 01 and 03, across all eight pinned checkpoints | Model-generated, not human-written answers. Keep `model_id`, full revision, input provenance and original responses. A checkpoint's Apache label does not itself grant rights over every output. Qwen3B's fixed [Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/LICENSE) sections 2 and 4(b) distinguish model-use permission and conditions involving outputs used to develop released models. Mantis declares Llama 3; see the output-use conditions below. | Confirm the publication/reuse notice and any rights that can actually be granted for model-answer files. No blanket Apache/CC BY/CC0 grant is made, and no automatic claim of copyright ownership or public-domain status is made. |
| Third-party material actually present | Three quantitative PDFs in attachment 04 contain ArialMT/Arial-BoldMT font subsets; SVGs name fonts, PNGs contain rendered charts | Six embedded font objects were inspected: each has OS/2 `fsType=8` (editable embedding). [Microsoft's document-embedding FAQ](https://learn.microsoft.com/en-us/typography/fonts/font-faq#document-embedding) and [OpenType fsType specification](https://learn.microsoft.com/en-us/typography/opentype/spec/os2#fstype) distinguish document embedding from standalone font redistribution. Font flags alone do not prove lawful acquisition. | Confirm lawful font-source/authoring entitlement before publishing the PDFs. Any grant for chart content excludes font software and does not authorize extracting/relicensing it. No fonts were replaced or figures regenerated in this review. |

## Mapping the five mixed-content attachments

| Attachment | Material classes to keep separate |
|---|---|
| `01_main_predictions_fixtures.zip` | Model answers + inherited M4/BEN question/label/ID fields + project-added allocation/provenance metadata. Not a single-license package. |
| `02_statistics_cached_draws.zip` | Pure numerical results and cached draws, but also normalized per-question records/source mappings with inherited data fields. Treat these separately. |
| `03_position_correction.zip` | Model answers + inherited fields + project-added pairing/correction/acceptance metadata; same per-material distinction as 01. |
| `04_figure_source_data.zip` | Own chart content/statistics, mixed source-data fields, and the PDF font subsets described above. No sensor imagery or standalone fonts. |
| `05_cpu_reproduction_checks.zip` | Project-created validation reports, logs and numerical checks. No automatic extension of the code license to these reports. |

The exact filenames, byte counts and SHA256 values remain in
[`RELEASE_ATTACHMENTS.json`](../reproduction/v2/RELEASE_ATTACHMENTS.json); they are
unchanged by this licensing review. Package provenance identifies original and
public-export hashes. This inventory does not rewrite any scientific field.

## Upstream assets **not** distributed

- **Checkpoint weights, original sensor imagery, model implementations, Python
  wheels and standalone font files:** absent from all five archives. There are
  293 ZIP members (including repeated manifests), three rendered-chart PNGs and
  no dataset JPG/TIFF images or weight/wheel files. First-party calling wrappers
  in Git are not the upstream modeling implementations.
- **Qwen2.5-VL-3B:** its fixed license permits model use for research/evaluation
  under section 2, and section 4(b) specifies attribution when outputs contribute
  to developing distributed models. Neither statement is a blanket output license.
- **Mantis/Llama 3:** the [fixed Mantis card](https://huggingface.co/TIGER-Lab/Mantis-8B-siglip-llama3/blob/28cc1fd9bccdce2c4276a9a784c54dcc58836881/README.md)
  declares `llama3`. The [official Llama 3 agreement](https://github.com/meta-llama/llama3/blob/main/LICENSE),
  notably section 1(b)(v), restricts using outputs to improve other large language
  models except Llama 3/derivatives. Material-redistribution notices and output-use
  conditions are distinct; do not infer that every model-weight obligation is
  automatically a license on the answer strings. The observed license-text SHA256
  is recorded in the evidence file, because the reference URL is not immutable.
- **Other six checkpoint declarations:** the fixed cards declare Apache-2.0
  (EarthMind also requires attention to bundled upstream notices). These are
  acquisition/use references, not proof of an unrestricted grant over generated
  answers. Some fixed snapshots lack a standalone LICENSE file; the earlier 404
  records in `configs/v2/models.json` are retained, not recast as full-text checks.
- Dataset-card declarations do not clear every upstream raw-image right or grant
  new official-test/protected-asset access. No such assets were obtained here.
- Observed direct runtime versions are not a complete GPU environment lock. No
  new environment was created, no model loaded and no prediction rerun.

## Required owner decisions before result publication

1. Select the license (or explicit limited grant) for **project-owned statistical,
   chart and validation-report contributions**; confirm the relevant rights holders.
2. Specify whether that grant also covers the **project's own fixture wording and
   allocation/arrangement contributions**. Preserve inherited CC BY/CDLA fields and
   notices separately; they are not an owner-selectable replacement license.
3. Approve a **model-answer publication/reuse notice**, taking account of per-model
   terms (especially Qwen3B and Mantis/Llama 3), and confirm the font-source
   entitlement for the already embedded PDFs. Do not promise unrestricted AI
   training/relicensing rights unless those rights have actually been established.

Until these decisions, the result Release remains draft. Code integration does
not settle them, and this table neither chooses a new license nor publishes data.
Official-source observations, hashes and the font inventory are recorded in
[`V2_LICENSE_EVIDENCE.json`](V2_LICENSE_EVIDENCE.json).
