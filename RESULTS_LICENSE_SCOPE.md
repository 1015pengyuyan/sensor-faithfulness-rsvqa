# Rights scope for v2.0.0-eight-model

Authorized by the repository owner on 2026-11-02, **only to the extent the project
rights holders have authority to license the contributions**.

1. Project-authored code and code documentation retain the existing Apache-2.0
   license in `LICENSE`. This notice does not change that license.
2. Project-owned statistical results, chart content, validation reports, added
   task wording, and selection/allocation/arrangement contributions are licensed
   under **Creative Commons Attribution 4.0 International**:
   https://creativecommons.org/licenses/by/4.0/legalcode.en.
   Attribute the project as `sensor-faithfulness-rsvqa`, identify the version,
   link https://github.com/1015pengyuyan/sensor-faithfulness-rsvqa/releases/tag/v2.0.0-eight-model,
   and indicate changes. This attribution identifier is not a manuscript author list.
3. Inherited M4 data fields retain CC BY 4.0 and their original source credit;
   inherited BEN data fields retain CDLA-Permissive-1.0, provider credit, and
   modification notices. Project authorization does not replace upstream rights.
4. Raw model answers, equivalent decoded answer content, and generated token
   content are **not included in the project's CC BY grant**. See
   `MODEL_OUTPUT_NOTICES.md`; no unrestricted training/commercial reuse promise
   or universal eight-model prohibition is added.
5. Third-party font software, upstream model implementations, checkpoint weights,
   original imagery, and license texts are not relicensed under project CC BY.
   Public chart PDFs embed DejaVu subsets under their own preserved font terms.

## File/field priority (most specific rule wins)

- `raw_output`, decoded/parsed model-answer content (`parsed_answer` or `answer`
  where it holds a model response), and generated token content: model-output notice.
- Source questions, gold/annotation labels, class strings, image/product/source
  identities and source pairing fields: inherited dataset terms, chosen by the
  row's dataset and its fixture/source mapping.
- A `user_prompt` or derived question may mix source question/class content with
  project-added instructions. The project grant covers only the added wording;
  it does not replace the license on the incorporated source portion.
- Fixed donor choices and arrangement: project CC BY contribution, with the
  underlying provider IDs/labels still governed by their source terms.
- Project numerical estimates/caches, answer-format annotations, run/provenance
  observations, own chart geometry and validation reports: project CC BY, subject
  to any inherited or model-content fields within the same container.
- License texts and third-party font subsets: their own notices, not project CC BY.

Each attachment contains `LICENSE_SCOPE.json`, `MATERIAL_LICENSES.json`, source
attribution/change notes, and `MODEL_OUTPUT_NOTICES.md`. **No ZIP as a whole is
declared to have one license.** Facts/short strings are not asserted to acquire
copyright merely by inclusion. Rights of other parties are not granted here.
