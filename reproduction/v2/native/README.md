# Native first-party components

These are the first-party adapters actually identified in the completed main
matrix. `SOURCE_PROVENANCE.json` separates original component SHA256 from the
portable module-location edit. All scientific class/function ASTs are unchanged;
only discovery of support modules is made relative to this directory.

This directory does **not** vendor Mantis, Transformers, checkpoint modeling
code, tokenizers, image data or weights. Install upstream implementations at
the revisions specified by `configs/v2/models.json`, in separate environments.
The `pinned_internvl_128` files are repository-authored native-interface/trace
wrappers, not copies of upstream modeling code; their loader verifies full hashes.

Entry components (explicitly pass `max_new_tokens=128`):

| Model | Component |
|---|---|
| Qwen2.5-VL-3B | `QwenTechnicalAdapter(QwenRunner(model_path, 128))` from `qwen_technical_adapter` / `qwen_frozen_runner` |
| Qwen2.5-VL-7B | `native_qwen25vl7b_128` |
| LLaVA-OneVision | `llava_native_128` |
| EarthMind | `native_earthmind_128` (SAR → image, Optical → rgb_image) |
| Mantis | `native_mantis_128` + official Mantis package |
| Idefics2 | `idefics2_native_128` |
| Qwen3 | `qwen3_native_128_bridge` + preserved `qwen3_native_adapter` |
| InternVL3.5 | `internvl_matrix_bridge` + pinned native/trace wrappers |

Some preserved lower-level components have historical defaults. The scientific
matrix always supplied 128 explicitly. Do not use a lower-level default as a
scientific configuration or change native visual processing to fit a new host.
Model/image/output paths are caller parameters, not machine paths in this bundle.
The public CPU reproduction does not call these inference components.
