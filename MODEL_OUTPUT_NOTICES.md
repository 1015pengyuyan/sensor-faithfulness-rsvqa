# Model-output publication and reuse notices

Version: v2.0.0-eight-model; checked 2026-11-02. These are original, automatically
generated research observations, not human answers. Preserve the model ID, full
revision, fixture/call identity, raw response, parser status and version provenance.
No answer was repaired or selected for publication by correctness/significance.

The project's CC BY 4.0 grant covers only its own contributions described in
`RESULTS_LICENSE_SCOPE.md`, **not a blanket grant over model-generated content**.
These notices do not decide whether a particular output is copyrightable, do not
promise unlimited commercial/training rights, and add no uniform ban across all
eight models. Upstream contractual conditions apply where the relevant agreement
binds the user; a weight license is not automatically an output copyright license.

| Checkpoint | Fixed revision | Verified source / relevant boundary |
|---|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | `66285546d2b821cf421d4f5eb2576359d3770cd3` | [Fixed Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/LICENSE): section 2 limits the grant for Materials to research/evaluation use; commercial use of Materials needs separate permission. Section 4(b) states attribution conditions when outputs contribute to developing distributed models. Do not turn this into an unsupported blanket license or eight-model non-commercial rule. |
| Qwen/Qwen2.5-VL-7B-Instruct | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | [Fixed card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/cc594898137f460bfe9f0759e9844b3ce807cfb5/README.md) declares Apache-2.0; no standalone LICENSE in this snapshot. This is not an explicit blanket grant over all generated answers. |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | `0d50680527681998e456c7b78950205bedd8a068` | [Fixed card](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf/blob/0d50680527681998e456c7b78950205bedd8a068/README.md) declares Apache-2.0. No additional output-use restriction is invented here. Its original technical FAIL/NOT_ADMITTED and dated scientific inclusion policy remain separate. |
| sy1998/EarthMind-4B | `f17c66b01e540268f3862a540e84a3f7ee329cd3` | [Fixed card](https://huggingface.co/sy1998/EarthMind-4B/blob/f17c66b01e540268f3862a540e84a3f7ee329cd3/README.md) declares Apache-2.0; bundled upstream notices must be considered when acquiring code/weights. No checkpoint code/weights are distributed here. |
| Qwen/Qwen3-VL-8B-Instruct | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | [Fixed card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/README.md) declares Apache-2.0; no inference that all answer rights have thereby been granted. |
| OpenGVLab/InternVL3_5-8B | `9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79` | [Fixed card](https://huggingface.co/OpenGVLab/InternVL3_5-8B/blob/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79/README.md) declares Apache-2.0. Code/checkpoint notices and generated-content rights remain distinct. |
| TIGER-Lab/Mantis-8B-siglip-llama3 | `28cc1fd9bccdce2c4276a9a784c54dcc58836881` | [Fixed card](https://huggingface.co/TIGER-Lab/Mantis-8B-siglip-llama3/blob/28cc1fd9bccdce2c4276a9a784c54dcc58836881/README.md) declares Llama 3. [Official Llama 3 agreement](https://github.com/meta-llama/llama3/blob/main/LICENSE), section 1(b)(v), addresses use of outputs to improve other LLMs (with its stated Llama 3/derivative exception); the agreement also incorporates its acceptable-use policy. Preserve this model-specific notice without imposing it on the other seven models or on inherited dataset fields. |
| HuggingFaceM4/idefics2-8b | `2c42686c57fe21cf0348c9ce1077d094b72e7698` | [Fixed card](https://huggingface.co/HuggingFaceM4/idefics2-8b/blob/2c42686c57fe21cf0348c9ce1077d094b72e7698/README.md) declares Apache-2.0. No unsupported new output license or common prohibition is supplied. |

The precise source observations and full-text/card distinctions are retained in
`configs/v2/models.json` and `docs/V2_LICENSE_EVIDENCE.json`. Third-party questions,
labels and imagery-origin content retain their own terms regardless of model ID.
Source imagery and checkpoints are not bundled. These notices document the
release's research provenance and verified conditions; they are not legal advice
or a warranty clearing all possible third-party rights.

## V3 eleven-model supplement (2026-11-20)

The table above is preserved as the historical eight-model record. The current
eleven-model matrix retains Qwen2.5-VL-3B, Mantis, Idefics2, EarthMind and
LLaVA-OneVision; the three other v2 checkpoints remain historical. The following
six configurations complete v3. Exact revisions/access settings are also in
`configs/v3/models.json`. No checkpoint code, weights or client binaries are bundled.

| V3 addition | Pinned source or recorded access configuration |
|---|---|
| Gemma 4 E4B-it | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it/tree/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c) |
| MiniCPM-V 4.5 | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5/tree/14a2004908140699317e2109e90573c1e6323c36) |
| GLM-4.1V-9B-Base | [zai-org/GLM-4.1V-9B-Base](https://huggingface.co/zai-org/GLM-4.1V-9B-Base/tree/5e8d1942554a9277eabcf9e83211bbf9c86136ee); distinct from GLM-4.6V-Flash |
| Phi-4-multimodal-instruct | [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/tree/93f923e1a7727d1c4f446756212d9d3e8fcc5d81) |
| GPT-6 Luna | Official Codex CLI; requested `gpt-6-luna`, reasoning effort `none` |
| Gemini 3.8 Flash | Official Antigravity CLI; requested `gemini-3.8-flash-low` |

The linked revisions identify archived provenance; no new general output-license
conclusion is inferred from a weight/model-card label. The two service labels are
client selections, not independently verified server checkpoint identities, and
their runs have no verified 128-token hard ceiling. Applicable model/service
agreements remain separate from the project's statistics license. This notice
does not promise training/commercial reuse rights or impose a blanket eleven-model
prohibition. Parsed answers and raw generated content remain outside the project's
CC BY grant; source-derived fields retain their dataset terms.
