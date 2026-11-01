# Scientific evaluation inclusion policy supplement

Policy version: scientific-evaluation-inclusion-policy-20261025-v1  
Frozen at: 2026-10-25T06:38:58Z  
Scope: all eight checkpoints in unified_128_eight_model_20261018_v1, on M4-SAR and BigEarthNet.txt.

## Decision

After reviewing the original LLaVA development smoke evidence, the separate revised-format compatibility branch, and the frozen analysis definition, the project adopts this uniform scientific-evaluation policy: a model's output-format or semantic-parser failure is retained as an observed evaluation outcome and is not, by itself, a reason to exclude the model from the scientific matrix. This policy applies to all eight models; it is not a LLaVA-specific scoring exception.

This is a dated policy supplement to the frozen matrix, not a preregistered rule and not a retrospective rewrite of technical admission. It authorizes only the LLaVA original-prompt scientific rerun in this version; predictions for the other seven models are reused unchanged.

## Unchanged technical records and hard gates

The original LLaVA 99% semantic-parse technical threshold remains unchanged. Its original 384-call smoke remains FAIL at 380/384 (98.9583%), and its original-protocol admission remains NOT_ADMITTED. The original environment and preflight PASS reports also remain unchanged. The new-format compatibility result is a separate diagnostic branch and is not evidence that the original protocol passed technical admission.

Scientific execution still requires verified frozen fixture and image hashes, exact image identity and order, valid sensor routing, complete unique call coverage, reconciled actual generation/return/error counts, exact replay outputs and parsed answers, and verified GPU identity and memory. Any failure in these engineering gates stops the affected stage and is recorded; this policy does not waive them, fabricate an upstream PASS, or permit retries to improve outputs.

## Scoring and reporting

The frozen ranking definition is unchanged. On the assigned fixed cohort, A0 balanced accuracy continues to count abstentions and parser failures as incorrect. CIFR and RSR retain their frozen A0-eligible denominators and formulas. A parser failure is not treated as a target-reverse answer. For each condition, the new LLaVA report will separately give the main-call denominator, parser-failure count/rate, abstention count/rate, and strict-JSON count/rate; replay calls remain outside scientific denominators. Low CIFR alone is not evidence of robustness or sensor faithfulness.

No output is repaired, forced into JSON, selectively interpreted, retried, or removed. The frozen prompt, parser, fixtures, model revision, deterministic decoding, and 128-new-token limit remain unchanged. The separate revised-format compatibility branch remains separate and is not merged into the unified-protocol ranking.

## Reviewed evidence and provenance

- Original LLaVA environment report: PASS; SHA256 68d0b9ec355423dcc448355394ff6a98f3d6141c412c52204ea2199a545c9e8a.
- Original LLaVA preflight: PASS, 7/7 generation returns; SHA256 89e8771df664ed40ca4f74bd20598ef940318de774f295ffc1f630dfcf142872.
- Original LLaVA 384-call smoke: FAIL solely because 380/384 outputs met the semantic parser rule against the unchanged 0.99 threshold; 384/384 generation returns, zero generation errors, and all recorded engineering/replay gates passed. Strict JSON was 145/384 and was a separate diagnostic. Report SHA256 03a340c8a153b760540a65ed3afb7a3199ac31f3874055a915834bcfeb7e90f7.
- Original smoke predictions SHA256 1bfe127bb186fe9f30a5de4d525b210bb921e52c84ad1bb7f22e8dc9456e2203; call journal SHA256 6b514a67edb878e1f075f5c0d4df0a54d04a17aecd1af20121a72e8918e453c9.
- Separate revised-format compatibility branch: 7/7 preflight and 384/384 smoke semantic parses; strict JSON was 0/7 and 0/384. It remains compatibility-only and is not included as original-protocol evidence.
- Frozen matrix analysis states that abstentions and parser failures count as incorrect for A0 balanced accuracy. The read-only seven-model scorer maps parser statuses outside {json, recovered} to parse_failure; equality scoring therefore treats them as incorrect, while a parse failure is not a directed reverse-label event. The audited local scorer is descriptive_review_20261024_v1/summarize_frozen_predictions.py, SHA256 4c1e8dd1f2c948c7f46f16746816096670991015f06ac3610851432871e06f1d.

The source reports and predictions reside in the immutable L40 migration snapshot. This policy supplement and its execution records are additive; no historical report, prediction, fixture, source code, or public release is overwritten.
