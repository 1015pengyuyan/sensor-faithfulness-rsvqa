"""Explicit public schemas. Never edit scientific responses during export."""
import hashlib
import json
import re

PRIVATE = re.compile(r'(?i)(?:(?<![A-Za-z0-9_])[A-Z]:[\\/]|/(?:home|root|mnt)/|(?:\d{1,3}\.){3}\d{1,3}|BEGIN [A-Z ]*PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,})')
PREDICTION_FIELDS = '''call_id call_status call_type fixture_id example_id model_id
dataset condition_id condition_family row_family requested_sensor requested_position
target_patch_id target_source_cluster_id target_label_audit_only expected_answer_audit_only
class_id class_name raw_output parsed_answer parser_status strict_json_valid failure
failure_stage interface_error image_content_status processor_content_status
processor_payload_status route_metadata_valid generation_trace processor_trace
ordered_image_identity image_refs_in_fixture_order elapsed_seconds peak_vram_bytes
model_generate_invocations_for_attempt successful_model_generate_returns_for_attempt
model_generate_errors_for_attempt runner_attempted runner_returned runner_call_attempted
runner_output_returned terminal_utc'''.split()
FIXTURE_FIELDS = '''fixture_id example_id target_id target_source_cluster_id target_patch_id
dataset requested_sensor non_requested_sensor requested_image_position requested_position
condition_id condition_family bigearthnet_condition_id condition_relation row_family
gold_answer expected_answer target_label opposite_gold_answer distractor_label distractor_answer
class_id class_name question_text question_norm system_prompt user_prompt answer_set
image_refs image_order is_replay replay_of source_fixture_id position_role
scientific_sample_unit selection_seed'''.split()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def assert_public(value):
    if PRIVATE.search(json.dumps(value, ensure_ascii=False)):
        raise ValueError('Private machine identifier/path detected; inspect locally, do not publish')


def metadata(value):
    """Redact operational metadata only; NEVER apply to raw model responses."""
    if isinstance(value, dict):
        return {k: metadata(v) for k, v in value.items()
                if not any(s in k.lower() for s in ('password', 'credential', 'private_key', 'access_token'))}
    if isinstance(value, list):
        return [metadata(v) for v in value]
    if isinstance(value, str) and PRIVATE.search(value):
        return '[omitted: machine-specific provenance; original SHA256 retained]'
    return value


def prediction(row):
    out = {k: row[k] for k in PREDICTION_FIELDS if k in row}
    for k in ('generation_trace', 'processor_trace', 'ordered_image_identity',
              'image_refs_in_fixture_order'):
        if k in out:
            out[k] = metadata(out[k])
    # Raw responses, parsed answers, statuses and failure messages are not rewritten.
    assert_public(out)
    return out


def fixture(row):
    out = {k: row[k] for k in FIXTURE_FIELDS if k in row}
    assert_public(out)
    return out


def check_images(expected, observed, visual_keys=None):
    if len(expected) != len(observed):
        raise ValueError('Image count differs')
    for a, b in zip(expected, observed):
        ah = a.get('sha256', a.get('expected_sha256'))
        bh = b.get('verified_sha256', b.get('sha256'))
        if not ah or ah != bh:
            raise ValueError('Image identity/order differs')
    if not expected and visual_keys:
        raise ValueError('Text-only record carries visual tensors')
