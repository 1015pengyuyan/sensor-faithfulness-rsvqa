# V3 release scope and verification

Release scope: eleven-model code, exact identity/configuration metadata and a
separate normalized-evidence attachment. Existing v1/v2 code, tags and assets
remain historical records. The published v2 release already has five assets; the
old draft wording in the previous root README was stale.

## Local verification

- Immutable source locks cover the current 195,618 scientific records and eight
  companion scientific/identity payloads.
- CPU replay reconstructs 6,888 main points and eligible denominators, checks 33
  cohorts and reparses 113,052 available raw responses. Both closed models retain
  18,894 planned/available calls and 43/52 parsed replay matches.
- Unit tests exercise contamination, score/parser mismatch, undefined ratios,
  missing evidence, path traversal and modified-byte failure behavior.
- No new inference, random draws, image/weight uploads or scientific outputs in
  Git. No current official figure appearance is changed by this update.

## Evidence distribution

The evidence attachment is distributed outside Git at
[v3.0.0-eleven-model](https://github.com/1015pengyuyan/sensor-faithfulness-rsvqa/releases/tag/v3.0.0-eleven-model).
The [attachment manifest](../reproduction/v3/RELEASE_ATTACHMENTS.json) pins its
4,699,373 bytes and SHA-256. Release verification checks an anonymous download
against that identity; local CPU replay checks the extracted scientific payloads
against a separate source lock. No final-author manuscript PDF is supplied.

The v3 main command verifies archived confidence-limit/supplement bytes and
reconstructs main points; it does not regenerate confidence intervals or the
separate sealed review supplement. This distinction remains part of the public
reproduction boundary after publication.
