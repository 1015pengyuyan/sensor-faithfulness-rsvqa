# Release privacy audit

## Scope

The candidate repository was scanned recursively after separating it from the
full research workspace.

## Results

| Check | Result |
|---|---|
| Raw M4-SAR images or archives | 0 |
| Raw BigEarthNet images or archives | 0 |
| Model weights/checkpoints | 0 |
| Official test assets | 0 |
| Credential assignments or embedded secrets | 0 |
| Personal usernames | 0 |
| Private server hostnames | 0 |
| Personal absolute paths | 0 |
| Prohibited binary/archive extensions | 0 |
| Files at least 100 MiB | 0 |

The only path-pattern matches from the broad Windows-path scan were escaped
LaTeX commands in the Table 3 generator; manual inspection confirmed they are
not file paths. Environment locations are represented by `${PROJECT_ROOT}`,
`${DATA_ROOT}`, and `${MODEL_ROOT}` or command-line arguments.

The two public fixture files independently report zero prohibited test-path
references. Generated raw predictions are directed to ignored `outputs/raw/`.

## Decision

`PASS_RELEASE_PRIVACY_SCAN`
