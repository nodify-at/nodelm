# Experiments

This ledger records executed work only. Planned checks belong in configuration, not here.

## 2026-08-24 — local bootstrap

- Dataset registry primary-source audit: `PASS` for three dataset IDs, revisions, dataset-card
  licenses, and live row totals.
- Core offline tests: `PASS` — 182 tests with 82.90% branch-aware coverage; Ruff and strict
  mypy also passed.
- Checked-in Node harness fixture: `PASS` — Node v24.10.0 ran `node --test` and passed one
  repository test with bounded structured command evidence.
- Python package build: `PASS` — both the source distribution and universal Python wheel built.
- Real model load/training: `NOT RUN` — no verified student model was selected.
- Remote GPU verification: `NOT RUN` — no host was requested.
- Student bake-off: `NOT RUN` — candidate registry is `UNVERIFIED`.

New experiment entries must include immutable input revisions, config/lock digests, commands,
versions, seed, artifact hashes, wall-clock time, and `PASS`, `FAIL`, `NOT RUN`, `BLOCKED`, or
`UNVERIFIED` per check.
