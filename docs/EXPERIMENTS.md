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
- Student bake-off: `NOT RUN` — the initial registry was unpopulated.

## 2026-08-24 — plan activation (preparation only)

- The NodeLM distillation plan was added as the tracked ground truth.
- Candidate and primary-teacher public metadata: `PASS` at immutable Hugging Face revisions;
  strict configuration validation distinguishes metadata from execution.
- Candidate execution, bake-off, student selection, teacher execution, model load, and training:
  `NOT RUN`.
- Full dataset snapshot transfer and bulk materialization: `NOT RUN`; deferred to a
  user-provisioned large-storage GPU instance.
- Only small model metadata records were queried. No dataset snapshot, model weight, checkpoint,
  or generated training artifact was downloaded.
- Full offline gate: `PASS` — local doctor, Ruff, strict mypy across 48 source files, and 217
  tests with 83.28% branch-aware coverage.

## 2026-08-24 — receipt-bound snapshot contract verification

- Status: `PASS` at code revision `3f4cb68c3dfa107d2dcc2f7248de34424c15e923`.
- Immutable dataset metadata inputs, without snapshot access: Open-SWE-Traces
  `ed95cef24df8d8bd79b4ceb0192cb420fde06521`, SWE-rebench V2
  `475dd5e8703bb5fb22dd3c60b5d038b019eba1e0`, and SWE-rebench V2 PRs
  `fbf0ecf50f268d5344149e2f0097db6bede83737`.
- Input digests: `uv.lock` SHA-256
  `88f316fbfffc905a14d9528706171e1e30428e72fde432c8775edebc67859160`; dataset registry
  SHA-256 `f92315a70a0c75ec909d83f4cb639b3a320f62526069f11ca87f0fe1d891637f`.
- Command: `/usr/bin/time -p make verify`.
- Versions: Python 3.11.7, pytest 9.1.1, Ruff 0.16.4, and mypy 1.20.2.
- Seed: not applicable; the gate used deterministic synthetic fixtures and no stochastic model
  or data operation.
- Result: local doctor, formatting, lint, and strict typing passed; 277 offline tests passed with
  83.64% branch-aware coverage. Wall-clock time was 4.92 seconds.
- Artifact hashes: not applicable; receipt, audit, ledger, and lineage evidence was generated
  only inside disposable pytest directories, and no real-source artifact was retained.
- Real snapshot transfer, real-source audit/lineage, model download, training, and GPU execution:
  `NOT RUN`. No network, dataset, model, or GPU operation was part of this verification.

New experiment entries must include immutable input revisions, config/lock digests, commands,
versions, seed, artifact hashes, wall-clock time, and `PASS`, `FAIL`, `NOT RUN`, `BLOCKED`, or
`UNVERIFIED` per check.
