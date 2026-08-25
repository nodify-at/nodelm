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

## 2026-08-24 — real full-snapshot transfer and receipt-bound core audit

- Status: `PASS` for all three pinned sources at code revision
  `bfcc3b6a46e76043a50acced55d45bb16bf3cbb6` using Python 3.11.13.
- Immutable inputs: Open-SWE-Traces
  `ed95cef24df8d8bd79b4ceb0192cb420fde06521`, SWE-rebench V2
  `475dd5e8703bb5fb22dd3c60b5d038b019eba1e0`, and SWE-rebench V2 PRs
  `fbf0ecf50f268d5344149e2f0097db6bede83737`.
- Input digests: `uv.lock` SHA-256
  `88f316fbfffc905a14d9528706171e1e30428e72fde432c8775edebc67859160`; dataset registry
  SHA-256 `f92315a70a0c75ec909d83f4cb639b3a320f62526069f11ca87f0fe1d891637f`.
- Commands: guarded `nodelm datasets download --confirm-large-download` followed by offline
  `nodelm datasets audit-snapshot`, one source at a time on external persistent storage.
- Result: 235 supported files, 49,734,682,463 bytes, and 726,203 rows were bound to immutable
  receipts. Every row count matched its declaration; all audit and lineage statuses were
  `PASS`; 33,937 license-gate rejections were retained in complete ledgers.
- Wall clock: the corrected SWE-rebench V2 PRs plus Open-SWE runner took 46 minutes 33 seconds
  (`2026-08-24T22:26:51Z` through `23:13:24Z`). The separate SWE-rebench V2 smoke run did not
  persist wrapper start/end timestamps. An earlier runner invocation failed before transfer on
  the invalid `--receipt` option; the corrected `--receipt-output` invocation completed.
- Seed: not applicable; transfer, hashing, streaming audit, and canonical publication are
  deterministic and use no stochastic model operation.
- Artifact hashes and result distributions are recorded in
  `artifacts/reports/FULL_DATASET_AUDIT.md`. The complete raw evidence remains under
  `/workspace/nodelm/{receipts,audits,logs}` outside Git. Its recorded hashes were independently
  rechecked on 2026-08-25.
- Normalization, tokenizer measurements, decontamination, split freezing, pilot construction,
  model execution, bake-off, training, and evaluation remain `NOT RUN`.

## 2026-08-25 — real task projection and receipt-replayed normalization canaries

- Code and host: commit `7366ec06e8c2bb098afc02382e38b5b57f6e9b5d` on the Runpod
  persistent `/workspace` volume with an NVIDIA RTX PRO 6000 Blackwell Server Edition. The GPU
  remained at 0 MiB/0% because these are CPU/data-integrity operations.
- Immutable inputs: Open-SWE-Traces
  `ed95cef24df8d8bd79b4ceb0192cb420fde06521`, SWE-rebench V2
  `475dd5e8703bb5fb22dd3c60b5d038b019eba1e0`, registry SHA-256
  `f92315a70a0c75ec909d83f4cb639b3a320f62526069f11ca87f0fe1d891637f`, partition-contract
  SHA-256 `aec2ae095a926dda09a5fe3eefede7a59fbd494b24fffd503fff4cb366b389b5`, and
  lock SHA-256 `88f316fbfffc905a14d9528706171e1e30428e72fde432c8775edebc67859160`.
- Runtime: Python 3.11.13, Node 24.10.0, datasets 5.0.1, PyArrow 25.0.1, pytest 9.1.1,
  Ruff 0.16.4, and mypy 1.20.2. The remote `make verify` gate passed 323 tests with 82.84%
  branch-aware coverage plus doctor, formatting, lint, and strict typing.
- Commands ran with `UV_PROJECT_ENVIRONMENT=/opt/nodelm-venv`, the locked project at
  `/workspace/nodelm/repo`, and the exact sealed receipt/contract paths shown below:

```bash
uv run --frozen --directory /workspace/nodelm/repo nodelm datasets project-task-provenance \
  --source swe-rebench-v2 --snapshot /workspace/nodelm/snapshots/swe-rebench-v2 \
  --transfer-receipt /workspace/nodelm/receipts/swe-rebench-v2.transfer.json \
  --output /workspace/nodelm/derived/normalization-canary-20260825-7366ec0/swe-rebench-v2.safe.jsonl \
  --config /workspace/nodelm/repo/configs/datasets/registry.yaml

uv run --frozen --directory /workspace/nodelm/repo nodelm datasets materialize \
  --source open-swe-traces --snapshot /workspace/nodelm/snapshots/open-swe-traces \
  --partition-contract /workspace/nodelm/repo/configs/datasets/open-swe-trace-partitions.yaml \
  --transfer-receipt /workspace/nodelm/receipts/open-swe-traces.transfer.json \
  --partition openhands/minimax_m25/swe-rebench-v2 --max-rows 1000 \
  --output /workspace/nodelm/derived/normalization-canary-20260825-7366ec0/openhands-minimax-v2.raw.jsonl \
  --config /workspace/nodelm/repo/configs/datasets/registry.yaml

uv run --frozen --directory /workspace/nodelm/repo nodelm datasets normalize \
  --source open-swe-traces --snapshot /workspace/nodelm/snapshots/open-swe-traces \
  --input /workspace/nodelm/derived/normalization-canary-20260825-7366ec0/openhands-minimax-v2.raw.jsonl \
  --materialization-manifest /workspace/nodelm/derived/normalization-canary-20260825-7366ec0/openhands-minimax-v2.raw.manifest.json \
  --partition-contract /workspace/nodelm/repo/configs/datasets/open-swe-trace-partitions.yaml \
  --transfer-receipt /workspace/nodelm/receipts/open-swe-traces.transfer.json \
  --task-provenance /workspace/nodelm/derived/normalization-canary-20260825-7366ec0/swe-rebench-v2.safe.jsonl \
  --task-provenance-manifest /workspace/nodelm/derived/normalization-canary-20260825-7366ec0/swe-rebench-v2.safe.manifest.json \
  --task-transfer-receipt /workspace/nodelm/receipts/swe-rebench-v2.transfer.json \
  --task-snapshot /workspace/nodelm/snapshots/swe-rebench-v2 \
  --expect-harness openhands --expect-generating-model source-label:minimax_m25 \
  --output /workspace/nodelm/derived/normalization-canary-20260825-7366ec0/openhands-minimax-v2.normalized.jsonl \
  --config /workspace/nodelm/repo/configs/datasets/registry.yaml
```

- Complete V2 task projection: `PASS`; 26,056 admitted and 6,023 rejected (5,964 unknown
  repository licenses and 59 disallowed licenses). Safe JSONL SHA-256
  `1e70b4d99cee7eea5dd40c4c36a553a53de3304caa7120ec45c00b5a2b6fdffd`, rejection-ledger
  SHA-256 `473679bf93386cd6bdbea8019e7991104c355fd21b30886632669c2e099d7bf2`, and manifest
  SHA-256 `93f17e1f466fa0e014b29112c34d5f05830c17f39e296bcc89915f7b5567cfb5`.
- First ordered canary (`openhands/qwen36_27b/swe-rebench-v2`): truthful `FAIL`; all 1,000
  source rows had `resolved=-1`, so all were recorded as `unknown_resolution`. Both raw replays
  were `PASS`, with zero duplicate/conflicting rollout identities. Raw SHA-256
  `d4837accf085797562c8db980b602966f92b12da95fe2cb8b729ab8938cee8ee`, normalization
  manifest SHA-256 `3ba7422fbe340a009dc72dafc91dca6aada2009ed4365b344c2fe7c304a9192a`, and complete
  rejection-ledger SHA-256 `28c43b5530726e832a1bfd951d48b842e6d6adaed3c6db3e2c02773554e9d4f1`.
- Second canary (`openhands/minimax_m25/swe-rebench-v2`): `PASS`; 1,000 input rows, 783
  admitted, and 217 truthful `unknown_resolution` rejections. Materialization and task-provenance
  replay were both `PASS`; 1,000 rollout keys were unique with zero duplicates/conflicts.
  The normalized set contains 249 JavaScript/TypeScript rows, 87 of them resolved. Raw JSONL
  SHA-256 `92a3c8d0f6c1967be151b9fd0eb9adc95ecc6c888d6ea5b8b1a6f6a5c0d192bf`, normalized
  JSONL SHA-256 `ec337f26ad8bcba1a64e716c090685f4da2a046898d49b137f8a7cae03cd7b33`, rejection-ledger
  SHA-256 `506d85d3ff51f2f31a1d2a80cf33b56ca32b9041e8e62ab9ec6baae516c65a71`, and normalization
  manifest SHA-256 `9d8ff2ac67a382d78b01a1cbd8b7d4b8a7498e5ce9c04a40a505582e6594a680`.
- Terminal artifact timestamps span 2026-08-25 10:29:51–10:35:06 UTC. Successful client-observed
  stage durations were approximately 15 seconds for task projection, 29–31 seconds per
  materialization, and 42–45 seconds per normalization. An initial `/usr/bin/time` wrapper failed
  before execution because that binary is absent; the command was rerun unchanged without it.
- Seed: not applicable. No stochastic model, training, or GPU operation ran. Structural
  gold-exposure auditing, decontamination, split freezing, pilot construction, full-partition
  normalization, candidate execution, and training remain `NOT RUN`. All generated evidence is
  retained outside Git under
  `/workspace/nodelm/derived/normalization-canary-20260825-7366ec0`.

New experiment entries must include immutable input revisions, config/lock digests, commands,
versions, seed, artifact hashes, wall-clock time, and `PASS`, `FAIL`, `NOT RUN`, `BLOCKED`, or
`UNVERIFIED` per check.
