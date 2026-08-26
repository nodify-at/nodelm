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

## 2026-08-25 — real full-partition Open-SWE normalization

- Code and command: commit `18c0ada5f396191d247cfe57640b6f2bb9fade86` ran the offline,
  commit-bound `scripts/run_full_normalization.sh` runner against all seven eligible SWE-rebench
  V2 leaves on persistent `/workspace` storage.
- Immutable inputs: Open-SWE-Traces
  `ed95cef24df8d8bd79b4ceb0192cb420fde06521`, SWE-rebench V2
  `475dd5e8703bb5fb22dd3c60b5d038b019eba1e0`, registry SHA-256
  `f92315a70a0c75ec909d83f4cb639b3a320f62526069f11ca87f0fe1d891637f`, partition-contract
  SHA-256 `aec2ae095a926dda09a5fe3eefede7a59fbd494b24fffd503fff4cb366b389b5`, and
  lock SHA-256 `88f316fbfffc905a14d9528706171e1e30428e72fde432c8775edebc67859160`.
- Runtime: the pre-provisioned offline `/opt/nodelm-venv` data environment documented above,
  using Python 3.11.13 and PyArrow 25.0.1. No stochastic seed applied because the runner performs
  deterministic materialization, replay, validation, and canonical publication only.
- Result: all seven leaves reached validated terminal evidence. Across the four labeled
  MiniMax/Qwen3.5 leaves, 207,489 inputs produced 160,731 accepted rows and 46,758 truthful
  `unknown_resolution` rejections. Across the three Qwen3.6 leaves, 179,533 inputs produced zero
  accepted rows and 179,533 truthful `unknown_resolution` rejections. The labeled leaves are
  `PASS`; the Qwen3.6 leaves are truthful `FAIL`, not runner failures.
- Normalization-manifest SHA-256 values:
  - `openhands/minimax_m25/swe-rebench-v2` (`PASS`):
    `68f21f192c1af397837610ef6e4033fb04e9e2a38043f461a42d70ab6b47082e`;
  - `openhands/qwen35_122b/swe-rebench-v2` (`PASS`):
    `339f4a2923eaa92b07155fb35604ce61c2358f554de09146226124ab01ca0996`;
  - `sweagent/minimax_m25/swe-rebench-v2` (`PASS`):
    `2ed86030086ededc8c69c8aa8801f49c4034a7cbee2878f7311cf60512cc1c2e`;
  - `sweagent/qwen35_122b/swe-rebench-v2` (`PASS`):
    `725ab61444546026c10d4e4d6745c324f0430063151650c905cfbc6b7d80b372`;
  - `minisweagent/qwen36_27b/swe-rebench-v2` (truthful `FAIL`):
    `17a262bf45d7406d47d0904ad1a26948171650580db5e77a8f1eb57172859c54`;
  - `openhands/qwen36_27b/swe-rebench-v2` (truthful `FAIL`):
    `1bd8a306a6ecf501db01f29aa7eb4bbffe893d2b1648576756afe958d16f64f7`;
  - `sweagent/qwen36_27b/swe-rebench-v2` (truthful `FAIL`):
    `054be3587000986ef1ff92ccd95726a26f820b71f8a4a3534962b0cf51f6dabd`.
- Wall clock: `2026-08-25T14:29:08Z` through `2026-08-25T18:12:53Z`, or 3 hours
  43 minutes 45 seconds. The complete evidence remains outside Git under
  `/workspace/nodelm/derived/full-normalization-18c0ada5f396191d247cfe57640b6f2bb9fade86`.
- GPU use, model loading, model execution, evaluation, and training: `NOT RUN`. Resolution-recovery
  derivation was still `NOT RUN` at this boundary; these terminal manifests do not authorize
  recovered labels or training admission.

## 2026-08-25 — real Qwen3.6 resolution-recovery derivation

- Code and command: commit `74c9b505eb1a608431ae3a18a3fca5d084f2ae3b` ran the offline,
  commit-bound `scripts/run_resolution_recovery.sh` runner against the completed persistent
  Open-SWE snapshot and its sealed receipt.
- Immutable inputs: Open-SWE-Traces revision
  `ed95cef24df8d8bd79b4ceb0192cb420fde06521` and SWE-rebench V2 revision
  `475dd5e8703bb5fb22dd3c60b5d038b019eba1e0`; the run revalidated the checked-in registry,
  partition contract, receipt, exact labeled/target leaf files, and code commit before terminal
  publication.
- Result: derivation `PASS`, admission `BLOCKED` by `harness_canary_pending`. Across 179,533
  Qwen3.6 target rows, 123,709 were ineligible, zero were already known, 3,960 had exact transfer
  evidence, and 51,864 required evaluation. The transfer sidecar contains 1,804 unique exact keys
  (2,545 resolved rows and 1,415 unresolved rows); the deduplicated evaluator queue contains
  49,572 unique requests. Conflict count: zero.
- Artifacts: exact-transfer candidates are 4,987,642 bytes with SHA-256
  `157b8595dc48bd1b131fa76ae911c95332129474d5a281a0b6ec142789de2afa`; the evaluator queue is
  3,550,730,380 bytes with SHA-256
  `cd2ab196f49ab9d2e3c1a1066085a0331258868ff43258d7604f9e7746d0b569`. Private artifacts and the
  terminal manifest remain outside Git under
  `/workspace/nodelm/derived/resolution-recovery-74c9b505eb1a608431ae3a18a3fca5d084f2ae3b`.
- Wall clock: `2026-08-25T21:23:50Z` through `2026-08-25T21:34:21Z`, or 10 minutes 31 seconds.
  No stochastic seed applied; this was deterministic projection, indexing, and publication.
- GPU use, model loading, training, and repository evaluation: `NOT RUN`. No recovered label was
  admitted; the later terminal canary remained `BLOCKED` as recorded below.

## 2026-08-26 — restricted Runpod sandbox compatibility probe

- Environment: the x86_64 Runpod CPU pod exposed 256 CPUs and 2 TiB RAM, but its outer seccomp
  policy rejected both rootless and rootful nested namespaces with `Operation not permitted`.
  Rootless Podman therefore could not satisfy the canary contract on this host.
- Fallback verification: `skopeo` pinned and converted one selected public image,
  `docker.io/swerebenchv2/eslint-doctrine@sha256:fd70e9c17a3b65c5588fd4178ece3f7a642705924fa08fc4728c047aaaab5c31`.
  Its converted local OCI manifest was 2,135 bytes with SHA-256
  `e864b49a35c8d9c6702e7749bd9f49d22cffffe2f77d6215ea3810c08995524b`.
- Real control result: case
  `34b691a7b91bb4303cdc82cc589a697962521738a859af3b236f284bb100a8c5` reproduced the
  baseline failure, observed all 240 expected tests in both attempts, passed the patched candidate,
  and agreed with its transferred `resolved=true` label. A direct launcher probe ran as UID 61000
  and an outbound IPv4 socket attempt returned `Operation not permitted`.
- Scope: this was a manual compatibility probe, not retained terminal evidence and not an admission
  result. It was superseded by the terminal commit-bound execution below.

## 2026-08-26 — terminal real-repository resolution canary

- Code and runtime: commit `3c9690abc7f676762f613b84897ca5cf0156cc4a` ran the fixed 12-case
  workset through the restricted-host `seccomp-chroot` backend. The hardened rootfs supplied an
  isolated root-owned read-only synthetic `/proc/self/stat`, restoring Node/libuv memory evidence
  without exposing host procfs. Every attempt remained offline, limited to two CPUs, 4 GiB memory,
  and a digest-pinned preloaded image.
- Immutable inputs: recovery manifest SHA-256
  `94867425f1cb75578ad7d8a8ad86e03212d156f3eed40534c61f3e3dbf5b650f`; workset SHA-256
  `40724da67b4c4502864f4f09b978efc787d5051315bd3a396a48a74643169c2f`; workset-manifest
  SHA-256 `ac327cf624ea8d14168fe2e757d7cb11beee8bebbb188bf71732f7e0624d3c07`; and 12-image lock
  SHA-256 `19d8a66e012c498d708ec6bdefab74cc943ee52502d33c1b01c9e47bee18d0b1`.
- Result: all 12 cases reached immutable terminal evidence. Execution is `FAIL` and admission is
  `BLOCKED`: 3 cases passed and 9 failed. Failure accounting is 5
  `incomplete_expected_test_evidence`, 3 `candidate_exit_status_contradicts_test_evidence`, and
  1 `failing_baseline_not_reproduced`. Three of six transfer controls agreed; none of the six
  evaluation requests produced an admissible resolved/unresolved outcome.
- Artifacts: public results are 21,657 bytes with SHA-256
  `5b40ca0c1f079310d308ee77ba185b48d7930430719bb1ee0d206f20264132c6`. The terminal execution
  manifest is 1,721 bytes with SHA-256
  `d289b4552b55937e2ad981a0844119e6da951e20b23236a7c8cee34f3dc641e7`. Private per-case evidence
  and public artifacts remain on persistent storage under
  `/workspace/nodelm/derived/resolution-canary-3c9690abc7f676762f613b84897ca5cf0156cc4a`.
- Wall clock: `2026-08-26T20:05:06Z` through `20:28:53Z`, approximately 23 minutes 47 seconds.
  Seed: not applicable; selection and evaluation were deterministic. No GPU, model loading,
  training, or paid inference ran.
- Decision: D-021 keeps all recovered Qwen3.6 labels quarantined and advances V1 only with the
  four already-`PASS` labeled normalization leaves. The canary is complete; its blocked verdict
  is not retried by weakening the oracle or replacing failed cases.

## 2026-08-26 — selected normalization cohort and structural gold scans

- Code and runtime: the initial real cohort and four structural scans ran at commit
  `24f4eb75f16f6782fdfa85762d3a27cd7fdbef10`. After review hardening, Python 3.11.13 replayed the
  cohort at commit `af476d1e85da133b623456d2a34f0ef12a25b857` through the installed module at
  `/opt/nodelm-venv`. These were deterministic CPU/data-integrity operations; no GPU, model load,
  training, or paid inference ran.
- Immutable selected members, expressed as normalization-manifest SHA-256 / normalized-artifact
  SHA-256 / accepted count:
  - `openhands/minimax_m25/swe-rebench-v2`:
    `68f21f192c1af397837610ef6e4033fb04e9e2a38043f461a42d70ab6b47082e` /
    `38d55c3679643877d9eda6262a430d3e8a77c21f8809599a8a4ee70658de646a` / 38,852;
  - `openhands/qwen35_122b/swe-rebench-v2`:
    `339f4a2923eaa92b07155fb35604ce61c2358f554de09146226124ab01ca0996` /
    `2a39222383f517dd465910a5723d72920be6756ebb987ed36ad3184b1435cd11` / 42,804;
  - `sweagent/minimax_m25/swe-rebench-v2`:
    `2ed86030086ededc8c69c8aa8801f49c4034a7cbee2878f7311cf60512cc1c2e` /
    `7084f03e5e2415e08b24a7160e68729ded045eecc1ff92c701cccfbd26da5dc5` / 44,105;
  - `sweagent/qwen35_122b/swe-rebench-v2`:
    `725ab61444546026c10d4e4d6745c324f0430063151650c905cfbc6b7d80b372` /
    `dbaade2d9c412158d651d9142da8dc4387e2795969ec9d18ae758a67d9f6eddd` / 34,970.
- Commands: `python -m nodelm datasets build-normalization-cohort` received those four exact
  manifests via repeated `--member-manifest` arguments. The audit runner then invoked
  `python -m nodelm datasets audit-gold-exposure` independently for each normalized member with
  its bound manifest and distinct audit/findings outputs; no `--oracle-isolation-attestation` was
  supplied. The hardened replay repeated the same cohort command into a new commit-addressed
  output and required the prior manifest/population identities before publishing `COMPLETE`.
- Cohort result: `PASS` for 4 members, 160,731 globally unique samples, and 51,063,015,261 ordered
  bytes. Manifest SHA-256:
  `10734b8e20d127bfe69df8c5ffd3c8540038cfa95ad2d2c230ea92fa7e8d2621`; ordered-population
  SHA-256: `e91207cdca52c6fd08d0fd672c482fb7072117856f380b0b951bbe403fa85269`.
  The initial build took 28 minutes 38 seconds. The hardened replay ran from `22:00:56Z` through
  `22:27:17Z`, or 26 minutes 21 seconds, and reproduced both identities exactly.
- Hardened replay evidence: output manifest bytes 3,457; run-state SHA-256
  `c0e44ac4762aa0a00a8bc6ad52e6ea310b93ad0f03ffd117de5864c15d7f9e1e`; events-log SHA-256
  `73919122b0c36860df65cace33cf3b564ce732cd9e223cdc75b5b9dd1bedd3e1`. Evidence is retained at
  `/workspace/nodelm/derived/normalization-cohort-af476d1e85da133b623456d2a34f0ef12a25b857`.
- Structural audit result: all four scans are `PASS`, expected and audited counts agree, and every
  findings ledger is the empty-file SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. Audit SHA-256 values in
  member order are `71efaa86488ca3c566be0cac75141a1d5371798834e21c75a166e6f301763683`,
  `6dfe439055d7ead4b5409fae33faaea9f47fd4bb3e64a38aa133e4da9e688ce5`,
  `8ec4acb308af9b3ebd6b816d937c7995ddf0af462fda0c4cb85e7a7e1f256188`, and
  `d353e25f6a103271a720c6ab153d82d68d8a16e99d1cf74afa1dd0b0708210a3`.
  The four-leaf runner took 35 minutes 24 seconds (`21:17:30Z` through `21:52:54Z`); its run-state
  SHA-256 is `fb76b84cbf80d196112273bf613fbe938d864fff6f00b98fe67512e6fa744d25` and events-log SHA-256 is
  `2a229ab668b571058b4714606b58d62257436a0c306ff3b4c11ca79f7563af5b`.
- Oracle isolation and every overall gold-exposure audit remain `BLOCKED` because no attestation
  was supplied. This is the intended fail-closed result, not a structural failure. Decontamination,
  split freezing, pilot construction, model execution, and training remain `NOT RUN`. Seed: not
  applicable.

New experiment entries must include immutable input revisions, config/lock digests, commands,
versions, seed, artifact hashes, wall-clock time, and `PASS`, `FAIL`, `NOT RUN`, `BLOCKED`, or
`UNVERIFIED` per check.
