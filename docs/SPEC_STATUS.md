# Specification status

`NodeLM_TypeScript_Node_Distillation_Plan.md` is present and tracked as the project ground
truth. Plan activation pins the primary teacher, all three student candidates, their immutable
Hugging Face revisions, licenses, architectures, loader classes, parameter counts, and native
context limits where verified. These are metadata claims only.

## Settled by the plan

- primary teacher: `deepseek-ai/DeepSeek-V4-Flash-0731`;
- candidate set: `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.5-35B-A3B`, and
  `Qwen/Qwen3-Coder-Next`;
- source datasets: Open-SWE-Traces, SWE-rebench V2, and SWE-rebench V2 PRs;
- decision sequence: audit and decontaminate data, build one common harness, run the candidate
  bake-off, then select and train a student;
- no scaling of teacher generation or full fine-tuning before the pilot evidence gates pass.

## Real-source core snapshot audit complete

The offline complete-snapshot input and lineage contracts are implemented and exercised against
synthetic JSONL/Parquet fixtures. A transfer receipt binds the pinned source and raw registry to
the sorted local data-file inventory; audit parses a matching private staged view and explicitly
labels its aggregate input identity in a `nodelm.dataset-audit/v2` report. Lineage binds the
receipt, logical rows, report, and complete rejection ledger, and each immutable publication
refuses dependency drift.

All three pinned real snapshots have now been transferred to external persistent storage. Their
receipt-bound complete-snapshot audits and lineage manifests report `PASS` across 726,203 rows
and 49,734,682,463 supported bytes, with 33,937 license-gate rejections retained in complete
ledgers. The compact digest-bound evidence index is
`artifacts/reports/FULL_DATASET_AUDIT.md`; the raw snapshots and evidence remain outside Git.
This completes the core snapshot-audit prerequisite, not full normalization or decontamination.

## Real-source projection and normalization canary complete

At commit `7366ec06e8c2bb098afc02382e38b5b57f6e9b5d`, the complete pinned SWE-rebench V2
snapshot produced a receipt-replayed, gold-free task projection with 26,056 admitted and 6,023
license rejections. A 1,000-row OpenHands/MiniMax leaf canary then replayed both raw snapshot
chains and reported `PASS`: 783 normalized rows, 217 truthful unknown-resolution rejections, and
no duplicate or conflicting rollout identities. An earlier ordered Qwen36 slice reported
truthful `FAIL` because all 1,000 source resolution labels were unknown. Both outcomes and exact
hashes are recorded in `docs/EXPERIMENTS.md`; raw evidence remains on persistent external storage.

This establishes real canary behavior only. Its normalized manifest deliberately records the
gold-exposure audit as `NOT RUN`, so it is not permission to train.

## Real-source full normalization has terminal evidence

At commit `18c0ada5f396191d247cfe57640b6f2bb9fade86`, all seven eligible Open-SWE
SWE-rebench V2 leaves were run to terminal evidence. The four MiniMax/Qwen3.5 leaves whose
source rows contain boolean resolution evidence report `PASS`. The three Qwen3.6 leaves report
truthful `FAIL` only because their source resolution values are unknown; NodeLM correctly
refuses to coerce unknown into unresolved.

## Real resolution recovery derived; admission remains blocked

At commit `74c9b505eb1a608431ae3a18a3fca5d084f2ae3b`, the persistent CPU runner
validated a complete real recovery derivation. It found 3,960 exact-transfer candidate rows
(1,804 unique task-and-patch keys) and 49,572 unique evaluator requests covering 51,864 target
rows, with zero label conflicts. The immutable recovery manifest truthfully remains admission
`BLOCKED` by `harness_canary_pending`; derivation alone cannot authorize recovered labels.

The next gate is prepared in code but not yet executed: a small deterministic canary selects
both transfer controls and previously unknown evaluation requests, joins private test material
without gold solution patches, pins the upstream evaluator implementation and every selected
container by digest, then runs baseline/candidate test suites sequentially in offline, bounded,
rootless Podman sandboxes or restricted-host seccomp/chroot OCI rootfs clones. A `PASS` canary must
reproduce every failing baseline, preserve every
regression test, obtain an outcome for every sampled request, and agree with every transferred
control label.

## Still open by design

- the winning student, precision, inference backend, and training framework;
- candidate load/generate compatibility and the same-harness 50–100-task bake-off;
- the frozen public/private evaluation manifests and measured near-duplicate threshold;
- unique issue/PR counts, harness/model distributions, tokenizer-based trajectory lengths,
  exact/near patch duplication, and public-evaluation overlap;
- resolution harness canary, full queue disposition policy, contamination-safe split freezing,
  and pilot construction;
- the Tier A–D quality policy and actual 10k pilot artifact;
- model memory profiles and training topology. A paid host has been used for data work, but no
  strict model lifecycle or training measurement has run.

Model metadata, offline synthetic contract verification, receipt-bound transfer/core audit and
lineage for all three real sources, the complete V2 safe projection, and one real normalization
canary are `PASS`. All seven eligible full-partition leaves now have terminal evidence: four are
`PASS`, while three Qwen3.6 leaves are truthful `FAIL` because source resolution is unknown.
Real resolution recovery is `PASS` for derivation and still `BLOCKED` for admission pending the
prepared canary. Gold-exposure auditing, decontamination, split and pilot construction, model
execution, bake-off, student selection, and training remain `NOT RUN`. Recovery, canary, or
re-execution follows `docs/DATA_DOWNLOAD_RUNBOOK.md`.
