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

## Real resolution recovery and canary complete; admission remains blocked

At commit `74c9b505eb1a608431ae3a18a3fca5d084f2ae3b`, the persistent CPU runner
validated a complete real recovery derivation. It found 3,960 exact-transfer candidate rows
(1,804 unique task-and-patch keys) and 49,572 unique evaluator requests covering 51,864 target
rows, with zero label conflicts. The immutable recovery manifest truthfully remains admission
`BLOCKED`; derivation alone cannot authorize recovered labels. The terminal commit-bound 12-case
canary subsequently ran at `3c9690abc7f676762f613b84897ca5cf0156cc4a`. It completed 3 `PASS` /
9 `FAIL`, so execution is `FAIL` and recovery admission remains `BLOCKED`. D-021 quarantines every
recovered Qwen3.6 label for V1 and advances only the four existing labeled `PASS` leaves (160,731
rows). No failed case was replaced and no oracle rule was weakened.

## Real normalization cohort complete

At commit `24f4eb75f16f6782fdfa85762d3a27cd7fdbef10`, the four D-021 `PASS` leaves were
bound by a copy-free cohort. It reports `PASS` for complete selected-member binding: 4 members,
160,731 samples and globally unique sample IDs, and 51,063,015,261 ordered bytes. The cohort
manifest SHA-256 is `10734b8e20d127bfe69df8c5ffd3c8540038cfa95ad2d2c230ea92fa7e8d2621`; the
ordered-population SHA-256 is
`e91207cdca52c6fd08d0fd672c482fb7072117856f380b0b951bbe403fa85269`. This status does not
authorize gold safety, a split, a pilot, or training. After review hardening, commit
`af476d1e85da133b623456d2a34f0ef12a25b857` replayed the same four real members and reproduced
both identities exactly.

## Structural gold scans complete; oracle isolation blocked

At commit `24f4eb75f16f6782fdfa85762d3a27cd7fdbef10`, all four selected leaves were scanned
independently. Every structural component is `PASS`, every expected count equals its audited
count, and all four findings ledgers are empty. No oracle-isolation attestation was supplied, so
all four oracle components and overall gold-exposure audits remain `BLOCKED`. This is the expected
fail-closed boundary, not a failed structural scan or gold authorization.

## Still open by design

- the winning student, precision, inference backend, and training framework;
- candidate load/generate compatibility and the same-harness 50–100-task bake-off;
- the frozen public/private evaluation manifests and measured near-duplicate threshold;
- unique issue/PR counts, harness/model distributions, tokenizer-based trajectory lengths,
  exact/near patch duplication, and public-evaluation overlap;
- any future versioned recovery experiment, contamination-safe split freezing, and pilot
  construction;
- reviewed oracle-isolation attestations and overall gold-exposure authorization for the four
  selected leaves;
- the Tier A–D quality policy and actual 10k pilot artifact;
- model memory profiles and training topology. A paid host has been used for data work, but no
  strict model lifecycle or training measurement has run.

Model metadata, offline synthetic contract verification, receipt-bound transfer/core audit and
lineage for all three real sources, the complete V2 safe projection, and one real normalization
canary are `PASS`. All seven eligible full-partition leaves now have terminal evidence: four are
`PASS`, while three Qwen3.6 leaves are truthful `FAIL` because source resolution is unknown.
Real resolution recovery is `PASS` for derivation and `BLOCKED` for admission after its terminal
canary. The selected normalization cohort is `PASS`. Four structural gold scans are `PASS` with
zero findings; oracle isolation and the overall gold-exposure audits are `BLOCKED`. Decontamination,
split and pilot construction, model execution, bake-off, student selection, and training remain
`NOT RUN`. Recovery, canary, or
re-execution follows `docs/DATA_DOWNLOAD_RUNBOOK.md`.
