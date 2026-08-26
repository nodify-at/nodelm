# Codex development guide

This file is the durable handoff for future Codex work on NodeLM. Read it together with
`README.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, and the tracked root
`NodeLM_TypeScript_Node_Distillation_Plan.md`.

## Non-negotiable rules

- Never fabricate dataset/model identifiers, revisions, licenses, row counts, benchmark
  scores, framework support, GPU requirements, API behavior, or CLI flags.
- Prefer primary sources: official repositories and documentation, then source/release
  notes, model or dataset cards, and original papers.
- Preserve immutable provenance from source snapshot through every derived sample.
- Keep repositories disjoint across training and private evaluation. Gold patches must
  never enter teacher or student solve context.
- Unknown and copyleft repository licenses are rejected from V1 training data but retained
  in an audit report.
- Treat network, environment, dependency-install, timeout, tool-protocol, test, and model
  failures as distinct outcomes.
- Never execute model-authored code through the trusted-local backend. Require a preloaded,
  digest-pinned rootless Podman image with a read-only host-workspace bind, no network, bounded
  resources, and verified forced cleanup.
- Accept model output only as a bounded, text-only Git patch. Reject protected paths when any
  path component is a symlink.
- Bind protected smoke evaluation to the exact expected `SolveContext` and approved before/after
  source hashes. Require the complete code-pinned regular-file fixture identity, evaluate only a
  private verified copy, and bind that tree digest into the terminal report. Never promote a
  generic repository-test exit to `PASS`; without an integrity-attested oracle it remains
  `UNVERIFIED`.
- Never commit credentials, raw private keys, generated datasets, model weights, checkpoints,
  or measured artifacts that contain sensitive source data.
- All three pinned dataset snapshots and their receipt-bound core audits and lineage are complete
  on external persistent storage; no re-download is currently required. Raw evidence is indexed
  by `artifacts/reports/FULL_DATASET_AUDIT.md`. One receipt-replayed real normalization canary is
  `PASS`. All seven eligible full-partition leaves have terminal evidence: four MiniMax/Qwen3.5
  leaves are `PASS`, while three Qwen3.6 leaves are truthful `FAIL` because their source resolution
  is unknown. Real resolution recovery is derived with zero conflicts, but its admission remains
  `BLOCKED` pending the prepared rootless-Podman canary. Gold auditing and decontamination remain
  `NOT RUN`. Require explicit current-session authorization for recovery or re-execution into new,
  empty destinations and for bulk materialization. Follow `docs/DATA_DOWNLOAD_RUNBOOK.md` and the
  exact evidence in `docs/EXPERIMENTS.md`.
- Treat `complete-snapshot` as an input-scope claim: it covers all supported JSONL/Parquet data
  files discovered at the supplied local path. It does not mean the plan's Phase 0 is complete,
  and a synthetic fixture `PASS` is never a real-source audit or lineage result.
- Require the guarded download's immutable transfer receipt before assigning a pinned source to
  local snapshot bytes. The receipt binds the source revision, registry identity, and supported
  snapshot inventory; filtered-transfer receipts cannot authorize a complete-snapshot audit.
- Audit only a private staged copy whose identity exactly matches the transfer receipt. Preserve
  legacy raw-file reports as `nodelm.dataset-audit/v1`; use `nodelm.dataset-audit/v2` only for
  complete aggregate-snapshot identity. Keep lineage bound to the receipt, registry, snapshot,
  logical rows, report, and complete rejection ledger; revalidate once at each immutable
  publication boundary and publish lineage last. Treat the private stage as same-account process
  isolation, not as a security sandbox against a hostile process running under the same OS user.
- Keep the receipt-bound dataset registry byte-for-byte stable until a deliberate re-attestation
  migration exists. Open-SWE materialization must use the separate digest-bound 11-leaf
  partition contract, a complete transfer receipt, and one partition per artifact. Derive
  harness, source-native model label, and task family from the contract; never relabel a batch
  from free text.
- Open-SWE normalization requires a v2 partition materialization, a complete license-safe task
  provenance projection, both code-authorized transfer-receipt seals, both raw snapshot roots,
  and a required matching task join. Replay materialization and task projection from private
  identity-verified staging before accepting either derived manifest. The safe
  task artifact may contain only repository, base commit, license, language, instance ID, and
  task-source identity. Poison the complete task identity if any duplicate row is unsafe or
  conflicting, while retaining its original rejection cause. Treat `resolved=-1`/null as
  `unknown_resolution`, not false. Scale-SWE and V2-PR task joins remain `BLOCKED` until their
  missing provenance is verified.
- Scope rollout uniqueness to the exact partition leaf and use a disk-backed index. Admit one
  copy of an exact duplicate, reject every row in a conflicting rollout group, and never collapse
  distinct rollout IDs. Every normalized sample includes its raw-row digest in lineage.
- Never build or train from a normalized JSONL alone. Pilot construction requires its replay-
  verified normalization manifest, a reviewed split digest authorized for the same bytes and row
  count, and a reviewed, code-authorized `PASS` gold-exposure audit with complete oracle-isolation
  coverage. Build the split from private identity-verified copies of every input. The one-step
  training lifecycle requires the exact reviewed pilot digest authorized for its samples, reads
  both through private staging, and rechecks every referenced input at report publication.
- Keep the project hardware-agnostic. Hardware-specific configuration belongs under
  `configs/infra/`, not in core dataset or harness logic.

## Change workflow

1. Read the affected contract, implementation, tests, and callers before editing.
2. Add or strengthen an externally meaningful test before behavior-bearing code where
   practical; observe the expected failure.
3. Make the smallest coherent change with typed Python and explicit error handling.
4. Run focused tests, then `make verify`.
5. Inspect `git diff --check`, the complete diff, security implications, error semantics,
   compatibility, and resource lifecycle.
6. Record material architectural choices in `docs/DECISIONS.md` and actual experiments in
   `docs/EXPERIMENTS.md`. Never promote a planned or unrun check to `PASS`.

## Status vocabulary

Use only `PASS`, `FAIL`, `NOT RUN`, `BLOCKED`, or `UNVERIFIED` in reports. Include the exact
command and observed versions for measured verification. A generated configuration is not
evidence that training, evaluation, or a benchmark succeeded.

## Repository conventions

- Python support: the locked Python 3.11 minor; strict typing; Ruff; mypy; pytest.
- Domain schemas are strict Pydantic models colocated with their owning domain; I/O adapters
  must not weaken them.
- External commands go through the harness executor and its policy—never raw interpolated
  shell strings.
- Generated artifacts are written atomically below `artifacts/` with deterministic names,
  content hashes, and an explicit status. Prefer streaming immutable writers for
  dataset-sized outputs; never reintroduce unbounded report collections.
- Shell scripts are small wrappers with `set -Eeuo pipefail`; orchestration belongs in
  typed Python.
- Tests must be offline and deterministic by default. Network, training, and GPU checks use
  explicit pytest markers.
