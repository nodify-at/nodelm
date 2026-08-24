# Codex development guide

This file is the durable handoff for future Codex work on NodeLM. Read it together with
`README.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, and the root NodeLM plan when that
plan is available.

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
  source hashes. Never promote a generic repository-test exit to `PASS`; without an
  integrity-attested oracle it remains `UNVERIFIED`.
- Never commit credentials, raw private keys, generated datasets, model weights, checkpoints,
  or measured artifacts that contain sensitive source data.
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
- Domain schemas belong in `src/nodelm/models.py`; I/O adapters must not weaken them.
- External commands go through the harness executor and its policy—never raw interpolated
  shell strings.
- Generated artifacts are written atomically below `artifacts/` with deterministic names,
  content hashes, and an explicit status. Prefer streaming immutable writers for
  dataset-sized outputs; never reintroduce unbounded report collections.
- Shell scripts are small wrappers with `set -Eeuo pipefail`; orchestration belongs in
  typed Python.
- Tests must be offline and deterministic by default. Network, training, and GPU checks use
  explicit pytest markers.
