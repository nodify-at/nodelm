# Decisions

## D-001 — one typed Python application

Use a single `nodelm` package and CLI. Shell remains limited to bootstrap and remote glue. This
keeps schemas, status semantics, and deterministic artifacts consistent across phases.

## D-002 — Python 3.11 exact minor

The lock targets Python 3.11 because the current dataset/training stack publishes compatible
wheels there and the project can reproduce it across local and remote hosts. Upgrades require a
new lock and the full smoke lifecycle.

## D-003 — conservative license gate

Only MIT, Apache-2.0, BSD-2-Clause, and BSD-3-Clause enter V1. Unknown and copyleft rows are
audited, not discarded without trace.

## D-004 — immutable, content-connected evaluation split

Repository identity is the minimum split group. Declared mirrors plus exact or measured-near
task/reference-patch/generated-patch matches join one connected group. Exact or near overlap
with an explicitly selected public benchmark excludes the entire group from training and
private evaluation. The benchmark JSONL and strictly positive threshold are required inputs, so
the currently unfrozen benchmark input and threshold remain honest blockers rather than being
replaced by implicit defaults. Offline mirror detection remains incomplete and still requires
an additional audit before tuning.

## D-005 — trusted local and untrusted execution stay separate

Local subprocess controls protect reproducibility and reduce accidents. Untrusted repository
or model-authored code runs only in a preloaded, digest-pinned rootless Podman image. Its host
workspace bind is read-only, networking and implicit pulls are disabled, resources are bounded,
and a named container is force-removed in a `finally` path. Only bounded text patches reach the
apply step, and protected repository-test paths reject symlink components.

## D-006 — no default model without the source plan (superseded by D-012)

At bootstrap, the NodeLM plan was unavailable. Configurations therefore kept model fields null
and reported `UNVERIFIED`/`BLOCKED` rather than inventing replacements. D-012 records the
activation policy now that the plan is tracked.

## D-007 — disk-backed split and bounded audit summaries

Repository split artifacts are built through SQLite and streamed to immutable canonical JSON,
so registry-scale JSONL does not require several in-memory copies. Audit identity counts are
also disk-backed; percentile, duplicate-ID, and inline rejection examples are capped and
explicitly labeled, while reports point to a complete streamed rejection ledger. Split evidence
records task-table and benchmark digests plus typed exact, near, overlap, connected-group, and
exclusion counts without serializing raw task or patch text.

## D-008 — truthful trusted-local harness capabilities

The local backend consumes a validated, hashed config. It declares network access enabled
because it cannot enforce network isolation, refuses dependency installation, and records a
bounded/redacted command observation. Node.js 20+ is a required local verification tool.

## D-009 — remote verification is executable but remains evidence-gated

The remote doctor records structured GPU, VRAM, driver, CUDA, topology, NCCL, and RDMA/IB
evidence. `remote_verify.sh` binds infrastructure, training-lifecycle, and harness JSON to a
fresh validated run ID and invocation token before copying them locally. Without all five
plan-selected runtime inputs, including a digest-pinned sandbox image, it exits `BLOCKED`. With
them, it installs only the locked training extra before the strict infrastructure gate and
requires rootless Podman with the exact image already loaded; implicit pulls are forbidden. A
real tokenization/forward/backward/optimizer/checkpoint lifecycle, including restricted
optimizer-state reload and a resumed optimizer step, plus inference, the protected
model-authored patch fixture, and general harness fixture must all pass before success.

## D-010 — pilot artifacts retain training-visible behavior

Normalized and pilot samples retain the generated trajectory and model patch, never the task
gold patch. The task-metadata join stores only the four provenance fields needed to complete a
trace. Pilot selection requires a non-empty trajectory and writes a compact manifest alongside
a directly consumable immutable JSONL sample artifact.

## D-011 — only integrity-attested oracles may report resolution

The fixed smoke harness is bound to one exact `SolveContext` and an explicit set of approved
before/after source hashes. It can pass only after the expected failing baseline, exact source
transition, protected-tree integrity, and final tests are observed. A generic repository test
command has no equivalent integrity-attested oracle, so a zero exit remains `UNVERIFIED` rather
than proving task resolution or regression safety.

## D-012 — plan activation separates metadata from execution

The tracked plan settles the teacher and three candidate identities, but explicitly requires an
empirical student selection. Immutable public metadata may report `PASS`; model load, execution,
bake-off, student selection, and training retain separate statuses and remain `NOT RUN` until
their evidence exists. Training configurations stay non-runnable rather than preselecting the
plan's current lead candidate.

## D-013 — full dataset transfer waits for large storage

The dataset registry, validation, download guard, streaming materializer, and runbook are
prepared locally. Full snapshots are not downloaded in the project workspace. Transfer and bulk
materialization require explicit current-session confirmation after the user provisions a GPU
instance with larger storage and records a capacity budget using
`docs/DATA_DOWNLOAD_RUNBOOK.md`.
