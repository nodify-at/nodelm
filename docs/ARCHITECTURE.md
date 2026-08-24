# Architecture

NodeLM is one typed Python application with a single `nodelm` CLI. Shell scripts provide
stable operator entry points but contain no research logic.

## Boundaries

- `datasets`: pinned-source registry, Hub verification, streamed snapshot materialization,
  content-bound snapshot audits/lineage, and pilot filtering.
- `provenance`: strict normalization into versioned, training-visible sample records with an
  optional disk-backed safe-field task join.
- `licenses`: conservative repository-level V1 policy; decisions remain auditable.
- `decontamination`: canonical identities, duplicate fingerprints, and repository splits.
- `harness`: trusted-fixture command policy, execution, failure classification, and
  TypeScript workspace discovery.
- `teacher`: backend-neutral rollout records with a model-visible `SolveContext` that cannot
  represent a gold patch, plus isolated repository execution whose generic test success remains
  `UNVERIFIED` without an integrity-attested oracle.
- `evaluation`: common candidate and response contracts plus a protected patch fixture bound to
  an exact `SolveContext` and approved before/after source identities; measurements remain
  nullable until observed.
- `training`: a required lifecycle protocol for load, tokenize, forward/backward/step,
  checkpoint save/reload with optimizer state, a resumed optimizer step, and inference.
- `infra`: local/remote machine evidence and explicit verification statuses.

Domain schemas reject unknown fields. Adapters may translate external data but may not weaken
provenance, license, split, or status semantics.

## Artifact contract

Canonical JSON uses sorted keys, compact separators, UTF-8, and a final newline. Writes are
atomic, immutable, and stream-capable: a deterministic path may be reused only for identical
bytes. Stable IDs derive from canonical inputs; timestamps are evidence metadata, not logical
identity. Dataset audits bound percentile/rejection examples while streaming the complete
rejection ledger; repository splitting uses a disk-backed index. Complete-snapshot identity
covers normalized sorted relative paths plus the bytes and digest of every supported `.jsonl`
and `.parquet` file discovered below the supplied local path—not arbitrary repository metadata.
The guarded download records a transfer receipt that joins the pinned source and registry bytes
to that aggregate; a filtered receipt cannot authorize a complete audit. Offline audit first
copies those exact files into a private staged view and parses only the verified staged bytes,
closing the source-path gap between hashing and row decoding. Legacy raw-file audit reports retain
the `nodelm.dataset-audit/v1` shape; complete aggregate-snapshot reports use
`nodelm.dataset-audit/v2` and name their identity scheme. The lineage manifest binds the receipt,
registry, snapshot, logical rows, canonical report, and ledger. Inputs and published dependencies
are revalidated once at each immutable publication boundary, and lineage is published last.

Parquet datasets must also record a logical row digest because byte identity can vary with the
writer version. Generated data, weights, and checkpoints stay outside Git.

## Trust boundary

The local harness uses argv-only processes, path containment, a sanitized environment,
timeouts, output limits, redaction, and process-group termination. These controls are not a
security sandbox; only checked-in trusted fixture code may be executed by that backend.

Untrusted model-authored code runs only in a preloaded, digest-pinned rootless Podman image with
implicit pulls disabled. The container receives the temporary host workspace as a read-only
bind, has no network, drops capabilities, runs as an unprivileged user with bounded
CPU/memory/PIDs/file resources, and is force-removed even after timeout or cancellation. Patch
input is bounded and text-only. Generic protected paths reject symlink components before test
execution.

The snapshot audit's unpredictable private staging directory isolates parsing from ordinary
changes to the original download path. It is a trusted-local, same-account boundary—not a
security sandbox against a hostile process already running as the same OS user.
