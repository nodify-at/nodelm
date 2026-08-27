# Full dataset download runbook

Real transfer and receipt-bound core audit status: `PASS` (completed 2026-08-24)

One real receipt-replayed normalization canary is `PASS`, and all seven eligible full-partition
Open-SWE leaves now have terminal evidence: four labeled MiniMax/Qwen3.5 leaves are `PASS`, while
the three Qwen3.6 leaves are truthful `FAIL` because every source resolution is unknown.
Resolution-recovery derivation is also complete, but its immutable manifest remains admission
`BLOCKED`: the terminal real-repository canary completed 3 `PASS` / 9 `FAIL`. D-021 advances V1
only with the four labeled `PASS` leaves and preserves all Qwen3.6 unknown rows as quarantined
Tier D. A real `PASS` cohort at commit `24f4eb75f16f6782fdfa85762d3a27cd7fdbef10`
binds those four leaves as 160,731 globally unique samples and 51,063,015,261 ordered bytes;
its manifest SHA-256 is `10734b8e20d127bfe69df8c5ffd3c8540038cfa95ad2d2c230ea92fa7e8d2621`
and its population SHA-256 is `e91207cdca52c6fd08d0fd672c482fb7072117856f380b0b951bbe403fa85269`.
The hardened builder at commit `af476d1e85da133b623456d2a34f0ef12a25b857` replayed the same
four real members and reproduced both identities exactly.
All four structural gold scans are `PASS` with zero findings. Their oracle-isolation components
and overall audits remain `BLOCKED`; decontamination and split freezing remain `NOT RUN`. The
completed snapshots and raw evidence live outside Git under
`/workspace/nodelm` on persistent storage; their compact digest index is
`artifacts/reports/FULL_DATASET_AUDIT.md`. No re-download is currently required. Use this runbook
only for canary execution, recovery, or an explicitly authorized re-execution on a new, empty
external-volume destination. Never run transfer commands from the local project workspace.
The cohort evidence is retained at
`/workspace/nodelm/derived/normalization-cohort-24f4eb75f16f6782fdfa85762d3a27cd7fdbef10`;
the hardened replay evidence is retained at
`/workspace/nodelm/derived/normalization-cohort-af476d1e85da133b623456d2a34f0ef12a25b857`.
The structural audit evidence is retained at
`/workspace/nodelm/audits/gold-exposure-structural-24f4eb75f16f6782fdfa85762d3a27cd7fdbef10`.

## Next CPU gate — oracle-isolation raw-context review

The offline four-leaf runner is prepared but has not yet produced real evidence. It reads the
existing full-normalization raw and normalized artifacts, uses local temporary storage for one
private staged file at a time, and writes only compact attestations/findings under the persistent
audit root. No download, GPU, model load, training, or paid inference is involved.

After pulling the exact clean commit into `/workspace/nodelm/repo`, launch it detached:

```bash
cd /workspace/nodelm/repo
git status --short
nohup bash scripts/run_oracle_isolation_reviews.sh \
  >/workspace/nodelm/logs/oracle-isolation-review.launch.log 2>&1 &
```

Monitor or stop it without keeping the SSH session open:

```bash
NODELM_RUN_COMMIT="$(git -C /workspace/nodelm/repo rev-parse HEAD)"
NODELM_ORACLE_DIR="/workspace/nodelm/audits/oracle-isolation-review-${NODELM_RUN_COMMIT}"
cat "${NODELM_ORACLE_DIR}/run.state"
grep -c ' COMPLETE leaf=' "${NODELM_ORACLE_DIR}/events.log"
tail -n 12 "${NODELM_ORACLE_DIR}/events.log"

# Clean stop between leaves:
touch "${NODELM_ORACLE_DIR}/STOP"
# During an active leaf, use the PID recorded in run.state:
kill -TERM "$(sed -n 's/^pid=//p' "${NODELM_ORACLE_DIR}/run.state")"
```

`COMPLETE` means four structural v2 reviews are `PASS` and still await exact digest review and
code authorization. It does not yet authorize a gold-exposure audit, split, pilot, or training.

## Completed evidence and safeguards

- All three dataset repositories and immutable revisions are pinned in
  `configs/datasets/registry.yaml`.
- Registry validation and live metadata comparison do not download snapshot contents.
- The download command refuses to run without `--confirm-large-download`.
- The offline `datasets audit-snapshot` command and strict lineage contract are tested with
  synthetic JSONL/Parquet fixtures. The guarded download emits an immutable transfer receipt
  that binds the pinned source and registry to every supported data-file identity. Offline audit
  requires that complete receipt, parses a matching private staged copy, emits an aggregate
  `nodelm.dataset-audit/v2` report, and binds the receipt, logical rows, report, and complete
  rejection ledger.
- `data/` is Git-ignored, and materialization/audit paths stream records instead of loading an
  entire snapshot into memory.
- No full snapshot, model weight, checkpoint, or generated training artifact is committed to
  the project workspace. The three real snapshots and their receipt/audit/rejection/lineage
  bundles are retained on the external persistent volume.

## Resolution-recovery derivation (completed, admission blocked)

The CPU/data-integrity runner completed at commit
`74c9b505eb1a608431ae3a18a3fca5d084f2ae3b`. It derived exact task-and-patch label-transfer
candidates plus a deduplicated evaluator queue. The validated manifest remains admission
`BLOCKED`; it did not download data, start a repository evaluator, load a model, train, use a GPU,
or admit recovered labels into normalization.

Terminal evidence is retained at
`/workspace/nodelm/derived/resolution-recovery-74c9b505eb1a608431ae3a18a3fca5d084f2ae3b`.
The run covered 179,533 target rows: 123,709 ineligible, 3,960 candidate rows, and 51,864 queued
fanout rows. It published 1,804 unique transfer candidates and 49,572 unique evaluator requests
with zero conflicts. Exact artifact hashes are recorded in `docs/EXPERIMENTS.md`.

The production layout is fixed to the persistent Runpod mount:

- clean exact-commit checkout: `/workspace/nodelm/repo`;
- Open-SWE snapshot: `/workspace/nodelm/snapshots/open-swe-traces`;
- sealed receipt: `/workspace/nodelm/receipts/open-swe-traces.transfer.json`;
- shared persistent root: `/workspace/nodelm`;
- exact output directory:
  `/workspace/nodelm/derived/resolution-recovery-<full-40-character-HEAD>`.

Launch only from a clean committed checkout. The runner binds the exact commit, tree, runner blob,
registry, partition contract, and transfer receipt, then stays fully offline:

```bash
cd /workspace/nodelm/repo
git status --short
mkdir -p /workspace/nodelm/logs
nohup bash scripts/run_resolution_recovery.sh \
  >/workspace/nodelm/logs/resolution-recovery.launch.log 2>&1 &
```

Resolve the content-addressed run directory from that exact commit and inspect its durable state
without keeping an SSH session open:

```bash
NODELM_RUN_COMMIT="$(git -C /workspace/nodelm/repo rev-parse HEAD)"
NODELM_RECOVERY_DIR="/workspace/nodelm/derived/resolution-recovery-${NODELM_RUN_COMMIT}"
cat "${NODELM_RECOVERY_DIR}/run.state"
tail -n 50 "${NODELM_RECOVERY_DIR}/events.log"
tail -n 100 "${NODELM_RECOVERY_DIR}/runner.log"
```

`run.state` records the runner PID, commit, phase, detail, and stop-file path. Before
`phase=build`, request a clean stop by creating the recorded sentinel:

```bash
touch "${NODELM_RECOVERY_DIR}/STOP"
```

Once `phase=build` is active, the sentinel is no longer polled. Read the current PID from
`run.state` and send TERM to that runner; it forwards TERM to its active child and records
`STOPPED`:

```bash
NODELM_RECOVERY_PID="$(sed -n 's/^pid=//p' "${NODELM_RECOVERY_DIR}/run.state")"
kill -TERM "${NODELM_RECOVERY_PID}"
```

For the same exact commit, a restart uses the same content-addressed directory. A complete
three-artifact terminal set is validated and reported `COMPLETE` without rebuilding. With no
terminal artifacts, move the `STOP` sentinel aside and rerun the same launch command. A partial set
that has the matching immutable `run.binding` is rebuilt deterministically: existing artifacts are
reused only when byte-identical and missing artifacts are published without overwriting prior
evidence. A differing existing artifact, or any terminal artifact without a binding, fails closed.
Preserve partial outputs and temporary staging remnants for inspection; do not remove them without
explicit approval. Any code change creates a different commit-bound run directory. A successful
recovery derivation historically ended with `admission=BLOCKED` and `harness_canary_pending`. Its
later terminal evaluator canary is now complete: 3 cases passed, 9 failed, and recovery admission
remains `BLOCKED`.

## Resolution harness canary (completed, admission blocked)

The terminal canary ran at commit `3c9690abc7f676762f613b84897ca5cf0156cc4a` and retained
validated evidence under
`/workspace/nodelm/derived/resolution-canary-3c9690abc7f676762f613b84897ca5cf0156cc4a`.
Its execution verdict is `FAIL` (3 passed, 9 failed) and recovery admission is `BLOCKED`; do not
rerun it merely to replace failed cases. Exact digests and failure accounting are in
`docs/EXPERIMENTS.md`.

For an explicitly authorized new-version investigation, the canary is a sequential CPU workload;
it needs no GPU and does not download any dataset. It
does need outbound network during preparation to clone the already-pinned public evaluator and
pull only the selected task images. Repository tests then run without network access. Use an x86_64
Linux host with the existing `/workspace` persistent mount, at least 16 host CPUs, 64 GiB RAM, and
ample local temporary storage. The default backend requires rootless Podman, user namespaces,
cgroup resource delegation, `fuse-overlayfs`, and `slirp4netns`, and must run unprivileged.

Runpod pods currently prohibit the nested namespaces needed by Podman. On those restricted hosts,
install `skopeo`, `umoci`, and `libseccomp2`, then select `seccomp-chroot` and run as root. This
backend binds the public registry digest to a locally hashed OCI manifest and gives each attempt a
fresh reflinked rootfs, dedicated numeric UID, chroot, no-new-privileges, dropped capability bounds,
network-denying seccomp filter, two-CPU affinity, monitored 4 GiB memory/512-process bounds, file and
output limits, timeout, and verified cleanup. OCI layers and rootfs clones stay on ephemeral local
storage; private worksets, evidence, state, and manifests stay under persistent `/workspace`.

After checking out the requested canary commit cleanly at `/workspace/nodelm/repo`, launch:

```bash
mkdir -p /workspace/nodelm/logs
NODELM_CANARY_RUNTIME=seccomp-chroot \
NODELM_IMAGE_ROOT=/var/lib/nodelm-canary/images \
nohup bash /workspace/nodelm/repo/scripts/run_resolution_canary.sh \
  >/workspace/nodelm/logs/resolution-canary.launch.log 2>&1 &
```

The exact output directory is content-bound to the checked-out commit:

```bash
NODELM_RUN_COMMIT="$(git -C /workspace/nodelm/repo rev-parse HEAD)"
NODELM_CANARY_DIR="/workspace/nodelm/derived/resolution-canary-${NODELM_RUN_COMMIT}"
cat "${NODELM_CANARY_DIR}/run.state"
tail -n 50 "${NODELM_CANARY_DIR}/events.log"
```

Before evaluation begins, request a clean stop with `touch "${NODELM_CANARY_DIR}/STOP"`. During
image pulling or evaluation, read `pid=` from `run.state` and send TERM; the runner stops its
active process group, cleans the active sandbox, and records `STOPPED`. Per-case private evidence
is immutable and restartable, so a same-commit restart reuses completed cases. Workset and image-lock
publication are also safely reusable after an interruption.
Terminal `COMPLETE` means all artifacts and raw-log hashes validated; read
`resolution-canary.execution.manifest.json` for the separate execution and admission verdicts.

## Large-storage host preflight

Before any transfer:

1. Use a dedicated data volume outside the Git checkout and record its mount path, free bytes,
   and free inodes.
2. Query repository file metadata at the pinned revisions and record the compressed byte total
   for each source. This is metadata inspection, not a snapshot download.
3. Record a capacity budget covering the snapshots, one temporary private staging copy of the
   source currently under audit, materialized JSONL/Parquet, complete audit ledgers, temporary
   SQLite state, and 20% operational headroom. Stop if observed free space is below that budget.
4. Run `uv run nodelm datasets validate` and `./scripts/verify_datasets.sh`. Stop on revision or
   license drift.
5. Confirm that every destination is new or empty and that no credentials will be written into
   the repository or logs.
6. Obtain explicit current-session confirmation for the exact sources and destination volume.

## Controlled re-execution only

Do not run these commands against the completed paths. After fresh current-session confirmation,
run one source at a time into new, empty destinations so capacity and integrity can be checked
between transfers:

```bash
./scripts/download_datasets.sh \
  --source open-swe-traces \
  --destination /large-volume/nodelm/snapshots/open-swe-traces \
  --receipt-output /large-volume/nodelm/receipts/open-swe-traces.transfer.json \
  --confirm-large-download

./scripts/download_datasets.sh \
  --source swe-rebench-v2 \
  --destination /large-volume/nodelm/snapshots/swe-rebench-v2 \
  --receipt-output /large-volume/nodelm/receipts/swe-rebench-v2.transfer.json \
  --confirm-large-download

./scripts/download_datasets.sh \
  --source swe-rebench-v2-prs \
  --destination /large-volume/nodelm/snapshots/swe-rebench-v2-prs \
  --receipt-output /large-volume/nodelm/receipts/swe-rebench-v2-prs.transfer.json \
  --confirm-large-download
```

After each transfer, record the source name, pinned revision, destination, wall-clock time,
on-disk bytes, and the command result. Then audit that transferred snapshot without network or
download behavior, keeping outputs outside the snapshot tree:

```bash
uv run nodelm datasets audit-snapshot \
  --source open-swe-traces \
  --snapshot /large-volume/nodelm/snapshots/open-swe-traces \
  --receipt /large-volume/nodelm/receipts/open-swe-traces.transfer.json \
  --staging-root /large-volume/nodelm/staging \
  --output /large-volume/nodelm/audits/open-swe-traces.audit.json \
  --lineage-output /large-volume/nodelm/audits/open-swe-traces.lineage.json \
  --rejections-output /large-volume/nodelm/audits/open-swe-traces.rejections.jsonl \
  --config configs/datasets/registry.yaml
```

Repeat with the matching source, receipt, and paths for the other snapshots. Retain the transfer
receipt, audit JSON, complete rejection JSONL, and lineage JSON together. The lineage file is
published last and marks a mutually bound artifact set. Private staging is transient; after an
interrupted run, inspect any tool-owned staging directory before removing it. `complete-snapshot`
means all supported `.jsonl` and `.parquet` files discovered below the supplied path; it does not
attest ignored repository metadata or unsupported formats. Private staging isolates the audit
from ordinary changes to the original source path; it is not a sandbox against a hostile process
running under the same OS account.

The core audit prerequisite now reports `PASS` for all three pinned sources. Do not begin pilot
construction or paid training until the remaining Phase 0/decontamination work is complete.
Tokenizer-based trajectory lengths, exact/near patch duplication, public-evaluation overlap,
unique issue/PR counts, and harness/model distributions are not established by this command and
remain `NOT RUN`.

## Stop conditions

Stop without retrying automatically if disk headroom falls below budget, the Hub revision or
license differs from the registry, the destination is not empty, a transfer receipt is missing
or filtered, its source/registry/inventory does not match, private staging changes, or the audit
contract changes. Also stop if a previously published dependency changes during publication.
Preserve partial destinations, tool-owned staging remnants, and orphan immutable evidence for
inspection; do not delete or overwrite them without explicit approval.
