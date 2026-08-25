# Full dataset download runbook

Real transfer and receipt-bound core audit status: `PASS` (completed 2026-08-24)

Normalization, decontamination, and split freezing remain `NOT RUN`. The completed snapshots and
raw evidence live outside Git under `/workspace/nodelm` on persistent storage; their compact
digest index is `artifacts/reports/FULL_DATASET_AUDIT.md`. No re-download is currently required.
Use this runbook only for recovery or an explicitly authorized re-execution on a new, empty
external-volume destination. Never run the transfer commands from the local project workspace.

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
