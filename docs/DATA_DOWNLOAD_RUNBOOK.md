# Full dataset download runbook

Status: `NOT RUN`

Full snapshot transfer is deliberately deferred. Do not run the commands in the execution
section from the local project workspace. Use this runbook only after the user provisions a GPU
instance with larger storage and explicitly confirms the transfer in that session.

## Already prepared locally

- All three dataset repositories and immutable revisions are pinned in
  `configs/datasets/registry.yaml`.
- Registry validation and live metadata comparison do not download snapshot contents.
- The download command refuses to run without `--confirm-large-download`.
- `data/` is Git-ignored, and materialization/audit paths stream records instead of loading an
  entire snapshot into memory.
- No full snapshot, model weight, checkpoint, or generated training artifact is present or
  required for plan activation.

## Large-storage host preflight

Before any transfer:

1. Use a dedicated data volume outside the Git checkout and record its mount path, free bytes,
   and free inodes.
2. Query repository file metadata at the pinned revisions and record the compressed byte total
   for each source. This is metadata inspection, not a snapshot download.
3. Record a capacity budget covering the snapshots, materialized JSONL/Parquet, complete audit
   ledgers, temporary files, and 20% operational headroom. Stop if observed free space is below
   that written budget.
4. Run `uv run nodelm datasets validate` and `./scripts/verify_datasets.sh`. Stop on revision or
   license drift.
5. Confirm that every destination is new or empty and that no credentials will be written into
   the repository or logs.
6. Obtain explicit current-session confirmation for the exact sources and destination volume.

## Future execution

Run one source at a time so capacity and integrity can be checked between transfers:

```bash
./scripts/download_datasets.sh \
  --source open-swe-traces \
  --destination /large-volume/nodelm/open-swe-traces \
  --confirm-large-download

./scripts/download_datasets.sh \
  --source swe-rebench-v2 \
  --destination /large-volume/nodelm/swe-rebench-v2 \
  --confirm-large-download

./scripts/download_datasets.sh \
  --source swe-rebench-v2-prs \
  --destination /large-volume/nodelm/swe-rebench-v2-prs \
  --confirm-large-download
```

After each transfer, record the source name, pinned revision, destination, wall-clock time,
on-disk bytes, and the command result. Do not begin normalization, contamination freezing, or
pilot construction until the complete-snapshot audit contract is ready and the transferred
snapshot passes it.

## Stop conditions

Stop without retrying automatically if disk headroom falls below budget, the Hub revision or
license differs from the registry, the destination is not empty, a snapshot is incomplete, or
the audit contract changes. Preserve the partial destination for inspection; do not delete or
overwrite it without explicit approval.
