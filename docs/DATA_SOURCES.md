# Data sources

Snapshot metadata below was checked against official Hugging Face APIs on 2026-08-24 and is
pinned in `configs/datasets/registry.yaml`.

| Dataset | Revision | Rows | Dataset license |
|---|---|---:|---|
| [`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | `ed95cef24df8d8bd79b4ceb0192cb420fde06521` | 567,824 | CC-BY-4.0 |
| [`nebius/SWE-rebench-V2`](https://huggingface.co/datasets/nebius/SWE-rebench-V2) | `475dd5e8703bb5fb22dd3c60b5d038b019eba1e0` | 32,079 | CC-BY-4.0 |
| [`nebius/SWE-rebench-V2-PRs`](https://huggingface.co/datasets/nebius/SWE-rebench-V2-PRs) | `fbf0ecf50f268d5344149e2f0097db6bede83737` | 126,300 | CC-BY-4.0 |

Official metadata APIs: [Open-SWE](https://huggingface.co/api/datasets/nvidia/Open-SWE-Traces),
[V2](https://huggingface.co/api/datasets/nebius/SWE-rebench-V2), and
[V2-PRs](https://huggingface.co/api/datasets/nebius/SWE-rebench-V2-PRs).

## Verified discrepancies

- Open-SWE's pinned snapshot contains 567,824 rows and 231 Parquet files across 11 leaf
  `(harness, generating-model label, upstream task source)` partitions, while its README still
  shows an older 207,489 total. The checked-in partition contract binds the exact receipt
  inventory; paper/card counts are not substituted.
- Recent Open-SWE rows add `hf_dataset_name`. Seven leaves identify
  `nebius/SWE-rebench-V2`; four identify `AweAI-Team/Scale-SWE`. Only the seven Rebench-backed
  leaves have a pinned task-source join today.
- The sealed registry spells one historical descriptive split as `deepseek_v4_flash`, while the
  receipt-bound snapshot path is `deepseek-v4-flash`. The registry bytes are preserved because
  every transfer receipt binds them; `configs/datasets/open-swe-trace-partitions.yaml` records
  the observed path separately.
- SWE-rebench-V2 and V2-PRs expose `created_at` as a string although prose describes an
  integer. Parsers follow the observed pinned schema.
- V2-PRs has no top-level language field and its nested metadata is irregular. Language must
  be derived and verified, not assumed.

Dataset-level CC-BY-4.0 metadata does not override source-repository licenses. Every row is
gated on its repository license before training use.
