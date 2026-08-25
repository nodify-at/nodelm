# Full dataset audit

Core receipt-bound snapshot audit status: `PASS`

Completion marker: `2026-08-24T23:13:24Z`

Code revision: `bfcc3b6a46e76043a50acced55d45bb16bf3cbb6`

Dataset registry SHA-256:
`f92315a70a0c75ec909d83f4cb639b3a320f62526069f11ca87f0fe1d891637f`

This report is the compact Git-tracked index for the real-source transfer and core audit.
The snapshots and complete receipt/audit/rejection/lineage evidence remain on the external
persistent volume under `/workspace/nodelm/{snapshots,receipts,audits,logs}`; they are not
committed to Git. On 2026-08-25, the evidence bundle was copied to a separate verifier and every
recorded artifact digest was checked independently with strict SHA-256 verification.

## Receipt-bound source results

| Source | Pinned revision | Supported files | Supported bytes | Rows | License-gate rejections | Status |
|---|---|---:|---:|---:|---:|---|
| `swe-rebench-v2` | `475dd5e8703bb5fb22dd3c60b5d038b019eba1e0` | 1 | 428,839,266 | 32,079 | 6,023 | PASS |
| `swe-rebench-v2-prs` | `fbf0ecf50f268d5344149e2f0097db6bede83737` | 3 | 2,682,022,715 | 126,300 | 27,914 | PASS |
| `open-swe-traces` | `ed95cef24df8d8bd79b4ceb0192cb420fde06521` | 231 | 46,623,820,482 | 567,824 | 0 | PASS |
| **Total** | — | **235** | **49,734,682,463 (46.319 GiB)** | **726,203** | **33,937** | **PASS** |

All observed row counts exactly matched their declarations. Every audit and lineage manifest
reported `PASS`, and every audit had an empty issues list. The gate classified 692,266 rows as
`ALLOW`, 33,878 as `UNKNOWN`, and 59 as `REJECT`; both `UNKNOWN` and `REJECT` rows are retained
in the complete rejection ledgers. `PASS` therefore means that the pinned complete-snapshot
identity, parsing, counts, ledger, and lineage contracts passed—not that every row is ready for
training.

## Bound identities

| Source | Snapshot SHA-256 | Logical rows SHA-256 |
|---|---|---|
| `swe-rebench-v2` | `4f4328b560d27918da8f2d251c037789add5b5f7566c46825eeed91aa9d9c117` | `8da7947c8ee99d667e0930c53e813ab9fe38fca07d11ffb693f73514fca7908f` |
| `swe-rebench-v2-prs` | `c2e6edf039c1e49f4cc4193c0b5cb26eb3376b21f71bae7148b17b157b463780` | `1cff658ac15aaa794afb7a1c0ceb2090d695b82898c2f969f593f199dbc9833f` |
| `open-swe-traces` | `218df319b86b21be12f22284e25cdbaf90e77fdbc3e2d996c152ec8c54c03aa3` | `c4090707daf6eb1c623d0b31015c78934e4555f24d27058b56954356acba47f9` |

The Open-SWE repository download fetched 233 repository files, while its snapshot identity
intentionally covers the 231 supported Parquet data files. Repository metadata files are
outside that identity.

## Canonical artifact manifest

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `swe-rebench-v2.transfer.json` | 1,308 | `fbcd4fbb2b9c4b887ef15f368f3673c07d82d4ba81d2b0d0eed7e3dd6d1fe254` |
| `swe-rebench-v2.audit.json` | 225,571 | `0c3af9007e090a73005ad590f6b858ed00fe79793570a832871e6829ada8e8ab` |
| `swe-rebench-v2.rejections.jsonl` | 1,351,299 | `0bf18d87d7e5b1904515446470017116a562b5714e1a738f5b094927a3b2e117` |
| `swe-rebench-v2.lineage.json` | 2,146 | `7def77832d5501b626278f964e7e385e0da8ef4bc33e7db93553ec0b9f4b8905` |
| `swe-rebench-v2-prs.transfer.json` | 1,598 | `d7e6c8e4abb7a8488c62588a0cc95c089bc12603c22c9cf85f2acad2a1c59570` |
| `swe-rebench-v2-prs.audit.json` | 255,324 | `384f12f76de602c3b025a83cddf172ba70d4e4daf6ce4484a53554083b140829` |
| `swe-rebench-v2-prs.rejections.jsonl` | 6,241,453 | `78ab9e0337a7791d7a6b8d94bd5b7575b3ae943b92d68baad4a7b82a8778088b` |
| `swe-rebench-v2-prs.lineage.json` | 2,450 | `fc150739a7fde88a9f47333ab3b0ed93304d58024d4c3ab5772509d2a2834d68` |
| `open-swe-traces.transfer.json` | 41,712 | `44ea157ebd802a5604301c82e8785003d67f90d0ed64efcc079059dfd4290a84` |
| `open-swe-traces.audit.json` | 29,557 | `13e539eaf3775ae9d2bad856946915f65402346a8de71cadd60c88a28b681b9a` |
| `open-swe-traces.rejections.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `open-swe-traces.lineage.json` | 42,552 | `7129e1337e8223efe1328d1813c9b2973a656afbfcd8cb6f6412f50875c3e328` |

## Audit observations

| Source | Unique repositories | Repeated instance IDs | Resolution: resolved / unresolved / unknown | Patch bytes: min / p50 / p95 / max |
|---|---:|---:|---|---|
| `swe-rebench-v2` | 3,615 | 0 | 0 / 0 / 32,079 | 198 / 3,931 / 22,992 / 252,252 |
| `swe-rebench-v2-prs` | 3,221 | 1,920 | 0 / 0 / 126,300 | 174 / 8,049 / 100,000 / 298,366 |
| `open-swe-traces` | 3,788 | 42,473 | 65,244 / 95,487 / 407,093 | 1 / 3,445 / 82,907 / 83,790,107 |

Repeated-ID counts are counts of distinct instance IDs that occur more than once inside one
source. They are not duplicate-row, duplicate-patch, or cross-source counts. Per-source unique
repository counts cannot be summed because cross-source overlap has not been measured.

The audit observed 122,081 TypeScript/JavaScript-labeled Open-SWE rows across the raw `ts`,
`typescript`, `js`, and `javascript` spellings, plus 8,342 `ts`/`js` rows in SWE-rebench V2.
SWE-rebench V2 PRs has no top-level language field, so all 126,300 rows remain `UNKNOWN` for
this measurement. Raw aliases are intentionally not normalized by the core audit.

Open-SWE trajectory lengths were 15 / 159 / 297 / 501 at min / p50 / p95 / max. These are
trajectory element counts, not tokenizer tokens. Zero trajectory lengths in the two task/oracle
sources mean that the list-valued trajectory field is absent, not that their records are
zero-token training examples. Capped p50/p95 distribution measurements are deterministic
approximations where the audit says so; min/max values are exact.

## Execution record and remaining gates

The corrected two-source runner started at `2026-08-24T22:26:51Z` and published its completion
marker at `2026-08-24T23:13:24Z`, a wall-clock span of 46 minutes 33 seconds. An earlier
invocation failed before transfer because it used the invalid `--receipt` option instead of
`--receipt-output`; the corrected invocation then completed normally. SWE-rebench V2 had
already completed as the preceding real-source smoke run.

At this audit's 2026-08-24 completion boundary, full normalization and all later gates were
`NOT RUN`. On 2026-08-25, all seven eligible full-partition leaves reached terminal normalization
evidence: four MiniMax/Qwen3.5 leaves are `PASS`, while three all-unknown Qwen3.6 leaves are
truthful `FAIL`; exact hashes are recorded in `docs/EXPERIMENTS.md`.

Tokenizer-based trajectory lengths, unique issue/PR counts, harness and generating-model
distributions, exact/near patch duplication, public-evaluation overlap, resolution recovery,
contamination-safe split freezing, pilot construction, candidate model execution, the student
bake-off, training, and evaluation remain `NOT RUN`. The core audit `PASS` is one Phase 0
prerequisite; it does not complete Phase 0 or authorize paid training.
