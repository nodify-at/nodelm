# First verification

Overall status: `NOT RUN`

## Local readiness evidence

| Check | Status | Evidence |
|---|---|---|
| Locked bootstrap | PASS | `./scripts/bootstrap.sh`; 65 locked packages audited |
| Required tools | PASS | Python 3.11.7; Node v24.10.0; Git, uv, and rg present |
| Static and test gates | PASS | Ruff, strict mypy, 217 tests, 83.28% branch-aware coverage |
| Node fixture harness | PASS | `node --test`; 1 test passed; structured command evidence emitted |
| Python package build | PASS | source distribution and `py3-none-any` wheel built successfully |
| Training config parse | PASS | strict `nodelm.training-config/v1` validation |
| Model/teacher metadata | PASS | strict registries validate exact revisions and official evidence |
| Model load/training | NOT RUN | no weights downloaded; no student selected by the bake-off |

| Area | Check | Status | Evidence |
|---|---|---|---|
| Infrastructure | GPU/CUDA/resources | NOT RUN | No remote host supplied or requested |
| Model | Pinned download/load/inference | NOT RUN | Candidate execution and bake-off are not run |
| Dataset | Full snapshots/audit/normalize/tokenize | NOT RUN | Deferred to large-storage GPU instance |
| Training | Initial step/checkpoint/reload/resumed step | NOT RUN | Requires verified model and GPU host |
| Evaluation | Fixture task through model and harness | NOT RUN | Requires verified model adapter |

## Dataset audit update — 2026-08-24

After the original readiness snapshot above, all three pinned real dataset snapshots were
transferred to external persistent storage and their receipt-bound core audits and lineage
manifests reported `PASS`: 726,203 rows across 49,734,682,463 supported bytes. See
`artifacts/reports/FULL_DATASET_AUDIT.md` for the bound revisions and digests. This supersedes the
Dataset row only for full transfer and core snapshot audit. Normalization, tokenizer-based
measurements, decontamination, split freezing, and pilot construction remain `NOT RUN`.

An RTX PRO 6000 host is available and has been observed for data work, but the strict remote
model lifecycle, candidate load/generate checks, training, and evaluation remain `NOT RUN`.
Accordingly the overall First Verification status and training decision below are unchanged.

GO / NO-GO: `NO-GO` for paid training. First complete real-source normalization and the
contamination freeze, deterministic common harness, candidate load/generate checks, 50–100-task
bake-off, and a measured one-step memory profile for the selected student.
