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

GO / NO-GO: `NO-GO` for paid training. First complete the full-snapshot audit and contamination
freeze, deterministic common harness, candidate load/generate checks, 50–100-task bake-off, and
a measured one-step memory profile for the selected student.
