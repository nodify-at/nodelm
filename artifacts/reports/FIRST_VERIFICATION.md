# First verification

Overall status: `NOT RUN`

## Local readiness evidence

| Check | Status | Evidence |
|---|---|---|
| Locked bootstrap | PASS | `./scripts/bootstrap.sh`; 65 locked packages audited |
| Required tools | PASS | Python 3.11.7; Node v24.10.0; Git, uv, and rg present |
| Static and test gates | PASS | Ruff, strict mypy, 182 tests, 82.90% branch-aware coverage |
| Node fixture harness | PASS | `node --test`; 1 test passed; structured command evidence emitted |
| Python package build | PASS | source distribution and `py3-none-any` wheel built successfully |
| Training config parse | PASS | strict `nodelm.training-config/v1` validation |
| Model load/training | NOT RUN | no pinned model from the missing ground-truth plan |

| Area | Check | Status | Evidence |
|---|---|---|---|
| Infrastructure | GPU/CUDA/resources | NOT RUN | No remote host supplied or requested |
| Model | Pinned download/load/inference | BLOCKED | Candidate model not selected |
| Dataset | Pinned subset/normalize/tokenize | NOT RUN | Requires selected training input |
| Training | Initial step/checkpoint/reload/resumed step | NOT RUN | Requires verified model and GPU host |
| Evaluation | Fixture task through model and harness | NOT RUN | Requires verified model adapter |

GO / NO-GO: `NO-GO` for rented training until the ground-truth plan is available, a model is
verified, and a measured memory estimate supports a minimal infrastructure request.
