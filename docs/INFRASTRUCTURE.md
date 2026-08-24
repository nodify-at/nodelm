# Infrastructure

NodeLM keeps dataset, harness, evaluation, and training contracts independent of hardware.
Named planning targets include one or two DGX Spark/GB10 systems, RTX PRO 5000 72 GB, RTX PRO
6000 96 GB, H100, H200, and B200. Listing a target is not a recommendation or compatibility
claim.

Run `./scripts/remote_doctor.sh --output artifacts/reports/infra/<host>.json` on a candidate
machine. The first remote milestone requires observed OS, CPU, RAM, disk, GPU/VRAM, driver,
CUDA, and—when applicable—NVLink/NVSwitch, NCCL, and RDMA/IB evidence. It then requires a pinned
model load, one pinned batch, a real forward/backward/optimizer step, checkpoint save/reload
with optimizer state, a resumed optimizer step, inference, and one fixture repository task.

The versioned host report records each GPU as structured index, UUID, model, VRAM-byte, and
driver evidence. CUDA runtime-library and toolkit versions are separate checks. Multi-GPU hosts
also record the observed topology and NCCL library availability; RDMA/InfiniBand query evidence
is recorded whenever suitable host tools exist. Missing tools or hardware remain `NOT RUN` with
an explicit `UNAVAILABLE` or `NOT_APPLICABLE` availability value. A successful availability
query may truthfully report that an optional fabric is unavailable; that is evidence, not a
fabric-performance claim. No NCCL or RDMA benchmark is implied by discovery alone.

No infrastructure request is issued yet. The plan identifies three candidates, but the student
is not selected and real load/training memory has not been measured. Any training GPU
recommendation is therefore `BLOCKED`, not inferred from marketing capacity. Full dataset
transfer is also deferred until the user provisions a GPU instance with larger storage; the
preflight and stop conditions are recorded in `docs/DATA_DOWNLOAD_RUNBOOK.md`.

SSH inputs, when needed, are limited to host/IP, username, non-standard port, and a local key
path or configured alias. Never request private-key contents, weaken host-key checking, modify
SSH configuration, or expose services publicly.

The SSH wrapper uses batch mode, strict host-key checking, a 10-second connection timeout,
one connection attempt, and bounded server-alive probes. Each invocation has a validated run ID
and a fresh remote run directory. A per-invocation marker token binds every copied report to
that run, so an old fixed-path report cannot satisfy a new verification attempt. Local evidence is
published under `artifacts/reports/infra/runs/host-<host>/<run-id>/` as separate infrastructure,
training-lifecycle, and general-harness JSON files.

Infrastructure collection alone is not first verification: `remote_verify.sh` returns
`BLOCKED` unless a remote pinned runtime config, matching pilot manifest/sample JSONL, fresh
checkpoint path, and digest-pinned sandbox image are supplied together. With those five inputs,
it installs the locked training dependency set before the strict infrastructure gate. Full
verification also requires rootless Podman and the exact digest-pinned image to be preloaded;
the workflow never pulls an image. It then runs the initial optimizer step, restricted
optimizer-state reload, a resumed optimizer step, deterministic inference, the protected
model-authored patch fixture, and general harness fixture. Any failed gate returns nonzero while
preserving valid current-run reports produced before the failure.
