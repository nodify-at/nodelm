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

A user-provisioned Runpod host was used on 2026-08-24/25 for external-volume data work. The
current host observation reports an NVIDIA RTX PRO 6000 Blackwell Server Edition with 97,887
MiB VRAM, about 283 GB cgroup memory, a 13.6-core CPU quota, a 1 TB container disk, and the
correct persistent network volume mounted read/write at `/workspace`. The core dataset transfer
and audit used CPU/storage; it did not require or exercise the GPU. The persistent evidence is
indexed by `artifacts/reports/FULL_DATASET_AUDIT.md`.

This host observation is not the strict `remote_verify.sh` lifecycle and makes no training
compatibility or performance claim. The student is not selected; no candidate weights were
loaded, no forward/backward step ran, and real model memory has not been measured. A training
GPU recommendation therefore remains `BLOCKED`, not inferred from nominal VRAM. Recovery and
re-execution stop conditions remain in `docs/DATA_DOWNLOAD_RUNBOOK.md`.

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
