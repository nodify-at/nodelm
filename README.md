# NodeLM

NodeLM is a hardware-agnostic research codebase for auditing software-engineering
datasets, building contamination-safe TypeScript/JavaScript subsets, exercising a
repository-level coding harness, comparing student models, and preparing reproducible
fine-tuning and evaluation runs.

The repository is designed to make local verification complete before rented GPU
infrastructure is requested. Generated measurements are never committed as if they were
verified facts; each artifact records its inputs, source revisions, timestamps, and status.

## Local bootstrap

Requirements: Git, `uv`, Python 3.11, Bash, `rg`, and Node.js 20 or newer. The default
verification gate runs the checked-in Node harness fixture. npm/pnpm and GPU tooling remain
optional until their explicit workflows are used.

```bash
./scripts/bootstrap.sh
./scripts/doctor.sh
make verify
```

The bootstrap creates `.venv`, installs the locked default and development dependencies,
and leaves optional GPU/training dependencies out of the local environment. Install those
only on a compatible machine with:

```bash
uv sync --extra training --group dev
```

## Common workflows

Full dataset transfer is intentionally deferred until a large-storage GPU instance is
provisioned. [`docs/DATA_DOWNLOAD_RUNBOOK.md`](docs/DATA_DOWNLOAD_RUNBOOK.md) is the sole
full-transfer procedure: run it only on that future host, with its absolute external-volume
destinations and explicit current-session confirmation. No full snapshot is required for the
current plan-activation work.

```bash
# Validate the dataset registry without downloading snapshots.
uv run nodelm datasets validate

# Check the pinned revision and dataset-card license against the live Hub.
./scripts/verify_datasets.sh

# Verify the local repository harness against the fixture project.
./scripts/verify_harness.sh

# Validate training configuration without loading a model.
./scripts/training_smoke_test.sh --dry-run

# Once the bake-off selects a student and a schema-valid runtime config is verified, execute one
# real lifecycle:
# tokenize a pilot batch, run an optimizer step, save/reload model and optimizer state, run a
# resumed optimizer step, then inference and a model-authored patch. NODELM_SANDBOX_IMAGE names an
# already-loaded rootless Podman image as name@sha256:<64 lowercase hexadecimal characters>.
./scripts/training_lifecycle.sh \
  --config configs/training/tiny-lora.yaml \
  --samples artifacts/manifests/pilot-sft.samples.jsonl \
  --pilot-manifest artifacts/manifests/pilot-sft.json \
  --checkpoint-dir checkpoints/first-verification \
  --sandbox-image "${NODELM_SANDBOX_IMAGE}" \
  --output artifacts/reports/infra/training-lifecycle.json

# Produce a machine-readable host report for a future GPU machine.
./scripts/remote_doctor.sh
```

`scripts/remote_verify.sh` gives each invocation a validated run ID and copies only evidence
bound to that invocation into `artifacts/reports/infra/runs/host-<host>/<run-id>/`. Without all
five remote lifecycle inputs it exits `BLOCKED`; with a pinned runtime config, matching pilot
manifest/sample artifact, fresh checkpoint path, and digest-pinned sandbox image, it installs
the locked training extra before the strict infrastructure gate. Full verification requires
rootless Podman and that exact image already loaded; the workflow never pulls an image. It then
runs the complete initial-step/checkpoint/resumed-step lifecycle and persists separate
infrastructure, lifecycle, and general-harness JSON reports.

## Security and evaluation semantics

Trusted-local repository-code execution is limited to checked-in fixtures. Model-authored code
runs only in a preloaded, digest-pinned rootless Podman image; implicit pulls are disabled. The
container sees the temporary host workspace through a read-only bind, has no network, runs with
bounded CPU/memory/process/file resources, and is force-removed in a `finally` cleanup path.

Model output is accepted only as a bounded, text-only Git patch. Protected repository-test paths
reject symlinks in any path component. The fixed smoke harness is bound to one exact
`SolveContext` and approved before/after source hashes; context or source-identity mismatches
fail. A successful generic repository test command remains `UNVERIFIED` until an
integrity-attested oracle can establish task resolution and regression-test integrity.

See [CODEX.md](CODEX.md) for future agent-development rules and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system boundaries.

## Specification status

`NodeLM_TypeScript_Node_Distillation_Plan.md` is now the tracked project ground truth. Its three
student candidates and primary teacher are pinned in strict metadata contracts. Metadata is
`PASS`; model execution, the 50–100-task candidate bake-off, student selection, full dataset
snapshot audit, and training remain `NOT RUN`. See [`docs/SPEC_STATUS.md`](docs/SPEC_STATUS.md)
for the active gates.
