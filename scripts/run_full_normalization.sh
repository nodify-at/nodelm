#!/usr/bin/env bash
set -Eeuo pipefail

# Run the seven receipt-bound Open-SWE-Traces partitions without downloading data.
# Every run is bound to an exact clean Git commit. Published artifacts are immutable;
# a rerun resumes only after validating an entire terminal artifact/manifest set.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${NODELM_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"
PERSIST_ROOT="${NODELM_PERSIST_ROOT:-/workspace/nodelm}"
EXEC_TMP_ROOT="${NODELM_EXEC_TMP_ROOT:-/tmp/nodelm-exec}"
TASK_DIR="${NODELM_TASK_PROVENANCE_DIR:-${PERSIST_ROOT}/derived/normalization-canary-20260825-7366ec0}"
UV_BIN="${NODELM_UV_BIN:-uv}"

EXEC_PARENT=""
EXEC_ROOT=""
EXPECTED_BINDING_FILE=""
RUN_BINDING_FILE=""
REGISTRY=""
PARTITION_CONTRACT=""
OPEN_SWE_RECEIPT="${PERSIST_ROOT}/receipts/open-swe-traces.transfer.json"
V2_RECEIPT="${PERSIST_ROOT}/receipts/swe-rebench-v2.transfer.json"
OPEN_SWE_SNAPSHOT="${PERSIST_ROOT}/snapshots/open-swe-traces"
V2_SNAPSHOT="${PERSIST_ROOT}/snapshots/swe-rebench-v2"
SAFE_TASKS="${TASK_DIR}/swe-rebench-v2.safe.jsonl"
SAFE_TASK_MANIFEST="${TASK_DIR}/swe-rebench-v2.safe.manifest.json"
SAFE_TASK_REJECTIONS="${TASK_DIR}/swe-rebench-v2.safe.rejections.jsonl"

readonly REGISTRY_SHA256="f92315a70a0c75ec909d83f4cb639b3a320f62526069f11ca87f0fe1d891637f"
readonly PARTITION_CONTRACT_SHA256="aec2ae095a926dda09a5fe3eefede7a59fbd494b24fffd503fff4cb366b389b5"
readonly OPEN_SWE_RECEIPT_SHA256="44ea157ebd802a5604301c82e8785003d67f90d0ed64efcc079059dfd4290a84"
readonly V2_RECEIPT_SHA256="fbcd4fbb2b9c4b887ef15f368f3673c07d82d4ba81d2b0d0eed7e3dd6d1fe254"
readonly SAFE_TASKS_SHA256="1e70b4d99cee7eea5dd40c4c36a553a53de3304caa7120ec45c00b5a2b6fdffd"
readonly SAFE_TASK_MANIFEST_SHA256="93f17e1f466fa0e014b29112c34d5f05830c17f39e296bcc89915f7b5567cfb5"
readonly SAFE_TASK_REJECTIONS_SHA256="473679bf93386cd6bdbea8019e7991104c355fd21b30886632669c2e099d7bf2"

# UV_OFFLINE is an explicit guardrail: this runner must never resolve or download packages.
export UV_OFFLINE=1
export UV_CACHE_DIR="${NODELM_UV_CACHE_DIR:-${PERSIST_ROOT}/cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/nodelm-venv}"
export UV_NO_SYNC=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CURRENT_STATE="STARTING"
CURRENT_PHASE="preflight"
CURRENT_LEAF="none"
LAST_DETAIL="initializing"
RUN_DIR=""
STATE_FILE=""
EVENTS_LOG=""
RUN_LOG=""

timestamp() {
  date -u +'%Y-%m-%dT%H:%M:%SZ'
}

write_state() {
  local state="$1"
  local detail="$2"
  local temporary

  CURRENT_STATE="${state}"
  LAST_DETAIL="${detail}"
  [[ -n "${STATE_FILE}" ]] || return 0
  temporary="$(mktemp "${RUN_DIR}/.run.state.XXXXXX")"
  {
    printf 'state=%s\n' "${state}"
    printf 'timestamp=%s\n' "$(timestamp)"
    printf 'pid=%s\n' "$$"
    printf 'commit=%s\n' "${HEAD_COMMIT}"
    printf 'leaf=%s\n' "${CURRENT_LEAF}"
    printf 'phase=%s\n' "${CURRENT_PHASE}"
    printf 'detail=%s\n' "${detail}"
  } >"${temporary}"
  mv -- "${temporary}" "${STATE_FILE}"
}

log_event() {
  local message="$1"
  local line
  line="$(timestamp) ${message}"
  printf '%s\n' "${line}"
  [[ -n "${EVENTS_LOG}" ]] && printf '%s\n' "${line}" >>"${EVENTS_LOG}"
}

fail() {
  local message="$1"
  log_event "ERROR ${message}"
  LAST_DETAIL="${message}"
  return 1
}

on_signal() {
  local signal="$1"
  log_event "STOPPED signal=${signal} leaf=${CURRENT_LEAF} phase=${CURRENT_PHASE}"
  write_state "STOPPED" "received signal ${signal}"
  exit 130
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  if [[ "${CURRENT_STATE}" == "RUNNING" || "${CURRENT_STATE}" == "STARTING" ]]; then
    write_state "FAILED" "${LAST_DETAIL}; exit=${exit_code}"
    log_event "FAILED exit=${exit_code} leaf=${CURRENT_LEAF} phase=${CURRENT_PHASE}"
  fi
  cleanup_execution_tree || true
  exit "${exit_code}"
}

trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap on_exit EXIT

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command is unavailable: $1"
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  [[ -f "${path}" && ! -L "${path}" ]] || fail \
    "required sealed input is missing, special, or a symlink: ${path}"
  actual="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || fail \
    "SHA-256 mismatch for ${path}: expected=${expected} actual=${actual}"
}

check_stop() {
  if [[ -e "${STOP_FILE}" ]]; then
    log_event "STOPPED sentinel=${STOP_FILE} leaf=${CURRENT_LEAF} phase=${CURRENT_PHASE}"
    write_state "STOPPED" "stop sentinel observed"
    exit 0
  fi
}

run_nodelm() {
  "${UV_BIN}" run --frozen --no-sync --directory "${EXEC_ROOT}" \
    python -m nodelm "$@"
}

require_code_unchanged() {
  [[ "$(git -C "${REPO_ROOT}" rev-parse --verify 'HEAD^{commit}')" == "${HEAD_COMMIT}" ]] ||
    fail "repository HEAD changed while the run was active"
  git -C "${REPO_ROOT}" diff --quiet || fail "tracked worktree changed while the run was active"
  git -C "${REPO_ROOT}" diff --cached --quiet ||
    fail "Git index changed while the run was active"
}

cleanup_execution_tree() {
  local expected_prefix

  [[ -n "${EXEC_PARENT}" && -n "${HEAD_COMMIT:-}" ]] || return 0
  expected_prefix="${EXEC_TMP_ROOT}/nodelm-exec-${HEAD_COMMIT}."
  [[ "${EXEC_PARENT}" == "${expected_prefix}"* ]] || return 1
  [[ -d "${EXEC_PARENT}" && ! -L "${EXEC_PARENT}" ]] || return 0
  chmod -R u+w "${EXEC_PARENT}" 2>/dev/null || true
  rm -rf -- "${EXEC_PARENT}"
}

materialize_execution_tree() {
  local archive

  [[ "${EXEC_TMP_ROOT}" == /* && "${EXEC_TMP_ROOT}" != "/" ]] || fail \
    "NODELM_EXEC_TMP_ROOT must be an absolute, non-root directory"
  mkdir -p -- "${EXEC_TMP_ROOT}"
  [[ -d "${EXEC_TMP_ROOT}" && ! -L "${EXEC_TMP_ROOT}" ]] || fail \
    "private execution root is missing, special, or a symlink: ${EXEC_TMP_ROOT}"
  EXEC_TMP_ROOT="$(cd -- "${EXEC_TMP_ROOT}" && pwd -P)"
  EXEC_PARENT="$(mktemp -d "${EXEC_TMP_ROOT}/nodelm-exec-${HEAD_COMMIT}.XXXXXX")"
  EXEC_ROOT="${EXEC_PARENT}/tree"
  EXPECTED_BINDING_FILE="${EXEC_PARENT}/expected-run.binding"
  archive="${EXEC_PARENT}/source.tar"
  mkdir -- "${EXEC_ROOT}"
  git -C "${REPO_ROOT}" archive --format=tar --output="${archive}" "${HEAD_COMMIT}"
  tar -xf "${archive}" -C "${EXEC_ROOT}"
  [[ -f "${EXEC_ROOT}/pyproject.toml" && ! -L "${EXEC_ROOT}/pyproject.toml" ]] || fail \
    "exact-commit execution tree is incomplete"
  [[ -f "${EXEC_ROOT}/scripts/run_full_normalization.sh" ]] || fail \
    "exact-commit execution tree does not contain this runner"
  [[ ! -e "${EXEC_ROOT}/.git" ]] || fail "private execution tree must not contain Git metadata"

  # Source code is imported only from this Git-object snapshot. uv is forbidden from
  # syncing, and Python is forbidden from writing bytecode into the sealed tree.
  export PYTHONPATH="${EXEC_ROOT}/src"
  find "${EXEC_ROOT}" -type d -exec chmod a-w {} +
  find "${EXEC_ROOT}" -type f -exec chmod a-w {} +
}

require_no_unbound_checkpoints() {
  local candidate

  for candidate in "${RUN_DIR}"/*; do
    [[ -e "${candidate}" || -L "${candidate}" ]] || continue
    case "${candidate##*/}" in
      *.raw.jsonl | *.raw.manifest.json | *.normalized.jsonl | \
        *.normalized.rejections.jsonl | *.normalized.manifest.json)
        fail "checkpoint evidence exists without a commit-bound run marker: ${candidate}"
        return 1
        ;;
    esac
  done
}

prepare_run_binding() {
  local temporary

  temporary="$(mktemp "${EXEC_PARENT}/.expected-run.binding.XXXXXX")"
  {
    printf 'format=nodelm-full-normalization-run-binding-v1\n'
    printf 'commit=%s\n' "${HEAD_COMMIT}"
    printf 'tree=%s\n' "${HEAD_TREE}"
    printf 'runner_blob=%s\n' "${RUNNER_BLOB}"
    printf 'registry_sha256=%s\n' "${REGISTRY_SHA256}"
    printf 'partition_contract_sha256=%s\n' "${PARTITION_CONTRACT_SHA256}"
  } >"${temporary}"
  chmod a-w "${temporary}"
  mv -- "${temporary}" "${EXPECTED_BINDING_FILE}"

  if [[ -e "${RUN_BINDING_FILE}" || -L "${RUN_BINDING_FILE}" ]]; then
    require_run_binding
    return
  fi
  require_no_unbound_checkpoints
  temporary="$(mktemp "${RUN_DIR}/.run.binding.XXXXXX")"
  cp -- "${EXPECTED_BINDING_FILE}" "${temporary}"
  chmod a-w "${temporary}"
  mv -- "${temporary}" "${RUN_BINDING_FILE}"
  require_run_binding
}

require_run_binding() {
  [[ -f "${RUN_BINDING_FILE}" && ! -L "${RUN_BINDING_FILE}" ]] || fail \
    "commit-bound run marker is missing, special, or a symlink"
  cmp -s -- "${EXPECTED_BINDING_FILE}" "${RUN_BINDING_FILE}" || fail \
    "commit-bound run marker does not match the exact execution tree"
}

# Parse manifests with the production strict Pydantic models, then stream-hash every
# referenced artifact. This is intentionally heavier than checking file existence: a
# terminal manifest is a resume checkpoint only when its complete evidence still binds.
validate_terminal() {
  "${UV_BIN}" run --frozen --no-sync --directory "${EXEC_ROOT}" python - "$@" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from nodelm.artifacts import file_identity as identity
from nodelm.provenance.manifests import (
    NormalizationManifestV2,
    SnapshotMaterializationManifestV2,
)


def bound_path(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    require(candidate.is_file(), f"bound artifact is missing or not a regular file: {candidate}")
    require(not candidate.is_symlink(), f"bound artifact must not be a symlink: {candidate}")
    return candidate.resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


mode = sys.argv[1]
manifest_argument = Path(sys.argv[2])
expected_output_argument = Path(sys.argv[3])
require(manifest_argument.is_file(), "terminal manifest is missing or not a regular file")
require(not manifest_argument.is_symlink(), "terminal manifest must not be a symlink")
require(expected_output_argument.is_file(), "expected output is missing or not a regular file")
require(not expected_output_argument.is_symlink(), "expected output must not be a symlink")
manifest_path = manifest_argument.resolve()
expected_output = expected_output_argument.resolve()
partition, harness, model = sys.argv[4:7]
registry_sha, contract_sha, receipt_sha = sys.argv[7:10]
contract_argument = Path(sys.argv[10])
receipt_argument = Path(sys.argv[11])
require(contract_argument.is_file() and not contract_argument.is_symlink(), "invalid partition contract file")
require(receipt_argument.is_file() and not receipt_argument.is_symlink(), "invalid transfer receipt file")
contract_identity = identity(contract_argument)
receipt_identity = identity(receipt_argument)
require(contract_identity[0] == contract_sha, "partition contract preflight digest changed")
require(receipt_identity[0] == receipt_sha, "transfer receipt preflight digest changed")
payload = manifest_path.read_bytes()

if mode == "materialization":
    manifest = SnapshotMaterializationManifestV2.model_validate_json(payload)
    require(manifest.source_name == "open-swe-traces", "unexpected materialization source")
    require(manifest.source_repository_id == "nvidia/Open-SWE-Traces", "unexpected source repository")
    require(manifest.source_revision == "ed95cef24df8d8bd79b4ceb0192cb420fde06521", "unexpected source revision")
    require(manifest.partition_name == partition, "materialization partition mismatch")
    require(manifest.harness == harness, "materialization harness mismatch")
    require(manifest.generating_model == model, "materialization model mismatch")
    require(manifest.upstream_source == "swe-rebench-v2", "unexpected upstream source")
    require(manifest.row_dataset_name == "nebius/SWE-rebench-V2", "unexpected row dataset")
    require(manifest.task_source_name == "swe-rebench-v2", "unexpected task source")
    require(manifest.task_source_revision == "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0", "unexpected task revision")
    require(manifest.normalization_status == "PASS", "partition is not normalization-eligible")
    require(manifest.materialization_scope == "complete-partition", "materialization is not complete-partition")
    require(manifest.max_rows is None, "full materialization must not have max_rows")
    require(manifest.registry_sha256 == registry_sha, "materialization registry digest mismatch")
    require(
        (manifest.partition_contract_sha256, manifest.partition_contract_bytes) == contract_identity,
        "materialization contract identity mismatch",
    )
    require(
        (manifest.transfer_receipt_sha256, manifest.transfer_receipt_bytes) == receipt_identity,
        "materialization receipt identity mismatch",
    )
    require(manifest.status == "PASS", "full materialization must terminate PASS")
    require(manifest.output == expected_output_argument.name, "materialization output reference is not canonical")
    artifact = bound_path(manifest_path, manifest.output)
    require(artifact == expected_output, "materialization output path mismatch")
    require(artifact.is_file(), "materialization output is missing")
    artifact_sha, artifact_bytes = identity(artifact)
    require(artifact_sha == manifest.output_sha256, "materialization output digest mismatch")
    require(artifact_bytes == manifest.output_bytes, "materialization output byte count mismatch")
    print(manifest.status)
elif mode == "normalization":
    evidence_paths = [Path(value) for value in sys.argv[12:18]]
    require(all(path.is_file() for path in evidence_paths), "normalization evidence includes a missing or special file")
    require(not any(path.is_symlink() for path in evidence_paths), "normalization evidence must not use symlinks")
    (
        expected_rejections,
        expected_input,
        materialization_manifest,
        task_path,
        task_manifest_path,
        task_receipt_path,
    ) = (path.resolve() for path in evidence_paths)
    task_sha, task_manifest_sha, task_receipt_sha = sys.argv[18:21]

    manifest = NormalizationManifestV2.model_validate_json(payload)
    require(manifest.source_name == "open-swe-traces", "unexpected normalization source")
    require(manifest.source_repository_id == "nvidia/Open-SWE-Traces", "unexpected source repository")
    require(manifest.source_revision == "ed95cef24df8d8bd79b4ceb0192cb420fde06521", "unexpected source revision")
    require(manifest.partition_name == partition, "normalization partition mismatch")
    require(manifest.harness == harness, "normalization harness mismatch")
    require(manifest.generating_model == model, "normalization model mismatch")
    require(manifest.upstream_source == "swe-rebench-v2", "unexpected upstream source")
    require(manifest.row_dataset_name == "nebius/SWE-rebench-V2", "unexpected row dataset")
    require(manifest.task_source_name == "swe-rebench-v2", "unexpected task source")
    require(manifest.task_source_revision == "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0", "unexpected task revision")
    require(manifest.uniqueness_scope == "complete-partition", "normalization is not complete-partition")
    require(manifest.registry_sha256 == registry_sha, "normalization registry digest mismatch")
    require(
        (manifest.partition_contract_sha256, manifest.partition_contract_bytes) == contract_identity,
        "normalization contract identity mismatch",
    )
    require(
        (manifest.transfer_receipt_sha256, manifest.transfer_receipt_bytes) == receipt_identity,
        "normalization receipt identity mismatch",
    )
    require(
        manifest.normalized_artifact == expected_output_argument.name,
        "normalized output reference is not canonical",
    )
    require(manifest.rejection_artifact == evidence_paths[0].name, "rejection output reference is not canonical")

    input_sha, input_bytes = identity(expected_input)
    materialization_sha, materialization_bytes = identity(materialization_manifest)
    require(
        (manifest.input_sha256, manifest.input_bytes) == (input_sha, input_bytes),
        "normalization input identity mismatch",
    )
    require(
        (manifest.materialization_manifest_sha256, manifest.materialization_manifest_bytes)
        == (materialization_sha, materialization_bytes),
        "normalization materialization-manifest identity mismatch",
    )
    task_identity = identity(task_path)
    task_manifest_identity = identity(task_manifest_path)
    task_receipt_identity = identity(task_receipt_path)
    require(
        task_identity == (task_sha, manifest.task_provenance_bytes)
        and manifest.task_provenance_sha256 == task_sha,
        "task projection identity mismatch",
    )
    require(
        task_manifest_identity == (task_manifest_sha, manifest.task_provenance_manifest_bytes)
        and manifest.task_provenance_manifest_sha256 == task_manifest_sha,
        "task projection manifest identity mismatch",
    )
    require(
        task_receipt_identity == (task_receipt_sha, manifest.task_transfer_receipt_bytes)
        and manifest.task_transfer_receipt_sha256 == task_receipt_sha,
        "task transfer receipt identity mismatch",
    )

    normalized = bound_path(manifest_path, manifest.normalized_artifact)
    rejections = bound_path(manifest_path, manifest.rejection_artifact)
    require(normalized == expected_output, "normalized output path mismatch")
    require(rejections == expected_rejections, "normalization rejection path mismatch")
    require(
        identity(normalized) == (manifest.normalized_sha256, manifest.normalized_bytes),
        "normalized artifact identity mismatch",
    )
    require(
        identity(rejections) == (manifest.rejection_sha256, manifest.rejection_bytes),
        "rejection artifact identity mismatch",
    )
    print(manifest.status)
else:
    raise SystemExit(f"unknown validation mode: {mode}")
PY
}

validate_materialization() {
  local manifest="$1"
  local output="$2"
  local partition="$3"
  local harness="$4"
  local model="$5"
  validate_terminal materialization "${manifest}" "${output}" "${partition}" "${harness}" \
    "${model}" "${REGISTRY_SHA256}" "${PARTITION_CONTRACT_SHA256}" \
    "${OPEN_SWE_RECEIPT_SHA256}" "${PARTITION_CONTRACT}" "${OPEN_SWE_RECEIPT}"
}

validate_normalization() {
  local manifest="$1"
  local output="$2"
  local rejections="$3"
  local input="$4"
  local materialization_manifest="$5"
  local partition="$6"
  local harness="$7"
  local model="$8"
  validate_terminal normalization "${manifest}" "${output}" "${partition}" "${harness}" \
    "${model}" "${REGISTRY_SHA256}" "${PARTITION_CONTRACT_SHA256}" \
    "${OPEN_SWE_RECEIPT_SHA256}" "${PARTITION_CONTRACT}" "${OPEN_SWE_RECEIPT}" \
    "${rejections}" "${input}" \
    "${materialization_manifest}" "${SAFE_TASKS}" "${SAFE_TASK_MANIFEST}" \
    "${V2_RECEIPT}" "${SAFE_TASKS_SHA256}" "${SAFE_TASK_MANIFEST_SHA256}" \
    "${V2_RECEIPT_SHA256}"
}

run_materialization() {
  local slug="$1"
  local partition="$2"
  local harness="$3"
  local model="$4"
  local raw_output="${RUN_DIR}/${slug}.raw.jsonl"
  local raw_manifest="${RUN_DIR}/${slug}.raw.manifest.json"
  local command_status manifest_status

  CURRENT_PHASE="materialize"
  write_state "RUNNING" "checking materialization checkpoint"
  require_code_unchanged
  require_run_binding
  check_stop

  if [[ -e "${raw_output}" || -e "${raw_manifest}" ]]; then
    [[ -f "${raw_output}" && -f "${raw_manifest}" ]] || fail \
      "conflicting partial materialization evidence for ${slug}; refusing to overwrite"
    if ! manifest_status="$(validate_materialization "${raw_manifest}" "${raw_output}" \
      "${partition}" "${harness}" "${model}")"; then
      fail "existing materialization checkpoint is invalid for ${slug}"
    fi
    require_run_binding
    log_event "RESUME leaf=${partition} phase=materialize status=${manifest_status}"
    return 0
  fi

  log_event "START leaf=${partition} phase=materialize"
  write_state "RUNNING" "materialization in progress"
  set +e
  run_nodelm datasets materialize \
    --source open-swe-traces \
    --snapshot "${OPEN_SWE_SNAPSHOT}" \
    --partition-contract "${PARTITION_CONTRACT}" \
    --transfer-receipt "${OPEN_SWE_RECEIPT}" \
    --partition "${partition}" \
    --output "${raw_output}" \
    --manifest-output "${raw_manifest}" \
    --config "${REGISTRY}" >>"${RUN_LOG}" 2>&1
  command_status=$?
  set -e
  require_code_unchanged
  require_run_binding

  [[ -f "${raw_output}" && -f "${raw_manifest}" ]] || fail \
    "materialization did not publish a complete terminal pair for ${slug}; exit=${command_status}"
  if ! manifest_status="$(validate_materialization "${raw_manifest}" "${raw_output}" \
    "${partition}" "${harness}" "${model}")"; then
    fail "published materialization checkpoint is invalid for ${slug}"
  fi
  if [[ "${manifest_status}" == "PASS" && ${command_status} -ne 0 ]]; then
    fail "materialization published PASS but exited ${command_status} for ${slug}"
  fi
  log_event "SEALED leaf=${partition} phase=materialize status=${manifest_status} exit=${command_status}"
}

run_normalization() {
  local slug="$1"
  local partition="$2"
  local harness="$3"
  local model="$4"
  local raw_output="${RUN_DIR}/${slug}.raw.jsonl"
  local raw_manifest="${RUN_DIR}/${slug}.raw.manifest.json"
  local normalized="${RUN_DIR}/${slug}.normalized.jsonl"
  local rejections="${RUN_DIR}/${slug}.normalized.rejections.jsonl"
  local manifest="${RUN_DIR}/${slug}.normalized.manifest.json"
  local present_count=0 command_status manifest_status

  CURRENT_PHASE="normalize"
  write_state "RUNNING" "checking normalization checkpoint"
  require_code_unchanged
  require_run_binding
  check_stop

  [[ -e "${normalized}" ]] && present_count=$((present_count + 1))
  [[ -e "${rejections}" ]] && present_count=$((present_count + 1))
  [[ -e "${manifest}" ]] && present_count=$((present_count + 1))
  if ((present_count > 0)); then
    ((present_count == 3)) || fail \
      "conflicting partial normalization evidence for ${slug}; refusing to overwrite"
    if ! manifest_status="$(validate_normalization "${manifest}" "${normalized}" \
      "${rejections}" "${raw_output}" "${raw_manifest}" "${partition}" "${harness}" \
      "${model}")"; then
      fail "existing normalization checkpoint is invalid for ${slug}"
    fi
    require_run_binding
    log_event "RESUME leaf=${partition} phase=normalize status=${manifest_status}"
    return 0
  fi

  log_event "START leaf=${partition} phase=normalize"
  write_state "RUNNING" "normalization in progress"
  set +e
  run_nodelm datasets normalize \
    --source open-swe-traces \
    --snapshot "${OPEN_SWE_SNAPSHOT}" \
    --input "${raw_output}" \
    --materialization-manifest "${raw_manifest}" \
    --partition-contract "${PARTITION_CONTRACT}" \
    --transfer-receipt "${OPEN_SWE_RECEIPT}" \
    --task-provenance "${SAFE_TASKS}" \
    --task-provenance-manifest "${SAFE_TASK_MANIFEST}" \
    --task-transfer-receipt "${V2_RECEIPT}" \
    --task-snapshot "${V2_SNAPSHOT}" \
    --expect-harness "${harness}" \
    --expect-generating-model "${model}" \
    --output "${normalized}" \
    --rejections-output "${rejections}" \
    --manifest-output "${manifest}" \
    --config "${REGISTRY}" >>"${RUN_LOG}" 2>&1
  command_status=$?
  set -e
  require_code_unchanged
  require_run_binding

  [[ -f "${normalized}" && -f "${rejections}" && -f "${manifest}" ]] || fail \
    "normalization did not publish a complete terminal set for ${slug}; exit=${command_status}"
  if ! manifest_status="$(validate_normalization "${manifest}" "${normalized}" \
    "${rejections}" "${raw_output}" "${raw_manifest}" "${partition}" "${harness}" \
    "${model}")"; then
    fail "published normalization checkpoint is invalid for ${slug}"
  fi
  if [[ "${manifest_status}" == "PASS" && ${command_status} -ne 0 ]]; then
    fail "normalization published PASS but exited ${command_status} for ${slug}"
  fi
  if [[ "${manifest_status}" == "FAIL" && ${command_status} -ne 1 ]]; then
    fail "normalization published FAIL with unexpected exit ${command_status} for ${slug}"
  fi
  log_event "SEALED leaf=${partition} phase=normalize status=${manifest_status} exit=${command_status}"
}

require_command git
require_command flock
require_command sha256sum
require_command awk
require_command tar
require_command find
require_command chmod
require_command rm
require_command cmp
require_command cp
require_command "${UV_BIN}"
[[ -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]] || fail \
  "pre-provisioned offline Python environment is missing: ${UV_PROJECT_ENVIRONMENT}"

[[ -d "${REPO_ROOT}/.git" ]] || fail "repository is not a Git worktree: ${REPO_ROOT}"
HEAD_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --verify 'HEAD^{commit}')"
[[ "${HEAD_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "HEAD is not an exact 40-character commit"
HEAD_TREE="$(git -C "${REPO_ROOT}" rev-parse --verify "${HEAD_COMMIT}^{tree}")"
RUNNER_BLOB="$(git -C "${REPO_ROOT}" rev-parse --verify \
  "${HEAD_COMMIT}:scripts/run_full_normalization.sh")"
[[ "${HEAD_TREE}" =~ ^[0-9a-f]{40}$ ]] || fail "HEAD tree is not an exact Git object"
[[ "${RUNNER_BLOB}" =~ ^[0-9a-f]{40}$ ]] || fail "runner is not present in the exact commit"
require_code_unchanged
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail \
  "untracked files are present; the production tree must be completely clean"

RUN_DIR="${NODELM_RUN_DIR:-${PERSIST_ROOT}/derived/full-normalization-${HEAD_COMMIT}}"
EXPECTED_RUN_DIR="${PERSIST_ROOT}/derived/full-normalization-${HEAD_COMMIT}"
[[ "${RUN_DIR}" == "${EXPECTED_RUN_DIR}" ]] || fail \
  "NODELM_RUN_DIR must equal the commit-bound path ${EXPECTED_RUN_DIR}"
mkdir -p -- "${RUN_DIR}" "${PERSIST_ROOT}/locks"
LOCK_FILE="${PERSIST_ROOT}/locks/full-normalization.lock"

exec 9>"${LOCK_FILE}"
flock -n 9 || fail "another runner already owns ${LOCK_FILE}"

STATE_FILE="${RUN_DIR}/run.state"
EVENTS_LOG="${RUN_DIR}/events.log"
RUN_LOG="${RUN_DIR}/runner.log"
STOP_FILE="${RUN_DIR}/STOP"
RUN_BINDING_FILE="${RUN_DIR}/run.binding"
materialize_execution_tree
REGISTRY="${EXEC_ROOT}/configs/datasets/registry.yaml"
PARTITION_CONTRACT="${EXEC_ROOT}/configs/datasets/open-swe-trace-partitions.yaml"
prepare_run_binding
PID_TEMP="$(mktemp "${RUN_DIR}/.runner.pid.XXXXXX")"
printf '%s\n' "$$" >"${PID_TEMP}"
mv -- "${PID_TEMP}" "${RUN_DIR}/runner.pid"
CURRENT_STATE="RUNNING"
CURRENT_PHASE="input-digests"
write_state "RUNNING" "sealed input preflight in progress"
log_event "RUNNING commit=${HEAD_COMMIT} root=${PERSIST_ROOT} output=${RUN_DIR}"
require_sha256 "${REGISTRY}" "${REGISTRY_SHA256}"
require_sha256 "${PARTITION_CONTRACT}" "${PARTITION_CONTRACT_SHA256}"
require_sha256 "${OPEN_SWE_RECEIPT}" "${OPEN_SWE_RECEIPT_SHA256}"
require_sha256 "${V2_RECEIPT}" "${V2_RECEIPT_SHA256}"
require_sha256 "${SAFE_TASKS}" "${SAFE_TASKS_SHA256}"
require_sha256 "${SAFE_TASK_MANIFEST}" "${SAFE_TASK_MANIFEST_SHA256}"
require_sha256 "${SAFE_TASK_REJECTIONS}" "${SAFE_TASK_REJECTIONS_SHA256}"
[[ -d "${OPEN_SWE_SNAPSHOT}" ]] || fail "Open-SWE snapshot is missing: ${OPEN_SWE_SNAPSHOT}"
[[ -d "${V2_SNAPSHOT}" ]] || fail "SWE-rebench V2 snapshot is missing: ${V2_SNAPSHOT}"

CURRENT_PHASE="preflight-complete"
write_state "RUNNING" "sealed inputs verified"
check_stop

# slug|partition|harness|generating model -- order is intentionally fixed.
LEAVES=(
  'openhands-minimax-v2|openhands/minimax_m25/swe-rebench-v2|openhands|source-label:minimax_m25'
  'sweagent-minimax-v2|sweagent/minimax_m25/swe-rebench-v2|sweagent|source-label:minimax_m25'
  'openhands-qwen35-v2|openhands/qwen35_122b/swe-rebench-v2|openhands|source-label:qwen35_122b'
  'sweagent-qwen35-v2|sweagent/qwen35_122b/swe-rebench-v2|sweagent|source-label:qwen35_122b'
  'minisweagent-qwen36-v2|minisweagent/qwen36_27b/swe-rebench-v2|minisweagent|source-label:qwen36_27b'
  'sweagent-qwen36-v2|sweagent/qwen36_27b/swe-rebench-v2|sweagent|source-label:qwen36_27b'
  'openhands-qwen36-v2|openhands/qwen36_27b/swe-rebench-v2|openhands|source-label:qwen36_27b'
)

for leaf in "${LEAVES[@]}"; do
  IFS='|' read -r slug partition harness model <<<"${leaf}"
  CURRENT_LEAF="${partition}"
  check_stop
  run_materialization "${slug}" "${partition}" "${harness}" "${model}"
  check_stop
  run_normalization "${slug}" "${partition}" "${harness}" "${model}"
done

CURRENT_LEAF="none"
CURRENT_PHASE="final-code-check"
require_code_unchanged
require_run_binding
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail \
  "untracked files appeared while the run was active"
CURRENT_PHASE="complete"
write_state "COMPLETE" "all seven leaves have validated terminal evidence"
log_event "COMPLETE commit=${HEAD_COMMIT} leaves=7"
