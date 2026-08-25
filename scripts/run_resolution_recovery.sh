#!/usr/bin/env bash
set -Eeuo pipefail

# Build provenance-bound Qwen3.6 resolution sidecars from the persistent trace snapshot.
# This runner only derives exact-transfer candidates and a blocked evaluation queue. It
# never starts a repository evaluator, container workload, or GPU process.
# Inspect run.state for the PID and stop_file. STOP is honored before artifact generation;
# once phase=build is active, send TERM to the recorded PID and the runner stops its child.

REPO_ROOT="${NODELM_REPO_ROOT:-/workspace/nodelm/repo}"
PERSIST_ROOT="${NODELM_PERSIST_ROOT:-/workspace/nodelm}"
EXEC_TMP_ROOT="${NODELM_EXEC_TMP_ROOT:-/tmp/nodelm-resolution-recovery-exec}"
UV_BIN="${NODELM_UV_BIN:-uv}"
TRACE_SNAPSHOT="${NODELM_TRACE_SNAPSHOT:-${PERSIST_ROOT}/snapshots/open-swe-traces}"
TRACE_RECEIPT="${NODELM_TRACE_RECEIPT:-${PERSIST_ROOT}/receipts/open-swe-traces.transfer.json}"
REGISTRY_OVERRIDE="${NODELM_REGISTRY_PATH:-}"
PARTITION_CONTRACT_OVERRIDE="${NODELM_PARTITION_CONTRACT_PATH:-}"
REGISTRY=""
PARTITION_CONTRACT=""

export UV_OFFLINE=1
export UV_CACHE_DIR="${NODELM_UV_CACHE_DIR:-${PERSIST_ROOT}/cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/nodelm-venv}"
export UV_NO_SYNC=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

readonly LABELED_PARTITIONS=(
  'openhands/minimax_m25/swe-rebench-v2'
  'openhands/qwen35_122b/swe-rebench-v2'
  'sweagent/minimax_m25/swe-rebench-v2'
  'sweagent/qwen35_122b/swe-rebench-v2'
)
readonly TARGET_PARTITIONS=(
  'minisweagent/qwen36_27b/swe-rebench-v2'
  'openhands/qwen36_27b/swe-rebench-v2'
  'sweagent/qwen36_27b/swe-rebench-v2'
)

CURRENT_STATE="STARTING"
CURRENT_PHASE="preflight"
LAST_DETAIL="initializing"
HEAD_COMMIT="unknown"
RUN_DIR=""
STATE_FILE=""
EVENTS_LOG=""
RUN_LOG=""
STOP_FILE=""
ACTIVE_PID=""
ACTIVE_PGID=""
EXEC_PARENT=""
EXEC_ROOT=""
PRESENT_COUNT=0

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
    printf 'phase=%s\n' "${CURRENT_PHASE}"
    printf 'detail=%s\n' "${detail}"
    if [[ -n "${STOP_FILE}" ]]; then
      printf 'stop_file=%s\n' "${STOP_FILE}"
      printf 'operator_stop=touch stop_file before build; kill -TERM pid during build\n'
    fi
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

  LAST_DETAIL="${message}"
  log_event "ERROR ${message}"
  return 1
}

stop_active_process_group() {
  local attempt

  if [[ "${ACTIVE_PGID}" =~ ^[1-9][0-9]*$ ]] &&
    kill -0 -- "-${ACTIVE_PGID}" 2>/dev/null; then
    kill -TERM -- "-${ACTIVE_PGID}" 2>/dev/null || true
    for ((attempt = 0; attempt < 50; attempt++)); do
      kill -0 -- "-${ACTIVE_PGID}" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 -- "-${ACTIVE_PGID}" 2>/dev/null; then
      kill -KILL -- "-${ACTIVE_PGID}" 2>/dev/null || true
    fi
  fi
  if [[ "${ACTIVE_PID}" =~ ^[1-9][0-9]*$ ]]; then
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  ACTIVE_PID=""
  ACTIVE_PGID=""
}

on_signal() {
  local signal="$1"

  trap '' TERM INT
  stop_active_process_group
  CURRENT_PHASE="signal"
  write_state "STOPPED" "received signal ${signal}; active child stopped"
  log_event "STOPPED signal=${signal}"
  exit 130
}

on_exit() {
  local exit_code=$?

  trap - EXIT
  if [[ "${CURRENT_STATE}" == "RUNNING" || "${CURRENT_STATE}" == "STARTING" ]]; then
    write_state "FAILED" "${LAST_DETAIL}; exit=${exit_code}"
    log_event "FAILED exit=${exit_code} phase=${CURRENT_PHASE}"
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

require_regular_file() {
  local path="$1"
  [[ -f "${path}" && ! -L "${path}" ]] || fail \
    "required input is missing, special, or a symlink: ${path}"
}

cleanup_execution_tree() {
  local expected_prefix

  [[ -n "${EXEC_PARENT}" && -n "${HEAD_COMMIT:-}" ]] || return 0
  expected_prefix="${EXEC_TMP_ROOT}/nodelm-resolution-recovery-${HEAD_COMMIT}."
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
  EXEC_PARENT="$(mktemp -d \
    "${EXEC_TMP_ROOT}/nodelm-resolution-recovery-${HEAD_COMMIT}.XXXXXX")"
  EXEC_ROOT="${EXEC_PARENT}/tree"
  archive="${EXEC_PARENT}/source.tar"
  mkdir -- "${EXEC_ROOT}"
  git -C "${REPO_ROOT}" archive --format=tar --output="${archive}" "${HEAD_COMMIT}"
  tar -xf "${archive}" -C "${EXEC_ROOT}"
  rm -- "${archive}"
  require_regular_file "${EXEC_ROOT}/pyproject.toml"
  require_regular_file "${EXEC_ROOT}/uv.lock"
  require_regular_file "${EXEC_ROOT}/scripts/run_resolution_recovery.sh"
  REGISTRY="${REGISTRY_OVERRIDE:-${EXEC_ROOT}/configs/datasets/registry.yaml}"
  PARTITION_CONTRACT="${PARTITION_CONTRACT_OVERRIDE:-${EXEC_ROOT}/configs/datasets/open-swe-trace-partitions.yaml}"
  export PYTHONPATH="${EXEC_ROOT}/src"
}

file_sha256() {
  sha256sum -- "$1" | awk '{print $1}'
}

require_code_unchanged() {
  [[ "$(git -C "${REPO_ROOT}" rev-parse --verify 'HEAD^{commit}')" == "${HEAD_COMMIT}" ]] ||
    fail "repository HEAD changed while the recovery run was active"
  [[ -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ]] ||
    fail "production tree changed while the recovery run was active"
}

require_inputs_unchanged() {
  [[ "$(file_sha256 "${REGISTRY}")" == "${REGISTRY_SHA256}" ]] ||
    fail "dataset registry changed while the recovery run was active"
  [[ "$(file_sha256 "${PARTITION_CONTRACT}")" == "${PARTITION_CONTRACT_SHA256}" ]] ||
    fail "partition contract changed while the recovery run was active"
  [[ "$(file_sha256 "${TRACE_RECEIPT}")" == "${TRACE_RECEIPT_SHA256}" ]] ||
    fail "trace transfer receipt changed while the recovery run was active"
  [[ "$(cd -- "${TRACE_SNAPSHOT}" && pwd -P)" == "${TRACE_SNAPSHOT_REAL}" ]] ||
    fail "trace snapshot path changed while the recovery run was active"
}

run_python() {
  "${UV_BIN}" run --frozen --no-sync --directory "${EXEC_ROOT}" python "$@"
}

count_artifacts() {
  local path

  PRESENT_COUNT=0
  for path in "${CANDIDATES}" "${QUEUE}" "${MANIFEST}"; do
    if [[ -e "${path}" || -L "${path}" ]]; then
      [[ -f "${path}" && ! -L "${path}" ]] || fail \
        "terminal artifact is special or a symlink: ${path}"
      PRESENT_COUNT=$((PRESENT_COUNT + 1))
    fi
  done
}

check_stop() {
  if [[ -e "${STOP_FILE}" || -L "${STOP_FILE}" ]]; then
    [[ -f "${STOP_FILE}" && ! -L "${STOP_FILE}" ]] || fail \
      "stop sentinel is special or a symlink: ${STOP_FILE}"
    CURRENT_PHASE="stopped"
    write_state "STOPPED" \
      "stop sentinel observed before artifact generation; terminal_artifacts_present=${PRESENT_COUNT}"
    log_event \
      "STOPPED sentinel=${STOP_FILE} terminal_artifacts_present=${PRESENT_COUNT}"
    exit 0
  fi
}

validate_terminal() {
  run_python - "${MANIFEST}" "${CANDIDATES}" "${QUEUE}" "${REGISTRY}" \
    "${PARTITION_CONTRACT}" "${TRACE_RECEIPT}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

from nodelm.artifacts import content_digest, file_identity
from nodelm.datasets.partitions import TracePartitionContract
from nodelm.provenance.manifests import ResolutionRecoveryManifestV1
from nodelm.provenance.resolution import (
    ExactResolutionCandidate,
    ResolutionEvaluationRequest,
)

LABELED_PARTITIONS = (
    "openhands/minimax_m25/swe-rebench-v2",
    "openhands/qwen35_122b/swe-rebench-v2",
    "sweagent/minimax_m25/swe-rebench-v2",
    "sweagent/qwen35_122b/swe-rebench-v2",
)
TARGET_PARTITIONS = (
    "minisweagent/qwen36_27b/swe-rebench-v2",
    "openhands/qwen36_27b/swe-rebench-v2",
    "sweagent/qwen36_27b/swe-rebench-v2",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def regular(path: Path, label: str) -> Path:
    require(path.is_file(), f"{label} is missing or not a regular file")
    require(not path.is_symlink(), f"{label} must not be a symlink")
    return path.resolve()


manifest_argument, candidates_argument, queue_argument, registry_argument, contract_argument, receipt_argument = (
    Path(value) for value in sys.argv[1:7]
)
manifest_path = regular(manifest_argument, "resolution recovery manifest")
candidates_path = regular(candidates_argument, "resolution candidate artifact")
queue_path = regular(queue_argument, "resolution queue artifact")
registry_path = regular(registry_argument, "dataset registry")
contract_path = regular(contract_argument, "partition contract")
receipt_path = regular(receipt_argument, "trace transfer receipt")

contract_payload = contract_path.read_bytes()
receipt_payload = receipt_path.read_bytes()
contract = TracePartitionContract.from_bytes(contract_payload)
contract.require_authorized_digest(content_digest(contract_payload))
receipt = contract.bind_transfer_receipt(receipt_payload)
registry_sha256, registry_bytes = file_identity(registry_path)
require(registry_sha256 == contract.sealed_registry_sha256, "registry digest does not match contract")
require(
    (receipt.registry_sha256, receipt.registry_bytes) == (registry_sha256, registry_bytes),
    "receipt registry identity does not match the exact registry",
)

manifest = ResolutionRecoveryManifestV1.model_validate_json(manifest_path.read_bytes())
require(contract.source_name == "open-swe-traces", "unexpected contract source name")
require(
    contract.source_repository_id == "nvidia/Open-SWE-Traces",
    "unexpected contract source repository",
)
require(
    contract.source_revision == "ed95cef24df8d8bd79b4ceb0192cb420fde06521",
    "unexpected contract source revision",
)
require(manifest.source_name == "open-swe-traces", "manifest source name mismatch")
require(
    manifest.source_repository_id == "nvidia/Open-SWE-Traces",
    "manifest source repository mismatch",
)
require(
    manifest.source_revision == "ed95cef24df8d8bd79b4ceb0192cb420fde06521",
    "trace revision mismatch",
)
require(manifest.task_source_name == "swe-rebench-v2", "unexpected task source")
require(
    manifest.task_source_revision == "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0",
    "unexpected task source revision",
)
require(
    (manifest.partition_contract_sha256, manifest.partition_contract_bytes)
    == file_identity(contract_path),
    "manifest partition contract identity mismatch",
)
require(
    (manifest.transfer_receipt_sha256, manifest.transfer_receipt_bytes)
    == file_identity(receipt_path),
    "manifest transfer receipt identity mismatch",
)
require(manifest.language_filter == ("JavaScript", "TypeScript"), "language filter mismatch")
require(manifest.derivation_status == "PASS", "recovery derivation is not PASS")
require(manifest.admission_status == "BLOCKED", "recovery admission is not BLOCKED")
require(
    manifest.admission_blocker == "harness_canary_pending",
    "unexpected recovery admission blocker",
)
require(
    tuple(item.partition_name for item in manifest.labeled_partitions) == LABELED_PARTITIONS,
    "labeled partition set mismatch",
)
require(
    tuple(item.partition_name for item in manifest.target_partitions) == TARGET_PARTITIONS,
    "target partition set mismatch",
)

receipt_files = {identity.path: identity for identity in receipt.snapshot.files}
for partition_input in (*manifest.labeled_partitions, *manifest.target_partitions):
    partition = contract.by_name(partition_input.partition_name)
    expected = tuple(
        sorted(
            (
                (identity.path, identity.sha256, identity.bytes)
                for identity in receipt.snapshot.files
                if any(
                    PurePosixPath(identity.path).match(pattern)
                    for pattern in partition.file_patterns
                )
            ),
            key=lambda item: item[0],
        )
    )
    observed = tuple(
        (identity.path, identity.sha256, identity.bytes) for identity in partition_input.files
    )
    require(observed == expected, f"receipt-bound files mismatch for {partition.name}")
    require(all(path in receipt_files for path, _, _ in observed), "unknown receipt file")

require(manifest.candidate_artifact == candidates_path.name, "candidate path is not canonical")
require(manifest.queue_artifact == queue_path.name, "queue path is not canonical")
require(manifest_path.parent / manifest.candidate_artifact == candidates_path, "candidate path mismatch")
require(manifest_path.parent / manifest.queue_artifact == queue_path, "queue path mismatch")
require(
    file_identity(candidates_path) == (manifest.candidate_sha256, manifest.candidate_bytes),
    "candidate artifact identity mismatch",
)
require(
    file_identity(queue_path) == (manifest.queue_sha256, manifest.queue_bytes),
    "queue artifact identity mismatch",
)

candidate_count = 0
candidate_keys: set[str] = set()
candidate_ids: set[str] = set()
resolved_count = 0
unresolved_count = 0
with candidates_path.open("rb") as stream:
    for raw_line in stream:
        require(bool(raw_line.strip()), "candidate artifact contains a blank line")
        candidate = ExactResolutionCandidate.model_validate_json(raw_line)
        candidate_count += 1
        candidate_keys.add(candidate.resolution_key)
        candidate_ids.add(candidate.candidate_id)
        resolved_count += int(candidate.resolved)
        unresolved_count += int(not candidate.resolved)
        require(
            candidate.trace_source_revision.casefold() == manifest.source_revision.casefold(),
            "candidate trace revision mismatch",
        )
        require(
            candidate.task_source_revision.casefold() == manifest.task_source_revision.casefold(),
            "candidate task revision mismatch",
        )
        require(
            candidate.target_reference.partition_name in TARGET_PARTITIONS,
            "candidate target reference is outside the authorized targets",
        )
        require(
            all(item.partition_name in LABELED_PARTITIONS for item in candidate.label_evidence),
            "candidate label evidence is outside the authorized labeled leaves",
        )

queue_count = 0
queue_keys: set[str] = set()
request_ids: set[str] = set()
queued_fanout = 0
with queue_path.open("rb") as stream:
    for raw_line in stream:
        require(bool(raw_line.strip()), "queue artifact contains a blank line")
        request = ResolutionEvaluationRequest.model_validate_json(raw_line)
        queue_count += 1
        queue_keys.add(request.resolution_key)
        request_ids.add(request.request_id)
        queued_fanout += len(request.target_references)
        require(
            request.trace_source_revision.casefold() == manifest.source_revision.casefold(),
            "queue trace revision mismatch",
        )
        require(
            request.task_source_revision.casefold() == manifest.task_source_revision.casefold(),
            "queue task revision mismatch",
        )
        require(
            all(item.partition_name in TARGET_PARTITIONS for item in request.target_references),
            "queue target reference is outside the authorized target leaves",
        )

require(candidate_count == manifest.candidate_row_count, "candidate row count mismatch")
require(len(candidate_keys) == manifest.candidate_unique_count, "candidate unique count mismatch")
require(len(candidate_ids) == candidate_count, "candidate identifiers are not unique")
require(resolved_count == manifest.candidate_resolved_count, "resolved candidate count mismatch")
require(unresolved_count == manifest.candidate_unresolved_count, "unresolved candidate count mismatch")
require(queue_count == manifest.queue_unique_count, "queue row count mismatch")
require(len(queue_keys) == queue_count, "queue resolution keys are not unique")
require(len(request_ids) == queue_count, "queue request identifiers are not unique")
require(queued_fanout == manifest.queued_fanout_row_count, "queue fanout count mismatch")
require(candidate_keys.isdisjoint(queue_keys), "candidate and queue resolution keys overlap")
print("PASS")
PY
}

require_command git
require_command flock
require_command sha256sum
require_command awk
require_command mktemp
require_command mv
require_command chmod
require_command rm
require_command sleep
require_command tar
require_command setsid
require_command "${UV_BIN}"

[[ "${PERSIST_ROOT}" == /* && "${PERSIST_ROOT}" != "/" ]] || fail \
  "NODELM_PERSIST_ROOT must be an absolute, non-root directory"
mkdir -p -- "${PERSIST_ROOT}"
[[ -d "${PERSIST_ROOT}" && ! -L "${PERSIST_ROOT}" ]] || fail \
  "persistent root is missing, special, or a symlink: ${PERSIST_ROOT}"
PERSIST_ROOT="$(cd -- "${PERSIST_ROOT}" && pwd -P)"
[[ -d "${REPO_ROOT}" && ! -L "${REPO_ROOT}" ]] || fail \
  "repository is missing or a symlink: ${REPO_ROOT}"
REPO_ROOT="$(cd -- "${REPO_ROOT}" && pwd -P)"
[[ "$(git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree)" == "true" ]] || fail \
  "repository is not a Git worktree: ${REPO_ROOT}"
HEAD_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --verify 'HEAD^{commit}')"
[[ "${HEAD_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "HEAD is not an exact 40-character commit"
HEAD_TREE="$(git -C "${REPO_ROOT}" rev-parse --verify "${HEAD_COMMIT}^{tree}")"
RUNNER_BLOB="$(git -C "${REPO_ROOT}" rev-parse --verify \
  "${HEAD_COMMIT}:scripts/run_resolution_recovery.sh")"
[[ "${HEAD_TREE}" =~ ^[0-9a-f]{40}$ ]] || fail "HEAD tree is not an exact Git object"
[[ "${RUNNER_BLOB}" =~ ^[0-9a-f]{40}$ ]] || fail "runner is not present in the exact commit"
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail \
  "tracked or untracked changes are present; the production tree must be completely clean"
require_code_unchanged

EXPECTED_RUN_DIR="${PERSIST_ROOT}/derived/resolution-recovery-${HEAD_COMMIT}"
RUN_DIR="${NODELM_RUN_DIR:-${EXPECTED_RUN_DIR}}"
[[ "${RUN_DIR}" == "${EXPECTED_RUN_DIR}" ]] || fail \
  "NODELM_RUN_DIR must equal the commit-bound path ${EXPECTED_RUN_DIR}"
mkdir -p -- "${RUN_DIR}" "${PERSIST_ROOT}/locks"
[[ -d "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] || fail \
  "run directory is missing, special, or a symlink: ${RUN_DIR}"
[[ -d "${PERSIST_ROOT}/locks" && ! -L "${PERSIST_ROOT}/locks" ]] || fail \
  "lock directory is missing, special, or a symlink"

EXPECTED_LOCK_FILE="${PERSIST_ROOT}/locks/resolution-recovery.lock"
LOCK_FILE="${NODELM_LOCK_FILE:-${EXPECTED_LOCK_FILE}}"
[[ "${LOCK_FILE}" == "${EXPECTED_LOCK_FILE}" ]] || fail \
  "NODELM_LOCK_FILE must equal the canonical single-run lock ${EXPECTED_LOCK_FILE}"
[[ ! -L "${LOCK_FILE}" ]] || fail "lock file must not be a symlink: ${LOCK_FILE}"
[[ ! -e "${LOCK_FILE}" || -f "${LOCK_FILE}" ]] || fail \
  "lock path is not a regular file: ${LOCK_FILE}"
exec 9>"${LOCK_FILE}"
flock -n 9 || fail "another recovery runner owns ${LOCK_FILE}"

STATE_FILE="${RUN_DIR}/run.state"
EVENTS_LOG="${RUN_DIR}/events.log"
RUN_LOG="${RUN_DIR}/runner.log"
PID_FILE="${RUN_DIR}/runner.pid"
RUN_BINDING_FILE="${RUN_DIR}/run.binding"
STOP_FILE="${RUN_DIR}/STOP"
CANDIDATES="${RUN_DIR}/exact-resolution-candidates.jsonl"
QUEUE="${RUN_DIR}/resolution-evaluation-queue.jsonl"
MANIFEST="${RUN_DIR}/resolution-recovery.manifest.json"
for managed_path in "${STATE_FILE}" "${EVENTS_LOG}" "${RUN_LOG}" "${PID_FILE}" \
  "${RUN_BINDING_FILE}" "${STOP_FILE}" "${CANDIDATES}" "${QUEUE}" "${MANIFEST}"; do
  [[ ! -L "${managed_path}" ]] || fail "managed run path must not be a symlink: ${managed_path}"
  [[ ! -e "${managed_path}" || -f "${managed_path}" ]] || fail \
    "managed run path is not a regular file: ${managed_path}"
done
for log_path in "${EVENTS_LOG}" "${RUN_LOG}"; do
  [[ ! -e "${log_path}" || -f "${log_path}" ]] || fail \
    "managed log path is not a regular file: ${log_path}"
  touch -- "${log_path}"
done

CURRENT_STATE="RUNNING"
CURRENT_PHASE="preflight"
write_state "RUNNING" "validating exact code and sealed inputs"
PID_TEMP="$(mktemp "${RUN_DIR}/.runner.pid.XXXXXX")"
printf '%s\n' "$$" >"${PID_TEMP}"
mv -- "${PID_TEMP}" "${PID_FILE}"
log_event "RUNNING commit=${HEAD_COMMIT} output=${RUN_DIR}"

materialize_execution_tree
require_regular_file "${REGISTRY}"
require_regular_file "${PARTITION_CONTRACT}"
require_regular_file "${TRACE_RECEIPT}"
[[ -d "${TRACE_SNAPSHOT}" && ! -L "${TRACE_SNAPSHOT}" ]] || fail \
  "trace snapshot is missing, special, or a symlink: ${TRACE_SNAPSHOT}"
TRACE_SNAPSHOT_REAL="$(cd -- "${TRACE_SNAPSHOT}" && pwd -P)"
REGISTRY_SHA256="$(file_sha256 "${REGISTRY}")"
PARTITION_CONTRACT_SHA256="$(file_sha256 "${PARTITION_CONTRACT}")"
TRACE_RECEIPT_SHA256="$(file_sha256 "${TRACE_RECEIPT}")"

EXPECTED_BINDING="$({
  printf 'format=nodelm-resolution-recovery-run-binding-v1\n'
  printf 'commit=%s\n' "${HEAD_COMMIT}"
  printf 'tree=%s\n' "${HEAD_TREE}"
  printf 'runner_blob=%s\n' "${RUNNER_BLOB}"
  printf 'registry_sha256=%s\n' "${REGISTRY_SHA256}"
  printf 'partition_contract_sha256=%s\n' "${PARTITION_CONTRACT_SHA256}"
  printf 'transfer_receipt_sha256=%s\n' "${TRACE_RECEIPT_SHA256}"
})"

if [[ ! -e "${RUN_BINDING_FILE}" ]]; then
  count_artifacts
  if [[ "${PRESENT_COUNT}" != "0" ]]; then
    fail "terminal artifacts exist without a commit-bound run binding"
  fi
  BINDING_TEMP="$(mktemp "${RUN_DIR}/.run.binding.XXXXXX")"
  printf '%s\n' "${EXPECTED_BINDING}" >"${BINDING_TEMP}"
  chmod a-w "${BINDING_TEMP}"
  mv -- "${BINDING_TEMP}" "${RUN_BINDING_FILE}"
fi
require_regular_file "${RUN_BINDING_FILE}"
[[ "$(cat -- "${RUN_BINDING_FILE}")" == "${EXPECTED_BINDING}" ]] || fail \
  "run binding does not match the exact code and sealed inputs"

CURRENT_PHASE="terminal-check"
write_state "RUNNING" "checking immutable terminal artifact set"
count_artifacts
if [[ "${PRESENT_COUNT}" == "3" ]]; then
  require_code_unchanged
  require_inputs_unchanged
  if ! validate_terminal >>"${RUN_LOG}" 2>&1; then
    fail "terminal recovery artifacts are invalid"
  fi
  require_code_unchanged
  require_inputs_unchanged
  CURRENT_PHASE="complete"
  write_state "COMPLETE" "validated existing terminal recovery artifacts"
  log_event "RESUME validated terminal artifacts commit=${HEAD_COMMIT}"
  exit 0
fi
if [[ "${PRESENT_COUNT}" != "0" ]]; then
  log_event \
    "RESUME bound_partial_artifacts=${PRESENT_COUNT} immutable_reuse_required=true"
fi
check_stop

CURRENT_PHASE="build"
write_state "RUNNING" "building exact-transfer candidates and blocked queue"
log_event "START phase=build labeled=4 targets=3 languages=TypeScript,JavaScript"
COMMAND=(
  datasets build-resolution-recovery
  --source open-swe-traces
  --snapshot "${TRACE_SNAPSHOT}"
  --partition-contract "${PARTITION_CONTRACT}"
  --transfer-receipt "${TRACE_RECEIPT}"
)
for partition in "${LABELED_PARTITIONS[@]}"; do
  COMMAND+=(--labeled-partition "${partition}")
done
for partition in "${TARGET_PARTITIONS[@]}"; do
  COMMAND+=(--target-partition "${partition}")
done
COMMAND+=(
  --language TypeScript
  --language JavaScript
  --candidates-output "${CANDIDATES}"
  --queue-output "${QUEUE}"
  --manifest-output "${MANIFEST}"
  --config "${REGISTRY}"
)

setsid --wait "${UV_BIN}" run --frozen --no-sync --directory "${EXEC_ROOT}" \
  python -m nodelm "${COMMAND[@]}" >>"${RUN_LOG}" 2>&1 &
ACTIVE_PID=$!
ACTIVE_PGID="${ACTIVE_PID}"
write_state "RUNNING" "artifact generation active; child_pid=${ACTIVE_PID}"
set +e
wait "${ACTIVE_PID}"
PRODUCER_STATUS=$?
set -e
ACTIVE_PID=""
ACTIVE_PGID=""
require_code_unchanged
require_inputs_unchanged
[[ ${PRODUCER_STATUS} -eq 0 ]] || fail "resolution recovery producer exited ${PRODUCER_STATUS}"
count_artifacts
[[ "${PRESENT_COUNT}" == "3" ]] || fail \
  "resolution recovery producer did not publish a complete terminal artifact set"

CURRENT_PHASE="validate"
write_state "RUNNING" "validating published immutable artifacts"
if ! validate_terminal >>"${RUN_LOG}" 2>&1; then
  fail "published recovery artifacts are invalid"
fi
require_code_unchanged
require_inputs_unchanged

CURRENT_PHASE="complete"
write_state "COMPLETE" "resolution recovery artifacts validated; evaluator canary remains pending"
log_event "COMPLETE commit=${HEAD_COMMIT} admission=BLOCKED blocker=harness_canary_pending"
