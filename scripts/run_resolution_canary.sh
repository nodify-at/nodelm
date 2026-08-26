#!/usr/bin/env bash
set -Eeuo pipefail

# Materialize and execute the provenance-bound resolution canary on persistent storage.
# Run this as an unprivileged user with rootless Podman. Image/evaluator preparation may use
# outbound network access; every repository test attempt is forced to --network=none.

REPO_ROOT="${NODELM_REPO_ROOT:-/workspace/nodelm/repo}"
PERSIST_ROOT="${NODELM_PERSIST_ROOT:-/workspace/nodelm}"
EXEC_TMP_ROOT="${NODELM_EXEC_TMP_ROOT:-/tmp/nodelm-resolution-canary-exec}"
UV_BIN="${NODELM_UV_BIN:-uv}"
RECOVERY_DIR="${NODELM_RECOVERY_DIR:-${PERSIST_ROOT}/derived/resolution-recovery-74c9b505eb1a608431ae3a18a3fca5d084f2ae3b}"
TRACE_SNAPSHOT="${NODELM_TRACE_SNAPSHOT:-${PERSIST_ROOT}/snapshots/open-swe-traces}"
TRACE_RECEIPT="${NODELM_TRACE_RECEIPT:-${PERSIST_ROOT}/receipts/open-swe-traces.transfer.json}"
TASK_SNAPSHOT="${NODELM_TASK_SNAPSHOT:-${PERSIST_ROOT}/snapshots/swe-rebench-v2}"
TASK_RECEIPT="${NODELM_TASK_RECEIPT:-${PERSIST_ROOT}/receipts/swe-rebench-v2.transfer.json}"
EVALUATOR_REVISION='c71902a8cf8d2b725f63d51f199f4d3e56f68d2d'
EVALUATOR_URL="${NODELM_EVALUATOR_URL:-https://github.com/SWE-rebench/SWE-rebench-V2.git}"
EVALUATOR_ROOT="${NODELM_EVALUATOR_ROOT:-${PERSIST_ROOT}/evaluators/SWE-rebench-V2-${EVALUATOR_REVISION}}"

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
LAST_DETAIL="initializing"
HEAD_COMMIT="unknown"
RUN_DIR=""
STATE_FILE=""
EVENTS_LOG=""
STOP_FILE=""
ACTIVE_PID=""
ACTIVE_PGID=""
EXEC_PARENT=""
EXEC_ROOT=""

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
    printf 'stop_file=%s\n' "${STOP_FILE}"
    printf 'operator_stop=touch stop_file before evaluation; kill -TERM pid during active work\n'
  } >"${temporary}"
  mv -- "${temporary}" "${STATE_FILE}"
}

log_event() {
  local line
  line="$(timestamp) $1"
  printf '%s\n' "${line}"
  [[ -n "${EVENTS_LOG}" ]] && printf '%s\n' "${line}" >>"${EVENTS_LOG}"
}

fail() {
  LAST_DETAIL="$1"
  log_event "ERROR $1"
  return 1
}

cleanup_execution_tree() {
  local expected_prefix
  [[ -n "${EXEC_PARENT}" && -n "${HEAD_COMMIT}" ]] || return 0
  expected_prefix="${EXEC_TMP_ROOT}/nodelm-resolution-canary-${HEAD_COMMIT}."
  [[ "${EXEC_PARENT}" == "${expected_prefix}"* ]] || return 1
  [[ -d "${EXEC_PARENT}" && ! -L "${EXEC_PARENT}" ]] || return 0
  chmod -R u+w "${EXEC_PARENT}" 2>/dev/null || true
  rm -rf -- "${EXEC_PARENT}"
}

stop_active_process_group() {
  local attempt
  local container_id
  if [[ "${ACTIVE_PGID}" =~ ^[1-9][0-9]*$ ]] &&
    kill -0 -- "-${ACTIVE_PGID}" 2>/dev/null; then
    kill -TERM -- "-${ACTIVE_PGID}" 2>/dev/null || true
    for ((attempt = 0; attempt < 50; attempt++)); do
      kill -0 -- "-${ACTIVE_PGID}" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-${ACTIVE_PGID}" 2>/dev/null || true
  fi
  if [[ "${ACTIVE_PID}" =~ ^[1-9][0-9]*$ ]]; then
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  ACTIVE_PID=""
  ACTIVE_PGID=""
  while IFS= read -r container_id; do
    [[ -n "${container_id}" ]] || continue
    podman rm --force --ignore --time=0 -- "${container_id}" >/dev/null 2>&1 || true
  done < <(podman ps --all --quiet --filter label=io.nodelm.resolution-canary=true 2>/dev/null)
}

on_signal() {
  trap '' TERM INT
  stop_active_process_group
  CURRENT_PHASE="signal"
  write_state "STOPPED" "received signal $1; active child and canary containers stopped"
  log_event "STOPPED signal=$1"
  exit 130
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  if [[ "${CURRENT_STATE}" == "STARTING" || "${CURRENT_STATE}" == "RUNNING" ]]; then
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
  [[ -f "$1" && ! -L "$1" ]] || fail "required input is missing, special, or a symlink: $1"
}

file_sha256() {
  sha256sum -- "$1" | awk '{print $1}'
}

check_stop() {
  if [[ -e "${STOP_FILE}" || -L "${STOP_FILE}" ]]; then
    [[ -f "${STOP_FILE}" && ! -L "${STOP_FILE}" ]] || fail "stop sentinel is unsafe"
    CURRENT_PHASE="stopped"
    write_state "STOPPED" "stop sentinel observed before evaluation"
    log_event "STOPPED sentinel=${STOP_FILE}"
    exit 0
  fi
}

materialize_execution_tree() {
  local archive
  [[ "${EXEC_TMP_ROOT}" == /* && "${EXEC_TMP_ROOT}" != "/" ]] || fail \
    "NODELM_EXEC_TMP_ROOT must be an absolute, non-root directory"
  mkdir -p -- "${EXEC_TMP_ROOT}"
  EXEC_TMP_ROOT="$(cd -- "${EXEC_TMP_ROOT}" && pwd -P)"
  EXEC_PARENT="$(mktemp -d "${EXEC_TMP_ROOT}/nodelm-resolution-canary-${HEAD_COMMIT}.XXXXXX")"
  EXEC_ROOT="${EXEC_PARENT}/tree"
  archive="${EXEC_PARENT}/source.tar"
  mkdir -- "${EXEC_ROOT}"
  git -C "${REPO_ROOT}" archive --format=tar --output="${archive}" "${HEAD_COMMIT}"
  tar -xf "${archive}" -C "${EXEC_ROOT}"
  rm -- "${archive}"
  require_regular_file "${EXEC_ROOT}/pyproject.toml"
  require_regular_file "${EXEC_ROOT}/uv.lock"
  require_regular_file "${EXEC_ROOT}/scripts/run_resolution_canary.sh"
  export PYTHONPATH="${EXEC_ROOT}/src"
}

run_python() {
  "${UV_BIN}" run --frozen --no-sync --directory "${EXEC_ROOT}" python "$@"
}

run_active() {
  local description="$1"
  shift
  LAST_DETAIL="${description}"
  setsid --wait "$@" &
  ACTIVE_PID=$!
  ACTIVE_PGID="${ACTIVE_PID}"
  wait "${ACTIVE_PID}"
  ACTIVE_PID=""
  ACTIVE_PGID=""
}

prepare_evaluator() {
  local parent temporary
  if [[ ! -e "${EVALUATOR_ROOT}" ]]; then
    parent="$(dirname -- "${EVALUATOR_ROOT}")"
    mkdir -p -- "${parent}"
    temporary="$(mktemp -d "${parent}/.SWE-rebench-V2.XXXXXX")"
    git clone --filter=blob:none --no-checkout "${EVALUATOR_URL}" "${temporary}"
    git -C "${temporary}" checkout --detach "${EVALUATOR_REVISION}"
    mv -- "${temporary}" "${EVALUATOR_ROOT}"
  fi
  [[ -d "${EVALUATOR_ROOT}" && ! -L "${EVALUATOR_ROOT}" ]] || fail \
    "evaluator checkout is missing, special, or a symlink"
  [[ "$(git -C "${EVALUATOR_ROOT}" rev-parse --verify 'HEAD^{commit}')" == \
    "${EVALUATOR_REVISION}" ]] || fail "evaluator checkout revision mismatch"
  [[ -z "$(git -C "${EVALUATOR_ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail \
    "evaluator checkout is dirty"
  [[ "$(file_sha256 "${EVALUATOR_ROOT}/lib/agent/log_parsers.py")" == \
    'a717b03efde1cb79dfb11e2a57d0262c0057d352a347a9fb09667ef6e5f6f20c' ]] || fail \
    "evaluator parser digest mismatch"
  [[ "$(file_sha256 "${EVALUATOR_ROOT}/scripts/eval.py")" == \
    '4768c0c3e2adf3540c2228f819f4b073e4665ada06fa00f2234a1f7620d69eda' ]] || fail \
    "evaluator script digest mismatch"
  [[ "$(file_sha256 "${EVALUATOR_ROOT}/lib/agent/swe_constants.py")" == \
    '823dd1ef512d363ed5d4dce05d70f22d7f93b25722cda5b0971f17010f5168a5' ]] || fail \
    "evaluator constants digest mismatch"
}

require_inputs_unchanged() {
  [[ "$(git -C "${REPO_ROOT}" rev-parse --verify 'HEAD^{commit}')" == "${HEAD_COMMIT}" ]] || fail \
    "repository HEAD changed while canary was active"
  [[ -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail \
    "production tree changed while canary was active"
  [[ "$(file_sha256 "${RECOVERY_MANIFEST}")" == "${RECOVERY_MANIFEST_SHA256}" ]] || fail \
    "recovery manifest changed while canary was active"
  [[ "$(file_sha256 "${CANDIDATES}")" == "${CANDIDATES_SHA256}" ]] || fail \
    "recovery candidates changed while canary was active"
  [[ "$(file_sha256 "${QUEUE}")" == "${QUEUE_SHA256}" ]] || fail \
    "recovery queue changed while canary was active"
  [[ "$(file_sha256 "${TRACE_RECEIPT}")" == "${TRACE_RECEIPT_SHA256}" ]] || fail \
    "trace receipt changed while canary was active"
  [[ "$(file_sha256 "${TASK_RECEIPT}")" == "${TASK_RECEIPT_SHA256}" ]] || fail \
    "task receipt changed while canary was active"
}

validate_terminal() {
  run_python - "${WORKSET}" "${WORKSET_MANIFEST}" "${IMAGE_LOCK}" "${RESULTS}" \
    "${EXECUTION_MANIFEST}" "${CASE_EVIDENCE_DIR}" "${HEAD_COMMIT}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from nodelm.artifacts import file_identity
from nodelm.evaluation.resolution_canary import (
    ResolutionCanaryCase,
    ResolutionCanaryCaseResult,
    ResolutionCanaryImageLock,
    ResolutionCanaryPrivateCaseEvidence,
)
from nodelm.provenance.manifests import (
    ResolutionCanaryExecutionManifestV1,
    ResolutionCanaryWorksetManifestV1,
)

workset, workset_manifest, image_lock, results, execution_manifest, evidence_dir = map(
    Path, sys.argv[1:7]
)
commit = sys.argv[7]
inputs = (workset, workset_manifest, image_lock, results, execution_manifest)
if any(not path.is_file() or path.is_symlink() for path in inputs):
    raise SystemExit("terminal canary artifact is missing, special, or a symlink")
if not evidence_dir.is_dir() or evidence_dir.is_symlink():
    raise SystemExit("terminal private evidence directory is missing or unsafe")
workset_record = ResolutionCanaryWorksetManifestV1.model_validate_json(
    workset_manifest.read_bytes()
)
lock = ResolutionCanaryImageLock.model_validate_json(image_lock.read_bytes())
execution = ResolutionCanaryExecutionManifestV1.model_validate_json(
    execution_manifest.read_bytes()
)
try:
    cases = tuple(
        ResolutionCanaryCase.model_validate_json(line)
        for line in workset.read_text(encoding="utf-8").splitlines()
    )
except ValueError:
    raise SystemExit("terminal private workset failed schema validation") from None
case_results = tuple(
    ResolutionCanaryCaseResult.model_validate_json(line)
    for line in results.read_text(encoding="utf-8").splitlines()
)
if file_identity(workset) != (workset_record.workset_sha256, workset_record.workset_bytes):
    raise SystemExit("terminal workset identity mismatch")
if lock.workset_sha256 != workset_record.workset_sha256:
    raise SystemExit("terminal image lock identity mismatch")
if file_identity(results) != (execution.results_sha256, execution.results_bytes):
    raise SystemExit("terminal result identity mismatch")
if execution.workset_manifest_sha256 != file_identity(workset_manifest)[0]:
    raise SystemExit("terminal execution/workset manifest mismatch")
if execution.image_lock_sha256 != file_identity(image_lock)[0]:
    raise SystemExit("terminal execution/image lock mismatch")
if execution.code_commit != commit:
    raise SystemExit("terminal execution commit mismatch")
if tuple(result.case_id for result in case_results) != tuple(case.case_id for case in cases):
    raise SystemExit("terminal result cases do not exactly cover workset")
expected_evidence = {f"{case.case_id}.json" for case in cases}
if {path.name for path in evidence_dir.iterdir()} != expected_evidence:
    raise SystemExit("terminal private evidence does not exactly cover workset")
for case, result in zip(cases, case_results, strict=True):
    path = evidence_dir / f"{case.case_id}.json"
    if not path.is_file() or path.is_symlink():
        raise SystemExit("terminal private evidence entry is unsafe")
    try:
        private = ResolutionCanaryPrivateCaseEvidence.model_validate_json(path.read_bytes())
    except ValueError:
        raise SystemExit("terminal private case evidence failed schema validation") from None
    if private.result != result:
        raise SystemExit("terminal private evidence/result mismatch")
print(
    f"validated cases={execution.case_count} execution={execution.execution_status} "
    f"admission={execution.admission_status}"
)
PY
}

require_command git
require_command tar
require_command mktemp
require_command sha256sum
require_command flock
require_command setsid
require_command podman
require_command "${UV_BIN}"
[[ "${EUID}" -ne 0 ]] || fail "resolution canary must run as an unprivileged user"
require_regular_file "${REPO_ROOT}/pyproject.toml"
require_regular_file "${REPO_ROOT}/uv.lock"
require_regular_file "${RECOVERY_DIR}/resolution-recovery.manifest.json"
require_regular_file "${RECOVERY_DIR}/exact-resolution-candidates.jsonl"
require_regular_file "${RECOVERY_DIR}/resolution-evaluation-queue.jsonl"
require_regular_file "${TRACE_RECEIPT}"
require_regular_file "${TASK_RECEIPT}"
[[ -d "${TRACE_SNAPSHOT}" && ! -L "${TRACE_SNAPSHOT}" ]] || fail "trace snapshot is unsafe"
[[ -d "${TASK_SNAPSHOT}" && ! -L "${TASK_SNAPSHOT}" ]] || fail "task snapshot is unsafe"

HEAD_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --verify 'HEAD^{commit}')"
[[ "${HEAD_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "repository HEAD is not a full commit"
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail \
  "production checkout must be clean"
RUN_DIR="${PERSIST_ROOT}/derived/resolution-canary-${HEAD_COMMIT}"
mkdir -p -- "${RUN_DIR}"
RUN_DIR="$(cd -- "${RUN_DIR}" && pwd -P)"
STATE_FILE="${RUN_DIR}/run.state"
EVENTS_LOG="${RUN_DIR}/events.log"
STOP_FILE="${RUN_DIR}/STOP"
exec 9>"${RUN_DIR}/run.lock"
flock -n 9 || fail "another canary runner owns this commit-bound run directory"

RECOVERY_MANIFEST="${RECOVERY_DIR}/resolution-recovery.manifest.json"
CANDIDATES="${RECOVERY_DIR}/exact-resolution-candidates.jsonl"
QUEUE="${RECOVERY_DIR}/resolution-evaluation-queue.jsonl"
RECOVERY_MANIFEST_SHA256="$(file_sha256 "${RECOVERY_MANIFEST}")"
CANDIDATES_SHA256="$(file_sha256 "${CANDIDATES}")"
QUEUE_SHA256="$(file_sha256 "${QUEUE}")"
TRACE_RECEIPT_SHA256="$(file_sha256 "${TRACE_RECEIPT}")"
TASK_RECEIPT_SHA256="$(file_sha256 "${TASK_RECEIPT}")"

WORKSET="${RUN_DIR}/resolution-canary.private.jsonl"
WORKSET_MANIFEST="${RUN_DIR}/resolution-canary.workset.manifest.json"
IMAGE_LOCK="${RUN_DIR}/resolution-canary.images.json"
CASE_EVIDENCE_DIR="${RUN_DIR}/private-case-evidence"
RESULTS="${RUN_DIR}/resolution-canary.results.jsonl"
EXECUTION_MANIFEST="${RUN_DIR}/resolution-canary.execution.manifest.json"

CURRENT_STATE="RUNNING"
write_state "RUNNING" "preflight passed"
log_event "RUNNING commit=${HEAD_COMMIT} output=${RUN_DIR}"
materialize_execution_tree
REGISTRY="${EXEC_ROOT}/configs/datasets/registry.yaml"
PARTITION_CONTRACT="${EXEC_ROOT}/configs/datasets/open-swe-trace-partitions.yaml"

if [[ -f "${EXECUTION_MANIFEST}" && -f "${RESULTS}" && -f "${IMAGE_LOCK}" &&
  -f "${WORKSET}" && -f "${WORKSET_MANIFEST}" ]]; then
  CURRENT_PHASE="validate"
  validate_terminal
  CURRENT_PHASE="complete"
  write_state "COMPLETE" "existing terminal canary artifacts validated"
  log_event "COMPLETE reused=true"
  exit 0
fi

check_stop
CURRENT_PHASE="materialize"
write_state "RUNNING" "materializing provenance-bound private workset"
run_active "materializing private canary workset" \
  "${UV_BIN}" run --frozen --no-sync --directory "${EXEC_ROOT}" python -m nodelm datasets \
  build-resolution-canary-workset \
  --recovery-manifest "${RECOVERY_MANIFEST}" \
  --candidates "${CANDIDATES}" \
  --queue "${QUEUE}" \
  --trace-snapshot "${TRACE_SNAPSHOT}" \
  --trace-transfer-receipt "${TRACE_RECEIPT}" \
  --partition-contract "${PARTITION_CONTRACT}" \
  --task-snapshot "${TASK_SNAPSHOT}" \
  --task-transfer-receipt "${TASK_RECEIPT}" \
  --workset-output "${WORKSET}" \
  --manifest-output "${WORKSET_MANIFEST}" \
  --config "${REGISTRY}"
require_inputs_unchanged

check_stop
CURRENT_PHASE="prepare_evaluator"
write_state "RUNNING" "preparing pinned evaluator checkout"
prepare_evaluator

check_stop
CURRENT_PHASE="pull_images"
write_state "RUNNING" "pulling selected canary images and recording immutable digests"
run_active "locking selected canary image digests" \
  "${UV_BIN}" run --frozen --no-sync --directory "${EXEC_ROOT}" python -m nodelm datasets \
  lock-resolution-canary-images \
  --workset "${WORKSET}" \
  --workset-manifest "${WORKSET_MANIFEST}" \
  --output "${IMAGE_LOCK}"
require_inputs_unchanged

check_stop
CURRENT_PHASE="evaluate"
write_state "RUNNING" "running offline real-repository canary attempts"
mkdir -p -- "${CASE_EVIDENCE_DIR}"
run_active "executing offline canary cases" \
  "${UV_BIN}" run --frozen --no-sync --directory "${EXEC_ROOT}" python -m nodelm datasets \
  run-resolution-canary \
  --workset "${WORKSET}" \
  --workset-manifest "${WORKSET_MANIFEST}" \
  --image-lock "${IMAGE_LOCK}" \
  --evaluator-root "${EVALUATOR_ROOT}" \
  --case-evidence-dir "${CASE_EVIDENCE_DIR}" \
  --results-output "${RESULTS}" \
  --manifest-output "${EXECUTION_MANIFEST}" \
  --code-commit "${HEAD_COMMIT}"
require_inputs_unchanged

CURRENT_PHASE="validate"
write_state "RUNNING" "validating terminal canary evidence"
validate_terminal
CURRENT_PHASE="complete"
write_state "COMPLETE" "resolution canary artifacts validated"
log_event "COMPLETE manifest=${EXECUTION_MANIFEST}"
