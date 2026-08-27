#!/usr/bin/env bash
set -Eeuo pipefail

# Review the four selected Open-SWE v1.0 leaves against their exact materialized raw
# rows. This is an offline CPU/data-integrity job; it does not load a model or use a GPU.
# STOP is honored between leaves. During a leaf, send TERM to the PID in run.state.

REPO_ROOT="${NODELM_REPO_ROOT:-/workspace/nodelm/repo}"
PERSIST_ROOT="${NODELM_PERSIST_ROOT:-/workspace/nodelm}"
NORMALIZATION_ROOT="${NODELM_NORMALIZATION_ROOT:-${PERSIST_ROOT}/derived/full-normalization-18c0ada5f396191d247cfe57640b6f2bb9fade86}"
TASK_PROVENANCE="${NODELM_TASK_PROVENANCE:-${PERSIST_ROOT}/derived/task-provenance/swe-rebench-v2.safe.jsonl}"
TASK_PROVENANCE_MANIFEST="${NODELM_TASK_PROVENANCE_MANIFEST:-${PERSIST_ROOT}/derived/task-provenance/swe-rebench-v2.safe.manifest.json}"
EXEC_TMP_ROOT="${NODELM_EXEC_TMP_ROOT:-/tmp/nodelm-oracle-review-exec}"
STAGE_TMP_ROOT="${NODELM_STAGE_TMP_ROOT:-/tmp/nodelm-oracle-review-stage}"
UV_BIN="${NODELM_UV_BIN:-uv}"

export UV_OFFLINE=1
export UV_CACHE_DIR="${NODELM_UV_CACHE_DIR:-${PERSIST_ROOT}/cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/nodelm-venv}"
export UV_NO_SYNC=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="${STAGE_TMP_ROOT}"

readonly LEAVES=(
  'openhands-minimax-v2|openhands/minimax_m25/swe-rebench-v2'
  'openhands-qwen35-v2|openhands/qwen35_122b/swe-rebench-v2'
  'sweagent-minimax-v2|sweagent/minimax_m25/swe-rebench-v2'
  'sweagent-qwen35-v2|sweagent/qwen35_122b/swe-rebench-v2'
)

CURRENT_STATE="STARTING"
CURRENT_PHASE="preflight"
CURRENT_LEAF="none"
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
    printf 'leaf=%s\n' "${CURRENT_LEAF}"
    printf 'detail=%s\n' "${detail}"
    printf 'stop_file=%s\n' "${STOP_FILE}"
    printf 'operator_stop=touch stop_file between leaves; kill -TERM pid during an active leaf\n'
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

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command is unavailable: $1"
}

require_regular_file() {
  [[ -f "$1" && ! -L "$1" ]] || fail "required input is missing, special, or a symlink: $1"
}

file_sha256() {
  sha256sum -- "$1" | awk '{print $1}'
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
    kill -KILL -- "-${ACTIVE_PGID}" 2>/dev/null || true
  fi
  if [[ "${ACTIVE_PID}" =~ ^[1-9][0-9]*$ ]]; then
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  ACTIVE_PID=""
  ACTIVE_PGID=""
}

cleanup_execution_tree() {
  local expected_prefix
  [[ -n "${EXEC_PARENT}" && -n "${HEAD_COMMIT}" ]] || return 0
  expected_prefix="${EXEC_TMP_ROOT}/nodelm-oracle-review-${HEAD_COMMIT}."
  [[ "${EXEC_PARENT}" == "${expected_prefix}"* ]] || return 1
  [[ -d "${EXEC_PARENT}" && ! -L "${EXEC_PARENT}" ]] || return 0
  chmod -R u+w "${EXEC_PARENT}" 2>/dev/null || true
  rm -rf -- "${EXEC_PARENT}"
}

on_signal() {
  trap '' TERM INT
  stop_active_process_group
  CURRENT_PHASE="signal"
  write_state "STOPPED" "received signal $1; active review child stopped"
  log_event "STOPPED signal=$1 leaf=${CURRENT_LEAF}"
  exit 130
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  if [[ "${CURRENT_STATE}" == "STARTING" || "${CURRENT_STATE}" == "RUNNING" ]]; then
    write_state "FAILED" "${LAST_DETAIL}; exit=${exit_code}"
    log_event "FAILED exit=${exit_code} phase=${CURRENT_PHASE} leaf=${CURRENT_LEAF}"
  fi
  cleanup_execution_tree || true
  exit "${exit_code}"
}

trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap on_exit EXIT

check_stop() {
  if [[ -e "${STOP_FILE}" || -L "${STOP_FILE}" ]]; then
    [[ -f "${STOP_FILE}" && ! -L "${STOP_FILE}" ]] || fail "stop sentinel is unsafe"
    CURRENT_PHASE="stopped"
    write_state "STOPPED" "stop sentinel observed between leaves"
    log_event "STOPPED sentinel=${STOP_FILE}"
    exit 0
  fi
}

require_code_unchanged() {
  [[ "$(git -C "${REPO_ROOT}" rev-parse --verify 'HEAD^{commit}')" == "${HEAD_COMMIT}" ]] ||
    fail "repository HEAD changed while the review run was active"
  [[ -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ]] ||
    fail "production tree changed while the review run was active"
}

materialize_execution_tree() {
  local archive
  [[ "${EXEC_TMP_ROOT}" == /* && "${EXEC_TMP_ROOT}" != "/" ]] || fail \
    "NODELM_EXEC_TMP_ROOT must be an absolute, non-root directory"
  mkdir -p -- "${EXEC_TMP_ROOT}"
  [[ -d "${EXEC_TMP_ROOT}" && ! -L "${EXEC_TMP_ROOT}" ]] || fail \
    "private execution root is missing, special, or a symlink"
  EXEC_TMP_ROOT="$(cd -- "${EXEC_TMP_ROOT}" && pwd -P)"
  EXEC_PARENT="$(mktemp -d "${EXEC_TMP_ROOT}/nodelm-oracle-review-${HEAD_COMMIT}.XXXXXX")"
  EXEC_ROOT="${EXEC_PARENT}/tree"
  archive="${EXEC_PARENT}/source.tar"
  mkdir -- "${EXEC_ROOT}"
  git -C "${REPO_ROOT}" archive --format=tar --output="${archive}" "${HEAD_COMMIT}"
  tar -xf "${archive}" -C "${EXEC_ROOT}"
  rm -- "${archive}"
  require_regular_file "${EXEC_ROOT}/pyproject.toml"
  require_regular_file "${EXEC_ROOT}/uv.lock"
  require_regular_file "${EXEC_ROOT}/scripts/run_oracle_isolation_reviews.sh"
  export PYTHONPATH="${EXEC_ROOT}/src"
}

run_python() {
  "${UV_BIN}" run --frozen --no-sync --directory "${EXEC_ROOT}" python "$@"
}

validate_terminal_pair() {
  local attestation="$1"
  local findings="$2"
  local raw="$3"
  local materialization="$4"
  local normalized="$5"
  local normalization="$6"
  local task_provenance="$7"
  local task_manifest="$8"
  local partition="$9"

  run_python - "${attestation}" "${findings}" "${raw}" "${materialization}" \
    "${normalized}" "${normalization}" "${task_provenance}" "${task_manifest}" "${partition}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from nodelm.artifacts import file_identity
from nodelm.models import VerificationStatus
from nodelm.provenance.gold import OracleIsolationAttestation
from nodelm.provenance.oracle_isolation import resolve_artifact


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


(
    attestation_path,
    findings_path,
    raw_path,
    materialization_path,
    normalized_path,
    normalization_path,
    task_provenance_path,
    task_manifest_path,
) = (Path(value).resolve() for value in sys.argv[1:9])
partition = sys.argv[9]
payload = json.loads(attestation_path.read_text(encoding="utf-8"))
attestation = OracleIsolationAttestation.model_validate(payload)
require(attestation.status is VerificationStatus.PASS, "attestation status is not PASS")
require(attestation.partition_name == partition, "attestation partition does not match leaf")
require(
    resolve_artifact(attestation_path, attestation.findings_artifact) == findings_path,
    "attestation findings path does not match leaf",
)
require(
    resolve_artifact(attestation_path, attestation.raw_artifact) == raw_path,
    "attestation raw path does not match leaf",
)
require(
    resolve_artifact(attestation_path, attestation.materialization_manifest_artifact)
    == materialization_path,
    "attestation materialization manifest path does not match leaf",
)
require(
    resolve_artifact(attestation_path, attestation.normalized_artifact) == normalized_path,
    "attestation normalized path does not match leaf",
)
require(
    resolve_artifact(attestation_path, attestation.normalization_manifest_artifact)
    == normalization_path,
    "attestation normalization manifest path does not match leaf",
)
require(file_identity(findings_path) == (
    attestation.findings_sha256,
    attestation.findings_bytes,
), "attestation findings identity does not match leaf")
require(file_identity(raw_path) == (attestation.raw_sha256, attestation.raw_bytes),
        "attestation raw identity does not match leaf")
require(file_identity(materialization_path) == (
    attestation.materialization_manifest_sha256,
    attestation.materialization_manifest_bytes,
), "attestation materialization manifest identity does not match leaf")
require(file_identity(normalized_path) == (
    attestation.normalized_sha256,
    attestation.normalized_bytes,
), "attestation normalized identity does not match leaf")
require(file_identity(normalization_path) == (
    attestation.normalization_manifest_sha256,
    attestation.normalization_manifest_bytes,
), "attestation normalization manifest identity does not match leaf")
PY
}

run_leaf() {
  local slug="$1"
  local partition="$2"
  local raw="${NORMALIZATION_ROOT}/${slug}.raw.jsonl"
  local materialization="${NORMALIZATION_ROOT}/${slug}.raw.manifest.json"
  local normalized="${NORMALIZATION_ROOT}/${slug}.normalized.jsonl"
  local normalization="${NORMALIZATION_ROOT}/${slug}.normalized.manifest.json"
  local task_provenance="${TASK_PROVENANCE}"
  local task_manifest="${TASK_PROVENANCE_MANIFEST}"
  local attestation="${RUN_DIR}/${slug}.oracle-isolation.attestation.json"
  local findings="${RUN_DIR}/${slug}.oracle-isolation.findings.jsonl"
  local present=0
  local command_status=0

  CURRENT_LEAF="${partition}"
  CURRENT_PHASE="review"
  write_state "RUNNING" "checking oracle-isolation checkpoint"
  require_code_unchanged
  check_stop
  require_regular_file "${raw}"
  require_regular_file "${materialization}"
  require_regular_file "${normalized}"
  require_regular_file "${normalization}"
  require_regular_file "${task_provenance}"
  require_regular_file "${task_manifest}"
  [[ -e "${attestation}" || -L "${attestation}" ]] && present=$((present + 1))
  [[ -e "${findings}" || -L "${findings}" ]] && present=$((present + 1))
  if ((present > 0)); then
    # Existing artifacts are untrusted claims until this exact materialized tree
    # recomputes them. The immutable writers below permit byte-identical replay and
    # reject any collision, including a forged self-consistent terminal pair.
    [[ ! -e "${attestation}" && ! -L "${attestation}" ]] || require_regular_file "${attestation}"
    [[ ! -e "${findings}" && ! -L "${findings}" ]] || require_regular_file "${findings}"
    log_event "REPLAY leaf=${partition} checkpoint_files=${present}"
  fi

  log_event "START leaf=${partition}"
  write_state "RUNNING" "oracle-isolation review in progress"
  LAST_DETAIL="reviewing ${partition}"
  set +e
  setsid --wait "${UV_BIN}" run --frozen --no-sync --directory "${EXEC_ROOT}" python \
    -m nodelm datasets review-oracle-isolation \
    --raw-input "${raw}" \
    --materialization-manifest "${materialization}" \
    --input "${normalized}" \
    --normalization-manifest "${normalization}" \
    --task-provenance "${task_provenance}" \
    --task-provenance-manifest "${task_manifest}" \
    --output "${attestation}" \
    --findings-output "${findings}" >>"${RUN_LOG}" 2>&1 &
  ACTIVE_PID=$!
  ACTIVE_PGID="${ACTIVE_PID}"
  wait "${ACTIVE_PID}"
  command_status=$?
  ACTIVE_PID=""
  ACTIVE_PGID=""
  set -e
  require_code_unchanged
  [[ -f "${attestation}" && ! -L "${attestation}" && -f "${findings}" && ! -L "${findings}" ]] ||
    fail "oracle-isolation review did not publish a terminal pair for ${slug}; exit=${command_status}"
  validate_terminal_pair "${attestation}" "${findings}" "${raw}" "${materialization}" \
    "${normalized}" "${normalization}" "${task_provenance}" "${task_manifest}" "${partition}" >>"${RUN_LOG}" 2>&1 ||
    fail "oracle-isolation review is not PASS for ${slug}; exit=${command_status}"
  ((command_status == 0)) || fail "PASS oracle-isolation review exited ${command_status} for ${slug}"
  log_event "COMPLETE leaf=${partition} attestation_sha256=$(file_sha256 "${attestation}")"
}

require_command git
require_command sha256sum
require_command flock
require_command setsid
require_command tar
require_command "${UV_BIN}"
[[ "${STAGE_TMP_ROOT}" == /* && "${STAGE_TMP_ROOT}" != "/" ]] || fail \
  "NODELM_STAGE_TMP_ROOT must be an absolute, non-root directory"
mkdir -p -- "${STAGE_TMP_ROOT}"
[[ -d "${STAGE_TMP_ROOT}" && ! -L "${STAGE_TMP_ROOT}" ]] || fail \
  "staging root is missing, special, or a symlink"
STAGE_TMP_ROOT="$(cd -- "${STAGE_TMP_ROOT}" && pwd -P)"
export TMPDIR="${STAGE_TMP_ROOT}"
[[ -d "${REPO_ROOT}" && ! -L "${REPO_ROOT}" ]] || fail "repository root is unsafe"
[[ -d "${NORMALIZATION_ROOT}" && ! -L "${NORMALIZATION_ROOT}" ]] || fail \
  "normalization root is missing, special, or a symlink"
HEAD_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --verify 'HEAD^{commit}')"
[[ "${HEAD_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "repository HEAD is not a full commit"
require_code_unchanged

RUN_DIR="${PERSIST_ROOT}/audits/oracle-isolation-review-${HEAD_COMMIT}"
mkdir -p -- "${RUN_DIR}"
[[ -d "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] || fail "run directory is unsafe"
STATE_FILE="${RUN_DIR}/run.state"
EVENTS_LOG="${RUN_DIR}/events.log"
RUN_LOG="${RUN_DIR}/runner.log"
STOP_FILE="${RUN_DIR}/STOP"
exec 9>"${RUN_DIR}/run.lock"
flock -n 9 || fail "another oracle-isolation review runner holds the run lock"

materialize_execution_tree
binding="format=nodelm-oracle-isolation-review-run-binding-v1
commit=${HEAD_COMMIT}
tree=$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')
runner_sha256=$(file_sha256 "${EXEC_ROOT}/scripts/run_oracle_isolation_reviews.sh")"
if [[ -e "${RUN_DIR}/run.binding" || -L "${RUN_DIR}/run.binding" ]]; then
  require_regular_file "${RUN_DIR}/run.binding"
  [[ "$(cat -- "${RUN_DIR}/run.binding")" == "${binding}" ]] || fail \
    "run binding does not match the exact execution tree"
else
  printf '%s\n' "${binding}" >"${RUN_DIR}/run.binding"
fi

CURRENT_STATE="RUNNING"
write_state "RUNNING" "preflight complete"
log_event "RUNNING commit=${HEAD_COMMIT} output=${RUN_DIR}"
check_stop
for leaf in "${LEAVES[@]}"; do
  IFS='|' read -r slug partition <<<"${leaf}"
  run_leaf "${slug}" "${partition}"
done

CURRENT_LEAF="none"
CURRENT_PHASE="final-code-check"
require_code_unchanged
CURRENT_PHASE="complete"
write_state "COMPLETE" "four oracle-isolation reviews PASS and await digest authorization"
log_event "COMPLETE leaves=4 authorization=PENDING"
