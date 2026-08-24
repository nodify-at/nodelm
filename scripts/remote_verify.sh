#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

on_error() {
  local exit_code=$?
  echo "Remote verification failed at line ${BASH_LINENO[0]} (exit ${exit_code})." >&2
  exit "${exit_code}"
}
trap on_error ERR

usage() {
  cat <<'EOF'
Usage: remote_verify.sh --host HOST --user USER --remote-root PATH [options]

Connection options:
  --port PORT                 SSH port (default: 22)
  --key PATH                  Local SSH private-key path
  --run-id ID                 Safe unique run ID (generated when omitted)

Full verification options (supply all five together):
  --training-config PATH      Remote pinned training configuration
  --training-samples PATH     Remote normalized pilot samples JSONL
  --pilot-manifest PATH       Remote pilot manifest matching the samples
  --checkpoint-dir PATH       Fresh remote checkpoint directory
  --sandbox-image IMAGE       Preloaded image as name@sha256:<64 lowercase hex>
EOF
}

HOST=""
USER_NAME=""
PORT="22"
KEY_PATH=""
REMOTE_ROOT=""
RUN_ID=""
TRAINING_CONFIG=""
TRAINING_SAMPLES=""
PILOT_MANIFEST=""
CHECKPOINT_DIR=""
SANDBOX_IMAGE=""

while (($#)); do
  case "$1" in
    --host) HOST="${2:?--host requires a value}"; shift 2 ;;
    --user) USER_NAME="${2:?--user requires a value}"; shift 2 ;;
    --port) PORT="${2:?--port requires a value}"; shift 2 ;;
    --key) KEY_PATH="${2:?--key requires a local path}"; shift 2 ;;
    --remote-root) REMOTE_ROOT="${2:?--remote-root requires a path}"; shift 2 ;;
    --run-id) RUN_ID="${2:?--run-id requires a value}"; shift 2 ;;
    --training-config) TRAINING_CONFIG="${2:?--training-config requires a value}"; shift 2 ;;
    --training-samples) TRAINING_SAMPLES="${2:?--training-samples requires a value}"; shift 2 ;;
    --pilot-manifest) PILOT_MANIFEST="${2:?--pilot-manifest requires a value}"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="${2:?--checkpoint-dir requires a value}"; shift 2 ;;
    --sandbox-image) SANDBOX_IMAGE="${2:?--sandbox-image requires a value}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${HOST}" =~ ^[A-Za-z0-9._:-]+$ ]] || {
  echo "Invalid or missing --host." >&2
  exit 2
}
[[ "${USER_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "Invalid or missing --user." >&2
  exit 2
}
[[ "${PORT}" =~ ^[0-9]+$ ]] && ((PORT >= 1 && PORT <= 65535)) || {
  echo "--port must be between 1 and 65535." >&2
  exit 2
}
[[ -n "${REMOTE_ROOT}" && "${REMOTE_ROOT}" == /* ]] || {
  echo "--remote-root must be an absolute path to an existing NodeLM checkout." >&2
  exit 2
}

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"
fi
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || {
  echo "--run-id must be 1-64 characters: letters, digits, dot, underscore, or hyphen." >&2
  exit 2
}

FULL_INPUT_COUNT=0
for value in \
  "${TRAINING_CONFIG}" \
  "${TRAINING_SAMPLES}" \
  "${PILOT_MANIFEST}" \
  "${CHECKPOINT_DIR}" \
  "${SANDBOX_IMAGE}"
do
  if [[ -n "${value}" ]]; then
    ((FULL_INPUT_COUNT += 1))
  fi
done
if ((FULL_INPUT_COUNT != 0 && FULL_INPUT_COUNT != 5)); then
  echo "Full verification requires --training-config, --training-samples, " \
    "--pilot-manifest, --checkpoint-dir, and --sandbox-image together." >&2
  exit 2
fi
FULL_VERIFICATION=false
if ((FULL_INPUT_COUNT == 5)); then
  FULL_VERIFICATION=true
  [[ "${SANDBOX_IMAGE}" =~ ^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$ ]] || {
    echo "--sandbox-image must be name@sha256:<64 lowercase hexadecimal characters>." >&2
    exit 2
  }
fi

SSH_ARGS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=yes
  -o ConnectTimeout=10
  -o ConnectionAttempts=1
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=3
  -p "${PORT}"
)
if [[ -n "${KEY_PATH}" ]]; then
  [[ -f "${KEY_PATH}" ]] || {
    echo "Local SSH key path does not exist." >&2
    exit 2
  }
  SSH_ARGS+=(-i "${KEY_PATH}")
fi

REMOTE_RUN_PARENT="artifacts/reports/infra/runs"
REMOTE_RUN_DIR="${REMOTE_RUN_PARENT}/${RUN_ID}"
REMOTE_MARKER="${REMOTE_RUN_DIR}/.invocation-token"
REMOTE_INFRA_REPORT="${REMOTE_RUN_DIR}/environment.json"
REMOTE_TRAINING_REPORT="${REMOTE_RUN_DIR}/training-lifecycle.json"
REMOTE_HARNESS_REPORT="${REMOTE_RUN_DIR}/harness.json"
INVOCATION_TOKEN="${RUN_ID}-$$-${RANDOM}-${RANDOM}"

LOCAL_HOST_COMPONENT="host-${HOST//:/_}"
LOCAL_RUN_PARENT="${PROJECT_ROOT}/artifacts/reports/infra/runs/${LOCAL_HOST_COMPONENT}"
LOCAL_RUN_DIR="${LOCAL_RUN_PARENT}/${RUN_ID}"
LOCAL_INFRA_REPORT="${LOCAL_RUN_DIR}/environment.json"
LOCAL_TRAINING_REPORT="${LOCAL_RUN_DIR}/training-lifecycle.json"
LOCAL_HARNESS_REPORT="${LOCAL_RUN_DIR}/harness.json"

umask 077
mkdir -p -- "${LOCAL_RUN_PARENT}"
if ! mkdir -- "${LOCAL_RUN_DIR}"; then
  echo "Local run directory already exists; choose a different --run-id: ${LOCAL_RUN_DIR}" >&2
  exit 2
fi

TEMP_FILES=()
cleanup() {
  local temporary
  for temporary in "${TEMP_FILES[@]:-}"; do
    if [[ -n "${temporary}" ]]; then
      rm -f -- "${temporary}"
    fi
  done
}
trap cleanup EXIT

command -v python3 >/dev/null 2>&1 || {
  echo "python3 is required to validate copied JSON evidence." >&2
  exit 127
}

validate_json_artifact() {
  local artifact_path=$1
  local expected_schema=$2
  local expected_status=${3:-}
  python3 -c \
    'import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(value, dict):
    raise SystemExit("artifact must contain a JSON object")
actual_schema = value.get("schema_version")
if actual_schema != sys.argv[2]:
    raise SystemExit(f"unexpected schema_version: {actual_schema!r}")
actual_status = value.get("status")
if actual_status not in {"PASS", "FAIL", "NOT RUN", "BLOCKED", "UNVERIFIED"}:
    raise SystemExit(f"unexpected status: {actual_status!r}")
if sys.argv[3] and actual_status != sys.argv[3]:
    raise SystemExit(f"expected status {sys.argv[3]!r}, observed {actual_status!r}")' \
    "${artifact_path}" "${expected_schema}" "${expected_status}"
}

printf -v REMOTE_ROOT_QUOTED '%q' "${REMOTE_ROOT}"
printf -v REMOTE_RUN_PARENT_QUOTED '%q' "${REMOTE_RUN_PARENT}"
printf -v REMOTE_RUN_DIR_QUOTED '%q' "${REMOTE_RUN_DIR}"
printf -v REMOTE_MARKER_QUOTED '%q' "${REMOTE_MARKER}"
printf -v REMOTE_INFRA_REPORT_QUOTED '%q' "${REMOTE_INFRA_REPORT}"
printf -v REMOTE_TRAINING_REPORT_QUOTED '%q' "${REMOTE_TRAINING_REPORT}"
printf -v REMOTE_HARNESS_REPORT_QUOTED '%q' "${REMOTE_HARNESS_REPORT}"
printf -v INVOCATION_TOKEN_QUOTED '%q' "${INVOCATION_TOKEN}"

copy_current_json() {
  local remote_artifact=$1
  local local_artifact=$2
  local expected_schema=$3
  local label=$4
  local expected_status=${5:-}
  local remote_artifact_quoted
  local temporary
  local remote_copy_command

  printf -v remote_artifact_quoted '%q' "${remote_artifact}"
  if ! temporary="$(mktemp "${local_artifact}.tmp.XXXXXX")"; then
    echo "Could not allocate a temporary file for ${label}." >&2
    return 1
  fi
  TEMP_FILES+=("${temporary}")
  remote_copy_command="cd -- ${REMOTE_ROOT_QUOTED} && \
test -f ${REMOTE_MARKER_QUOTED} && \
test \"\$(cat -- ${REMOTE_MARKER_QUOTED})\" = ${INVOCATION_TOKEN_QUOTED} && \
test -f ${remote_artifact_quoted} && \
cat -- ${remote_artifact_quoted}"
  if ! ssh "${SSH_ARGS[@]}" "${USER_NAME}@${HOST}" \
    "${remote_copy_command}" > "${temporary}"
  then
    echo "No current-invocation ${label} was available to copy." >&2
    return 1
  fi
  if ! validate_json_artifact "${temporary}" "${expected_schema}" "${expected_status}"; then
    echo "Refusing invalid ${label} from the remote run." >&2
    return 1
  fi
  if ! mv -- "${temporary}" "${local_artifact}"; then
    echo "Could not publish local ${label}." >&2
    return 1
  fi
}

REMOTE_PREP_COMMAND="umask 077 && \
cd -- ${REMOTE_ROOT_QUOTED} && \
mkdir -p -- ${REMOTE_RUN_PARENT_QUOTED} && \
mkdir -- ${REMOTE_RUN_DIR_QUOTED} && \
printf '%s\\n' ${INVOCATION_TOKEN_QUOTED} > ${REMOTE_MARKER_QUOTED}"
if [[ "${FULL_VERIFICATION}" == true ]]; then
  REMOTE_INFRA_COMMAND="${REMOTE_PREP_COMMAND} && \
command -v uv >/dev/null 2>&1 && \
uv sync --frozen --extra training && \
./scripts/remote_doctor.sh --with-training-runtime --output ${REMOTE_INFRA_REPORT_QUOTED}"
else
  REMOTE_INFRA_COMMAND="${REMOTE_PREP_COMMAND} && \
./scripts/remote_doctor.sh --output ${REMOTE_INFRA_REPORT_QUOTED}"
fi

if ssh "${SSH_ARGS[@]}" "${USER_NAME}@${HOST}" "${REMOTE_INFRA_COMMAND}"; then
  INFRA_EXIT=0
else
  INFRA_EXIT=$?
fi
INFRA_COPIED=false
INFRA_EXPECTED_STATUS=""
if ((INFRA_EXIT == 0)); then
  INFRA_EXPECTED_STATUS="PASS"
fi
if copy_current_json \
  "${REMOTE_INFRA_REPORT}" \
  "${LOCAL_INFRA_REPORT}" \
  "nodelm.infrastructure/v2" \
  "infrastructure report" \
  "${INFRA_EXPECTED_STATUS}"
then
  INFRA_COPIED=true
fi
if ((INFRA_EXIT != 0)); then
  if [[ "${INFRA_COPIED}" == true ]]; then
    echo "FAIL: strict remote infrastructure gate failed; evidence: ${LOCAL_INFRA_REPORT}" >&2
  else
    echo "FAIL: remote setup or infrastructure collection failed without current evidence." >&2
  fi
  exit "${INFRA_EXIT}"
fi
if [[ "${INFRA_COPIED}" != true ]]; then
  echo "FAIL: the infrastructure gate passed without a valid current report." >&2
  exit 1
fi

if [[ "${FULL_VERIFICATION}" != true ]]; then
  echo "Infrastructure evidence: ${LOCAL_INFRA_REPORT}"
  echo "Run ID: ${RUN_ID}"
  echo "BLOCKED: full verification requires all five lifecycle inputs, including --sandbox-image."
  exit 2
fi

printf -v TRAINING_CONFIG_QUOTED '%q' "${TRAINING_CONFIG}"
printf -v TRAINING_SAMPLES_QUOTED '%q' "${TRAINING_SAMPLES}"
printf -v PILOT_MANIFEST_QUOTED '%q' "${PILOT_MANIFEST}"
printf -v CHECKPOINT_DIR_QUOTED '%q' "${CHECKPOINT_DIR}"
printf -v SANDBOX_IMAGE_QUOTED '%q' "${SANDBOX_IMAGE}"

REMOTE_SANDBOX_COMMAND="cd -- ${REMOTE_ROOT_QUOTED} && \
command -v podman >/dev/null 2>&1 && \
test \"\$(podman info --format '{{.Host.Security.Rootless}}')\" = true && \
podman image exists ${SANDBOX_IMAGE_QUOTED}"
if ! ssh "${SSH_ARGS[@]}" "${USER_NAME}@${HOST}" "${REMOTE_SANDBOX_COMMAND}"; then
  echo "BLOCKED: full verification requires rootless Podman and the exact preloaded " \
    "--sandbox-image; implicit pulls are forbidden." >&2
  exit 2
fi

REMOTE_TRAINING_COMMAND="cd -- ${REMOTE_ROOT_QUOTED} && \
./scripts/training_lifecycle.sh \
--config ${TRAINING_CONFIG_QUOTED} \
--samples ${TRAINING_SAMPLES_QUOTED} \
--pilot-manifest ${PILOT_MANIFEST_QUOTED} \
--checkpoint-dir ${CHECKPOINT_DIR_QUOTED} \
--sandbox-image ${SANDBOX_IMAGE_QUOTED} \
--output ${REMOTE_TRAINING_REPORT_QUOTED}"
if ssh "${SSH_ARGS[@]}" "${USER_NAME}@${HOST}" "${REMOTE_TRAINING_COMMAND}"; then
  TRAINING_EXIT=0
else
  TRAINING_EXIT=$?
fi
TRAINING_COPIED=false
TRAINING_EXPECTED_STATUS=""
if ((TRAINING_EXIT == 0)); then
  TRAINING_EXPECTED_STATUS="PASS"
fi
if copy_current_json \
  "${REMOTE_TRAINING_REPORT}" \
  "${LOCAL_TRAINING_REPORT}" \
  "nodelm.training-lifecycle-command/v1" \
  "training lifecycle report" \
  "${TRAINING_EXPECTED_STATUS}"
then
  TRAINING_COPIED=true
fi
if ((TRAINING_EXIT != 0)); then
  if [[ "${TRAINING_COPIED}" == true ]]; then
    echo "FAIL: training lifecycle gate failed; evidence: ${LOCAL_TRAINING_REPORT}" >&2
  else
    echo "FAIL: training lifecycle failed before producing current JSON evidence." >&2
  fi
  exit "${TRAINING_EXIT}"
fi
if [[ "${TRAINING_COPIED}" != true ]]; then
  echo "FAIL: training lifecycle passed without a valid current report." >&2
  exit 1
fi

REMOTE_HARNESS_TEMP="${REMOTE_HARNESS_REPORT}.tmp"
printf -v REMOTE_HARNESS_TEMP_QUOTED '%q' "${REMOTE_HARNESS_TEMP}"
REMOTE_HARNESS_COMMAND="umask 077
cd -- ${REMOTE_ROOT_QUOTED} || exit 1
./scripts/verify_harness.sh --json > ${REMOTE_HARNESS_TEMP_QUOTED}
harness_status=\$?
mv -- ${REMOTE_HARNESS_TEMP_QUOTED} ${REMOTE_HARNESS_REPORT_QUOTED} || exit 1
exit \${harness_status}"
if ssh "${SSH_ARGS[@]}" "${USER_NAME}@${HOST}" "${REMOTE_HARNESS_COMMAND}"; then
  HARNESS_EXIT=0
else
  HARNESS_EXIT=$?
fi
HARNESS_COPIED=false
HARNESS_EXPECTED_STATUS=""
if ((HARNESS_EXIT == 0)); then
  HARNESS_EXPECTED_STATUS="PASS"
fi
if copy_current_json \
  "${REMOTE_HARNESS_REPORT}" \
  "${LOCAL_HARNESS_REPORT}" \
  "nodelm.harness-verification/v1" \
  "harness report" \
  "${HARNESS_EXPECTED_STATUS}"
then
  HARNESS_COPIED=true
fi
if ((HARNESS_EXIT != 0)); then
  if [[ "${HARNESS_COPIED}" == true ]]; then
    echo "FAIL: general harness gate failed; evidence: ${LOCAL_HARNESS_REPORT}" >&2
  else
    echo "FAIL: general harness gate failed without valid current JSON evidence." >&2
  fi
  exit "${HARNESS_EXIT}"
fi
if [[ "${HARNESS_COPIED}" != true ]]; then
  echo "FAIL: general harness passed without a valid current report." >&2
  exit 1
fi

echo "PASS: remote verification completed for run ${RUN_ID}."
echo "Infrastructure evidence: ${LOCAL_INFRA_REPORT}"
echo "Training lifecycle evidence: ${LOCAL_TRAINING_REPORT}"
echo "Harness evidence: ${LOCAL_HARNESS_REPORT}"
