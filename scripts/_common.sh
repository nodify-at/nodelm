#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
export UV_CACHE_DIR="${NODELM_UV_CACHE_DIR:-${PROJECT_ROOT}/.cache/uv}"

on_error() {
  local exit_code=$?
  echo "NodeLM command failed at line ${BASH_LINENO[0]} (exit ${exit_code})." >&2
  exit "${exit_code}"
}
trap on_error ERR

cd -- "${PROJECT_ROOT}"

require_uv() {
  command -v uv >/dev/null 2>&1 || {
    echo "uv is required. Install it from https://docs.astral.sh/uv/." >&2
    return 127
  }
}

run_nodelm() {
  require_uv
  uv run --frozen nodelm "$@"
}

run_nodelm_training() {
  require_uv
  uv run --frozen --extra training nodelm "$@"
}
