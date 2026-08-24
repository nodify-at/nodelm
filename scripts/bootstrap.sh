#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
export UV_CACHE_DIR="${NODELM_UV_CACHE_DIR:-${PROJECT_ROOT}/.cache/uv}"

on_error() {
  local exit_code=$?
  echo "NodeLM bootstrap failed at line ${BASH_LINENO[0]} (exit ${exit_code})." >&2
  exit "${exit_code}"
}
trap on_error ERR

command -v uv >/dev/null 2>&1 || {
  echo "uv is required. Install it from https://docs.astral.sh/uv/." >&2
  exit 127
}

cd -- "${PROJECT_ROOT}"
mkdir -p artifacts/manifests artifacts/reports artifacts/logs .cache/nodelm
uv sync --frozen --group dev
uv run --frozen nodelm datasets validate
echo "NodeLM bootstrap complete. Run ./scripts/doctor.sh and make verify."
