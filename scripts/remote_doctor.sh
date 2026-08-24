#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

if [[ "${1:-}" == "--with-training-runtime" ]]; then
  shift
  run_nodelm_training infra doctor --require-gpu "$@"
else
  run_nodelm infra doctor --require-gpu "$@"
fi
