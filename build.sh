#!/bin/bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ "$#" -gt 2 ]; then
  echo "Usage: build.sh [VLLM_SOURCE [LIBOMP_ROOT]]" >&2
  exit 2
fi

if [ "$#" -ge 1 ]; then
  export FLAGOS_VLLM_SOURCE=$1
fi
if [ "$#" -ge 2 ]; then
  export FLAGOS_LIBOMP_ROOT=$2
fi

: "${FLAGOS_VLLM_SOURCE:?Pass VLLM_SOURCE or set FLAGOS_VLLM_SOURCE}"
: "${FLAGOS_LIBOMP_ROOT:?Pass LIBOMP_ROOT or set FLAGOS_LIBOMP_ROOT}"

exec "$ROOT/scripts/package_release.sh"
