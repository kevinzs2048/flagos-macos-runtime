#!/bin/bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VLLM_SOURCE=${FLAGOS_VLLM_SOURCE:?Set FLAGOS_VLLM_SOURCE to the clean vLLM v0.20.2 source root}
PYTHON=${FLAGOS_BUILD_PYTHON:-$VLLM_SOURCE/.venv311/bin/python}

[ -x "$PYTHON" ] || { echo "Build Python is missing: $PYTHON" >&2; exit 2; }

run_with_timeout() {
  seconds=$1
  shift
  /usr/bin/perl -e 'alarm shift; exec @ARGV' "$seconds" "$@"
}

run_with_timeout 900 "$PYTHON" "$ROOT/scripts/materialize_sources.py"
run_with_timeout 1200 "$ROOT/scripts/build_runtime.sh"
run_with_timeout 600 "$ROOT/scripts/build_wheelhouse.sh"
run_with_timeout 60 "$PYTHON" "$ROOT/scripts/update_release_metadata.py"
run_with_timeout 600 "$PYTHON" "$ROOT/scripts/verify_release.py"
env -u FLAGOS_VLLM_SOURCE -u FLAGOS_LIBOMP_ROOT -u FLAGOS_BUILD_PYTHON \
  -u FLAGOS_CMAKE -u FLAGOS_NINJA \
  /usr/bin/perl -e 'alarm shift; exec @ARGV' 600 \
  "$ROOT/scripts/test_install_workflow.sh"
env -u FLAGOS_VLLM_SOURCE -u FLAGOS_LIBOMP_ROOT -u FLAGOS_BUILD_PYTHON \
  -u FLAGOS_CMAKE -u FLAGOS_NINJA \
  /usr/bin/perl -e 'alarm shift; exec @ARGV' 300 \
  "$ROOT/scripts/test_wheelhouse.sh"

echo "Developer release is ready under $ROOT/artifacts"
