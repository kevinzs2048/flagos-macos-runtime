#!/bin/bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VERSION=0.1.0-alpha.1
RUNTIME="$ROOT/build/runtime-$VERSION/flagos-runtime-$VERSION-darwin-arm64-m5pro"
PYTHON="$RUNTIME/python/bin/python3.11"
ARCHIVE="$ROOT/artifacts/flagos-wheelhouse-$VERSION-cp311-darwin-arm64.tar.gz"
TEST_ROOT=$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/flagos-wheel-smoke.XXXXXX")

cleanup() {
  case "$TEST_ROOT" in
    "${TMPDIR:-/tmp}"/flagos-wheel-smoke.*) /bin/rm -rf -- "$TEST_ROOT" ;;
    *) echo "Refusing unsafe smoke-test cleanup: $TEST_ROOT" >&2 ;;
  esac
}
trap cleanup EXIT

[ -x "$PYTHON" ] || { echo "Built Runtime Python is missing" >&2; exit 2; }
[ -f "$ARCHIVE" ] || { echo "Wheelhouse asset is missing" >&2; exit 2; }

/usr/bin/tar -C "$TEST_ROOT" -xzf "$ARCHIVE"
WHEEL_ROOT="$TEST_ROOT/wheelhouse-$VERSION"
SITE="$TEST_ROOT/site"
(cd "$WHEEL_ROOT" && /usr/bin/shasum -a 256 -c SHA256SUMS)

PYTHONHOME="$RUNTIME/python" \
DYLD_LIBRARY_PATH="$RUNTIME/lib:$RUNTIME/python/lib:$RUNTIME/python/readline/lib:$RUNTIME/python/openssl/lib:$RUNTIME/python/lib/python3.11/site-packages/torch/lib" \
  "$PYTHON" -m pip install --no-index --no-deps --target "$SITE" \
  "$WHEEL_ROOT"/*.whl

PYTHONHOME="$RUNTIME/python" PYTHONPATH="$SITE" \
DYLD_LIBRARY_PATH="$RUNTIME/lib:$RUNTIME/python/lib:$RUNTIME/python/readline/lib:$RUNTIME/python/openssl/lib:$RUNTIME/python/lib/python3.11/site-packages/torch/lib" \
  "$PYTHON" -c '
import importlib.metadata as metadata

expected = {
    "flag-gems",
    "triton",
    "vllm",
    "vllm-plugin-fl",
}
for name in expected:
    metadata.version(name)
'

while IFS= read -r link; do
  target=$(/usr/bin/readlink "$link")
  case "$target" in
    /*) echo "wheel installed an absolute symlink: $link -> $target" >&2; exit 2 ;;
  esac
done < <(/usr/bin/find "$SITE" -type l -print)

echo "Offline wheelhouse workflow: PASS"
