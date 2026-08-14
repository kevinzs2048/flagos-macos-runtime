#!/bin/bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VERSION=0.1.0-alpha.1
STAGE="$ROOT/build/runtime-$VERSION/flagos-runtime-$VERSION-darwin-arm64-m5pro"
PYTHON="$STAGE/python/bin/python3.11"
SITE="$STAGE/python/lib/python3.11/site-packages"
DEST="$ROOT/build/wheelhouse-$VERSION"

export PYTHONHOME="$STAGE/python"
export PYTHONPATH="$SITE"
export DYLD_LIBRARY_PATH="$STAGE/lib:$STAGE/python/lib:$STAGE/python/readline/lib:$STAGE/python/openssl/lib:$SITE/torch/lib"

[ -x "$PYTHON" ] || { echo "Build Runtime first: scripts/build_runtime.sh" >&2; exit 2; }
mkdir -p "$DEST" "$ROOT/artifacts"
find "$DEST" -mindepth 1 -maxdepth 1 -delete
WORK=$(/usr/bin/mktemp -d "$ROOT/build/wheel-pack.XXXXXX")
trap '/bin/rm -rf -- "$WORK"' EXIT

pack_component() {
  package=$1
  metadata_glob=$2
  component="$WORK/$package"
  mkdir -p "$component"
  /usr/bin/ditto "$SITE/$package" "$component/$package"
  metadata=$(find "$SITE" -maxdepth 1 -type d -name "$metadata_glob" -print -quit)
  [ -n "$metadata" ] || { echo "Missing metadata: $metadata_glob" >&2; exit 2; }
  /usr/bin/ditto "$metadata" "$component/$(basename "$metadata")"
  "$PYTHON" -m wheel pack "$component" --dest-dir "$DEST"
}

pack_component vllm 'vllm-*.dist-info'
pack_component triton 'triton-*.dist-info'
pack_component flag_gems 'flag_gems-*.dist-info'
pack_component vllm_fl 'vllm_plugin_fl-*.dist-info'

(cd "$DEST" && /usr/bin/shasum -a 256 *.whl > SHA256SUMS)
ARCHIVE="$ROOT/artifacts/flagos-wheelhouse-$VERSION-cp311-darwin-arm64.tar.gz"
COPYFILE_DISABLE=1 /usr/bin/tar -C "$ROOT/build" -czf "$ARCHIVE" "$(basename "$DEST")"
(cd "$ROOT/artifacts" && /usr/bin/shasum -a 256 "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
echo "$ARCHIVE"
