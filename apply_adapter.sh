#!/bin/bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNTIME=${1:-"$ROOT/runtime/flagos-runtime-0.2.0-alpha.1-xing4-0-darwin-arm64-m5pro"}
SITE="$RUNTIME/python/lib/python3.11/site-packages"
[ -x "$RUNTIME/python/bin/python3" ] || { echo "Runtime is missing: $RUNTIME" >&2; exit 2; }
/usr/bin/rsync -a --exclude '__pycache__/' --exclude '*.pyc' "$ROOT/adapter/vllm_fl/" "$SITE/vllm_fl/"
/bin/cp "$ROOT/launcher/vllm" "$RUNTIME/bin/vllm"
/bin/cp "$ROOT/launcher/bench_xingchen4_w4a8_stages.py" \
  "$RUNTIME/share/flagos/examples/bench_xingchen4_w4a8_stages.py"
"$RUNTIME/python/bin/python3" -m compileall -q --invalidation-mode checked-hash \
  -s "$RUNTIME" -p flagos-runtime "$SITE/vllm_fl"
echo "Xing4_0 alias adapter installed in $RUNTIME"
