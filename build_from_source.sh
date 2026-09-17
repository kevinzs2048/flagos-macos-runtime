#!/bin/bash
# Reuse the pinned base builder and apply the Xing4_0 overlay to a NEW Runtime.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo 'Usage: bash build_from_source.sh DEVELOPER_SOURCES BOOTSTRAP_RUNTIME OUTPUT_RUNTIME' >&2
  exit 2
fi
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || {
  echo 'Native macOS arm64 is required.' >&2; exit 2;
}
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SOURCE_ROOT=$(CDPATH= cd -- "$1" && pwd)
BOOTSTRAP_ROOT=$(CDPATH= cd -- "$2" && pwd)
OUTPUT_INPUT=$3
[ ! -e "$OUTPUT_INPUT" ] && [ ! -L "$OUTPUT_INPUT" ] || {
  echo 'Output Runtime must not exist.' >&2; exit 2;
}
BASE_BUILDER="$SOURCE_ROOT/flagos-macos-runtime-xingchen4-w4a8-v024"
[ -f "$BASE_BUILDER/build.sh" ] || { echo 'Matching base builder is missing.' >&2; exit 2; }
[ -x "$BOOTSTRAP_ROOT/bin/vllm" ] || { echo 'Bootstrap launcher is missing.' >&2; exit 2; }
command -v ninja >/dev/null || { echo 'Ninja is required on PATH.' >&2; exit 2; }
for task_path in "$TASK_ROOT" "$SOURCE_ROOT" "$BOOTSTRAP_ROOT" "$OUTPUT_INPUT"; do
  case "$task_path" in *[[:space:]]*) echo 'Use paths without whitespace.' >&2; exit 2 ;; esac
done
case "$("$BOOTSTRAP_ROOT/bin/vllm" --version)" in
  0.24.0+cpu) ;; *) echo 'Bootstrap must be vLLM 0.24.0+cpu.' >&2; exit 2 ;;
esac

# Reject mismatched/modified checkouts before starting the native build.
"$BOOTSTRAP_ROOT/bin/vllm" --flagos-run-python - "$TASK_ROOT" "$SOURCE_ROOT" "$BOOTSTRAP_ROOT" "$OUTPUT_INPUT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
task, source, bootstrap, output = map(Path, sys.argv[1:])
output = output.resolve()
if output.is_relative_to(source) or output.is_relative_to(bootstrap):
    raise SystemExit('Output must be outside the source inputs and bootstrap.')
public_path = task / 'sources.public.lock.json'
release = json.loads((public_path if public_path.exists() else task / 'sources.lock.json').read_text())
lock = release['base_components']
snapshots = release.get('public_snapshot_inputs', {})
for directory, key in [('vllm-0.24.0', 'vllm'), ('FlagGems', 'flag_gems'),
                       ('vllm-plugin-FL-v024', 'vllm_plugin_fl'),
                       ('libtriton_jit', 'libtriton_jit'),
                       ('KleidiAI-prefill-experiment', 'kleidiai')]:
    checkout = source / directory
    head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    expected = snapshots.get(directory, {}).get('commit', lock[key]['commit'])
    if head != expected:
        raise SystemExit('Pinned commit mismatch: ' + directory)
    dirty = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty.strip():
        raise SystemExit('Tracked modifications in source input: ' + directory)
for directory, entry in snapshots.items():
    for name, expected in entry['files_sha256'].items():
        file = source / directory / name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise SystemExit('Source inventory mismatch: ' + directory + '/' + name)
print('Pinned source revisions and file inventories verified.')
PY

# The old builder replaces these generated paths. Require a fresh builder copy.
for generated in \
  "$BASE_BUILDER/build/runtime-0.2.0-alpha.1-xingchen4" \
  "$BASE_BUILDER/build/plugin-wheel-0.2.0-alpha.1-xingchen4" \
  "$BASE_BUILDER/build/plugin-install-0.2.0-alpha.1-xingchen4" \
  "$BASE_BUILDER/artifacts/flagos-runtime-0.2.0-alpha.1-xingchen4-darwin-arm64-m5pro.tar.gz"; do
  [ ! -e "$generated" ] || { echo 'Use a fresh base-builder copy; generated outputs already exist.' >&2; exit 2; }
done

export FLAGOS_XINGCHEN_WORK_ROOT="$SOURCE_ROOT"
export FLAGOS_BOOTSTRAP_RUNTIME="$BOOTSTRAP_ROOT"
export FLAGOS_VLLM_SOURCE="$SOURCE_ROOT/vllm-0.24.0"
export FLAGOS_FLAGGEMS_SOURCE="$SOURCE_ROOT/FlagGems"
export FLAGOS_PLUGIN_SOURCE="$SOURCE_ROOT/vllm-plugin-FL-v024"
export FLAGOS_JIT_SOURCE="$SOURCE_ROOT/libtriton_jit"
export FLAGOS_KLEIDIAI_SOURCE="$SOURCE_ROOT/KleidiAI-prefill-experiment"
export FLAGGEMS_BUILD="$SOURCE_ROOT/FlagGems/build-xing4-0-mac"
export TRITON_JIT_BUILD="$SOURCE_ROOT/libtriton_jit/build-xing4-0-mac"
export BUILD_JOBS=4
CMAKE_BIN="$BOOTSTRAP_ROOT/python/lib/python3.11/site-packages/cmake/data/bin/cmake"
"$CMAKE_BIN" -S "$FLAGOS_KLEIDIAI_SOURCE" -B "$FLAGOS_KLEIDIAI_SOURCE/build-xingchen4-prefill" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DKLEIDIAI_BUILD_TESTS=OFF -DKLEIDIAI_BUILD_BENCHMARK=OFF
"$CMAKE_BIN" --build "$FLAGOS_KLEIDIAI_SOURCE/build-xingchen4-prefill" --parallel 4
bash "$BASE_BUILDER/build.sh"

STAGE="$BASE_BUILDER/build/runtime-0.2.0-alpha.1-xingchen4/flagos-runtime-0.2.0-alpha.1-xingchen4-darwin-arm64-m5pro"
mkdir -p "$(dirname -- "$OUTPUT_INPUT")"
cp -cR "$STAGE" "$OUTPUT_INPUT"
OUTPUT_ROOT=$(CDPATH= cd -- "$OUTPUT_INPUT" && pwd)
bash "$TASK_ROOT/apply_adapter.sh" "$OUTPUT_ROOT"
cp "$TASK_ROOT/sources.lock.json" "$OUTPUT_ROOT/share/flagos/sources.lock.json"
if [ -f "$TASK_ROOT/sources.public.lock.json" ]; then
  cp "$TASK_ROOT/sources.public.lock.json" "$OUTPUT_ROOT/share/flagos/sources.public.lock.json"
fi
"$OUTPUT_ROOT/bin/vllm" --flagos-run-python - "$OUTPUT_ROOT" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1]) / 'share/flagos/runtime-manifest.json'
manifest = json.loads(path.read_text())
manifest['name'] = 'flagos-macos-runtime-xing4-0-w4a8-development'
manifest['version'] = '0.2.0-alpha.1-xing4-0-local-source-build'
manifest['base_historical_supported_models'] = manifest.pop('supported_models', [])
manifest['validation'] = {'status': 'pending-local-source-build-validation'}
manifest['supported_models'] = [{'architecture': 'Xing4_0ForCausalLM', 'model_type': 'xing4_0',
                                'speculative_decoding': False, 'validation_status': 'pending'}]
manifest['source_build_policy'] = 'Pinned native/plugin build; reuse validated Python/Torch/Triton CPU/vLLM ABI bootstrap. Validate this build separately.'
path.write_text(json.dumps(manifest, indent=2) + '\n')
PY
"$OUTPUT_ROOT/bin/vllm" --flagos-run-python "$TASK_ROOT/scripts/hash_tree.py" \
  "$OUTPUT_ROOT" "$OUTPUT_ROOT/share/flagos/runtime-files.sha256.json"
"$OUTPUT_ROOT/bin/vllm" --flagos-run-python "$TASK_ROOT/scripts/verify_runtime.py" "$OUTPUT_ROOT"
"$OUTPUT_ROOT/bin/vllm" --version
echo 'Source Runtime built. Run correctness and performance tests before publishing it.'
