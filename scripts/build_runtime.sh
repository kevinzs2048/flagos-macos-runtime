#!/bin/bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VERSION=0.1.0-alpha.1
VLLM_SOURCE=${FLAGOS_VLLM_SOURCE:?Set FLAGOS_VLLM_SOURCE to the clean vLLM v0.20.2 source root}
LIBOMP_ROOT=${FLAGOS_LIBOMP_ROOT:?Set FLAGOS_LIBOMP_ROOT to the relocatable libomp prefix}
BUILD_PYTHON=${FLAGOS_BUILD_PYTHON:-$VLLM_SOURCE/.venv311/bin/python}
CMAKE=${FLAGOS_CMAKE:-$VLLM_SOURCE/.venv311/bin/cmake}
NINJA=${FLAGOS_NINJA:-$VLLM_SOURCE/.venv311/bin/ninja}
PYTHON_BASE=$($BUILD_PYTHON -c 'import sys; print(sys.base_prefix)')
BUILD_SITE=$($BUILD_PYTHON -c 'import site; print(site.getsitepackages()[0])')
TRITON_RUNTIME_PACKAGE=$($BUILD_PYTHON -c \
  'import pathlib, triton; print(pathlib.Path(triton.__file__).resolve().parent)')

BUILD_ROOT="$ROOT/build/runtime-$VERSION"
STAGE="$BUILD_ROOT/flagos-runtime-$VERSION-darwin-arm64-m5pro"
SITE="$STAGE/python/lib/python3.11/site-packages"
NATIVE_JIT_BUILD="$ROOT/build/native-libtriton-jit"
NATIVE_OPS_BUILD="$ROOT/build/native-flag-gems-arm"
PLUGIN_WHEEL_ROOT="$ROOT/build/plugin-wheel"
PLUGIN_INSTALL_ROOT="$ROOT/build/plugin-install"
SOURCE_ROOT="$ROOT/build/sources"

for required in "$BUILD_PYTHON" "$CMAKE" "$NINJA"; do
  [ -x "$required" ] || { echo "Required build tool is missing: $required" >&2; exit 2; }
done
for source in \
  "$SOURCE_ROOT/vllm/pyproject.toml" \
  "$SOURCE_ROOT/triton-cpu/python/triton/__init__.py" \
  "$SOURCE_ROOT/FlagGems/src/flag_gems/__init__.py" \
  "$SOURCE_ROOT/vllm-plugin-FL/pyproject.toml" \
  "$SOURCE_ROOT/libtriton_jit/CMakeLists.txt" \
  "$SOURCE_ROOT/libtriton_jit/scripts/gen_ssig.py" \
  "$SOURCE_ROOT/libtriton_jit/scripts/standalone_compile.py"
do
  [ -f "$source" ] || { echo "Materialized source is missing: $source" >&2; exit 2; }
done
[ -d "$TRITON_RUNTIME_PACKAGE/_C" ] || {
  echo "The verified Triton compiler package has no _C directory" >&2
  exit 2
}

for generated in \
  "$STAGE" "$NATIVE_JIT_BUILD" "$NATIVE_OPS_BUILD" \
  "$PLUGIN_WHEEL_ROOT" "$PLUGIN_INSTALL_ROOT"
do
  case "$generated" in
    "$ROOT"/build/*) /bin/rm -rf -- "$generated" ;;
    *) echo "Refusing unsafe generated path: $generated" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_ROOT" "$STAGE/python" "$STAGE/lib" "$STAGE/include" \
  "$STAGE/bin" "$STAGE/share/flagos" "$SITE"

# Copy the verified Python distribution and non-editable dependencies. Source
# packages under development are replaced below by exact locked Git exports.
/usr/bin/rsync -a --delete \
  --exclude 'lib/python3.11/site-packages/' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude 'test/' --exclude 'tests/' \
  "$PYTHON_BASE/" "$STAGE/python/"
# The inference Runtime exposes only its embedded interpreter. Developer
# console scripts carry build-prefix shebangs and are neither relocatable nor
# needed; the Runtime installs its own standard vLLM launcher below.
find "$STAGE/python/bin" -mindepth 1 -maxdepth 1 \
  ! -name python ! -name python3 ! -name python3.11 -delete
/bin/rm -rf -- "$STAGE/python/lib/pkgconfig" \
  "$STAGE/python/lib/python3.11/config-3.11-darwin" \
  "$STAGE/python/readline/lib/pkgconfig" \
  "$STAGE/python/openssl/lib/pkgconfig"
/usr/bin/rsync -a --delete \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' \
  --exclude '__editable__*' --exclude '_flag_gems_editable*' \
  "$BUILD_SITE/" "$SITE/"
"$BUILD_PYTHON" "$ROOT/scripts/relocate_python_sysconfig.py" \
  "$STAGE/python" "$PYTHON_BASE"

# vLLM: stock committed Python sources plus its already-validated CPU extension.
/usr/bin/rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' \
  "$SOURCE_ROOT/vllm/vllm/" "$SITE/vllm/"
/usr/bin/ditto "$VLLM_SOURCE/vllm/_C.abi3.so" "$SITE/vllm/_C.abi3.so"
# vcs-versioning generates this ignored module beside the validated extension.
# Keep the generated version/commit identity paired with that compiled vLLM.
[ -f "$VLLM_SOURCE/vllm/_version.py" ] || {
  echo "The validated vLLM build has no generated _version.py" >&2
  exit 2
}
/usr/bin/ditto "$VLLM_SOURCE/vllm/_version.py" "$SITE/vllm/_version.py"

# Triton-CPU: committed common Python sources and committed CPU backend. The
# compiler extension is the binary exercised by the performance baseline.
/usr/bin/rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' \
  "$SOURCE_ROOT/triton-cpu/python/triton/" "$SITE/triton/"
mkdir -p "$SITE/triton/backends/cpu" "$SITE/triton/language/extra/cpu"
/usr/bin/rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' \
  "$SOURCE_ROOT/triton-cpu/third_party/cpu/backend/" \
  "$SITE/triton/backends/cpu/"
/usr/bin/rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' \
  "$SOURCE_ROOT/triton-cpu/third_party/cpu/language/cpu/" \
  "$SITE/triton/language/extra/cpu/"
/usr/bin/rsync -aL --delete --exclude '__pycache__/' --exclude '*.pyc' \
  "$TRITON_RUNTIME_PACKAGE/_C/" "$SITE/triton/_C/"
"$BUILD_PYTHON" "$ROOT/scripts/configure_triton_cpu_distribution.py" "$SITE"

# FlagGems owns all Q4/W8/GDN implementation code.
/usr/bin/rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' \
  "$SOURCE_ROOT/FlagGems/src/flag_gems/" "$SITE/flag_gems/"

# Build the thin vLLM integration as a pure-Python wheel so setuptools-scm
# writes immutable version metadata even though the Git export has no .git.
mkdir -p "$PLUGIN_WHEEL_ROOT" "$PLUGIN_INSTALL_ROOT"
env -u VLLM_VENDOR \
  SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+g2ccd0485a \
  "$BUILD_PYTHON" -m pip wheel --no-deps --no-build-isolation \
  "$SOURCE_ROOT/vllm-plugin-FL" --wheel-dir "$PLUGIN_WHEEL_ROOT"
PLUGIN_WHEEL=$(find "$PLUGIN_WHEEL_ROOT" -maxdepth 1 -type f \
  -name 'vllm_plugin_fl-*.whl' -print -quit)
[ -n "$PLUGIN_WHEEL" ] || { echo "vllm-plugin-FL wheel was not built" >&2; exit 2; }
"$BUILD_PYTHON" -m pip install --no-deps --no-index \
  --target "$PLUGIN_INSTALL_ROOT" "$PLUGIN_WHEEL"
/usr/bin/rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' \
  "$PLUGIN_INSTALL_ROOT/vllm_fl/" "$SITE/vllm_fl/"
find "$SITE" -maxdepth 1 -type d -name 'vllm_plugin_fl-*.dist-info' \
  -exec /bin/rm -rf -- {} +
PLUGIN_METADATA=$(find "$PLUGIN_INSTALL_ROOT" -maxdepth 1 -type d \
  -name 'vllm_plugin_fl-*.dist-info' -print -quit)
[ -n "$PLUGIN_METADATA" ] || { echo "vllm-plugin-FL metadata is missing" >&2; exit 2; }
/usr/bin/ditto "$PLUGIN_METADATA" "$SITE/$(basename "$PLUGIN_METADATA")"

# Remove every retired standalone integration after copying the development
# environment. They must not be importable from a public Runtime.
/bin/rm -rf -- "$SITE/vllm_triton_cpu_qwen35"
find "$SITE" -maxdepth 1 -type d -name 'vllm_triton_cpu_qwen35-*.dist-info' \
  -exec /bin/rm -rf -- {} +
# The Runtime exposes the standard vLLM CLI; old flagos-mac Python CLI copies
# from the build environment are intentionally excluded.
/bin/rm -rf -- "$SITE/flagos_mac"
find "$SITE" -maxdepth 1 -type d \
  \( -name 'flagos_macos_runtime_cli-*.dist-info' \
     -o -name 'flagos_macos_runtime_build_tools-*.dist-info' \) \
  -exec /bin/rm -rf -- {} +
find "$SITE" -path '*/direct_url.json' -delete
# Finder metadata is not part of any Python distribution and can appear in a
# developer site-packages tree between otherwise identical builds.
find "$STAGE" -type f -name '.DS_Store' -delete

# Keep compiler diagnostics independent of the maintainer's checkout paths.
SOURCE_PREFIX_MAP_FLAGS="-ffile-prefix-map=$VLLM_SOURCE=vllm-0.20.2 -ffile-prefix-map=$ROOT=flagos-macos-runtime"

# Build the C++ JIT runtime from its exact locked Git export.
"$CMAKE" -S "$SOURCE_ROOT/libtriton_jit" -B "$NATIVE_JIT_BUILD" \
  -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DBACKEND=CPU -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$BUILD_PYTHON" -DOpenMP_ROOT="$LIBOMP_ROOT" \
  -DCMAKE_CXX_FLAGS="$SOURCE_PREFIX_MAP_FLAGS" \
  -DTRITON_JIT_FMT_TAG=11.1.4 -DTRITON_JIT_BUILD_OPERATORS=OFF \
  -DTRITON_JIT_INSTALL=ON -DBUILD_TESTING=OFF
"$CMAKE" --build "$NATIVE_JIT_BUILD" --parallel 8

# Build the single FlagGems Q4/W8/GDN operator bundle against that runtime.
"$CMAKE" -S "$SOURCE_ROOT/FlagGems/src/flag_gems/csrc/arm" \
  -B "$NATIVE_OPS_BUILD" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$BUILD_PYTHON" \
  -DOpenMP_ROOT="$LIBOMP_ROOT" \
  -DCMAKE_CXX_FLAGS="$SOURCE_PREFIX_MAP_FLAGS" \
  -DTRITON_JIT_ROOT="$SOURCE_ROOT/libtriton_jit" \
  -DTRITON_JIT_BUILD="$NATIVE_JIT_BUILD"
"$CMAKE" --build "$NATIVE_OPS_BUILD" --parallel 8

/usr/bin/ditto "$NATIVE_JIT_BUILD/src/libtriton_jit.dylib" \
  "$STAGE/lib/libtriton_jit.dylib"
# libtriton_jit resolves these Python helpers relative to its installed dylib.
# Shipping them is required: falling back to the build checkout would make the
# Runtime non-relocatable and silently depend on the maintainer's filesystem.
mkdir -p "$STAGE/share/triton_jit/scripts"
/usr/bin/rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' \
  "$SOURCE_ROOT/libtriton_jit/scripts/" \
  "$STAGE/share/triton_jit/scripts/"
/usr/bin/ditto "$NATIVE_OPS_BUILD/libflag_gems_arm_ops.dylib" \
  "$SITE/flag_gems/csrc/arm/libflag_gems_arm_ops.dylib"
/usr/bin/ditto "$LIBOMP_ROOT/lib/libomp.dylib" "$STAGE/lib/libomp.dylib"
/usr/bin/ditto "$LIBOMP_ROOT/include/omp.h" "$STAGE/include/omp.h"

# Relocatable Python extensions use @rpath for OpenSSL/readline.  The embedded
# executable has room for a Runtime/lib rpath but not one rpath per Python
# sub-prefix, so expose those existing images through relative links.  This
# also avoids depending on DYLD_LIBRARY_PATH, which macOS may sanitize.
for library in libssl.3.dylib libcrypto.3.dylib; do
  /bin/ln -s "../python/openssl/lib/$library" "$STAGE/lib/$library"
done
for library in libreadline.8.dylib libreadline.8.2.dylib; do
  /bin/ln -s "../python/readline/lib/$library" "$STAGE/lib/$library"
done

# Keep one physical OpenMP image. Torch's expected relative location resolves
# to the Runtime-owned dylib.
/bin/rm -f -- "$SITE/torch/lib/libomp.dylib"
/bin/ln -s ../../../../../../lib/libomp.dylib "$SITE/torch/lib/libomp.dylib"

/usr/bin/ditto "$ROOT/bin/vllm" "$STAGE/bin/vllm"
/usr/bin/ditto "$ROOT/profiles/m5-pro.env" \
  "$STAGE/share/flagos/m5-pro.env"
/usr/bin/ditto "$ROOT/runtime-manifest.json" \
  "$STAGE/share/flagos/runtime-manifest.json"
chmod 755 "$STAGE/bin/vllm"

"$BUILD_PYTHON" "$ROOT/scripts/relocate_macho.py" "$STAGE"

# A fresh Runtime must not spend minutes compiling the standard vLLM CLI on
# first launch.  Precompile only the inference stack, using a reproducible
# relocatable source prefix and the exact Python version shipped above.
BYTECODE_PACKAGES=(
  torch vllm triton flag_gems vllm_fl transformers fastapi anyio starlette
  pydantic safetensors uvicorn uvloop
)
BYTECODE_PATHS=()
for package in "${BYTECODE_PACKAGES[@]}"; do
  [ ! -d "$SITE/$package" ] || BYTECODE_PATHS+=("$SITE/$package")
done
"$BUILD_PYTHON" -m compileall -q --invalidation-mode checked-hash \
  -s "$STAGE" -p flagos-runtime "${BYTECODE_PATHS[@]}"

"$BUILD_PYTHON" "$ROOT/scripts/hash_tree.py" \
  "$STAGE" "$STAGE/share/flagos/runtime-files.sha256.json"

mkdir -p "$ROOT/artifacts"
ARCHIVE="$ROOT/artifacts/flagos-runtime-$VERSION-darwin-arm64-m5pro.tar.gz"
COPYFILE_DISABLE=1 /usr/bin/tar -C "$BUILD_ROOT" -czf "$ARCHIVE" \
  "$(basename "$STAGE")"
(cd "$ROOT/artifacts" && /usr/bin/shasum -a 256 \
  "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
echo "$ARCHIVE"
