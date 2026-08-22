# Building the FlagOS macOS multi-model Runtime

The build creates one self-contained Apple M5 Pro Runtime for every model in
`runtime-manifest.json` and exposes the standard `vllm` CLI. It applies the
audited Darwin OpenMP and stock-Inductor AOT compatibility patches to an
isolated copy of vLLM and does not include model weights.

## Required inputs

1. A clean vLLM v0.20.2 checkout and a Python 3.11 build environment at
   `.venv311`. The CPU extension is rebuilt from source.
2. A relocatable macOS libomp prefix containing `include/omp.h` and
   `lib/libomp.dylib`.
3. Network access to fetch the exact published component commits, or clean
   local checkouts supplied through the source override variables below for an
   offline build.

No input path is hardcoded. The two required paths may be anywhere:

```bash
./build.sh /path/to/vllm-0.20.2 /path/to/relocatable-libomp
```

If build tools are outside `.venv311`, set them explicitly:

```bash
export FLAGOS_VLLM_SOURCE=/path/to/vllm-0.20.2
export FLAGOS_LIBOMP_ROOT=/path/to/relocatable-libomp
export FLAGOS_BUILD_PYTHON=/path/to/python3.11
export FLAGOS_CMAKE=/path/to/cmake
export FLAGOS_NINJA=/path/to/ninja
./build.sh
```

## What the build does

`build.sh` runs `scripts/package_release.sh`, which:

1. Materializes all five exact Git commits and verifies both commit and tree
   IDs against `sources.lock.json`.
2. Copies and relocates the validated Python distribution and dependencies.
3. Installs stock vLLM 0.20.2 Python code, rebuilds its CPU extension with
   Darwin OpenMP enabled, and adds the pinned vLLM-Plugin-FL adapter.
4. Installs the pinned Triton CPU and FlagGems Python sources.
5. Rebuilds libtriton_jit and the FlagGems Q4/W8/GDN native operator bundle.
6. Adds the shared M5 Pro performance profile, supported-model manifest and
   standard `vllm` launcher.
7. Relocates all Mach-O dependencies and consolidates libomp.
8. Creates the Runtime archive and four-wheel developer wheelhouse.
9. Verifies checksums, archive paths, source provenance, Mach-O dependencies,
   JIT helper imports, vLLM import/version and all nine required native ops.
10. Installs the archive into an isolated temporary user root and verifies the
   standard `vllm --version` and `vllm serve --help` workflow.

Every long-running build/test command is bounded by the calling release job;
network downloads in the installer use explicit connection and total timeouts.

## Individual steps

```bash
export FLAGOS_VLLM_SOURCE=/path/to/vllm-0.20.2
export FLAGOS_LIBOMP_ROOT=/path/to/relocatable-libomp

./scripts/materialize_sources.py
./scripts/build_runtime.sh
./scripts/build_wheelhouse.sh
./scripts/update_release_metadata.py
./scripts/verify_release.py
./scripts/test_install_workflow.sh
./scripts/test_wheelhouse.sh
```

## Local source overrides

Published non-vLLM components are fetched from their exact locked commits and
cached below the ignored `build/source-cache/` directory. For a fully offline
build, point every component at a local checkout:

```bash
export FLAGOS_VLLM_SOURCE=/path/to/vllm-0.20.2
export FLAGOS_TRITON_SOURCE=/path/to/triton-cpu
export FLAGOS_FLAGGEMS_SOURCE=/path/to/FlagGems
# The following two are optional when their locked commits are fetchable.
export FLAGOS_PLUGIN_SOURCE=/path/to/vllm-plugin-FL
export FLAGOS_LIBTRITON_JIT_SOURCE=/path/to/libtriton_jit

./build.sh
```

Every overridden tree must be clean and match both the commit and tree recorded
in `sources.lock.json`; otherwise the build stops before compilation. Generated
materialized sources live under the ignored `build/sources/` directory and are
never committed.
