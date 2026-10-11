# FlagOS MiniCPM V4.7 on Apple M5 and Cix P1

This release packages vLLM 0.24.0+cpu, Torch 2.11.0, Transformers 5.15.1,
FlagGems, FlagTree CPU (Triton 3.7.2), and vllm-plugin-FL. Component commits
are pinned in `sources.lock.json`. The vLLM core is unchanged. Model and
processor adaptation lives in the plugin; operators live in FlagGems.

Package qualification and remaining performance limitations are recorded in
[VALIDATION.md](VALIDATION.md).
The P1 investigation and control experiments are documented in
[P1_REGRESSION.md](P1_REGRESSION.md).

Supported checkpoints:

* MiniCPM-V-4.7-35B-A3B: W4A8 G32, with W8A8 output head.
* MiniCPM-V-4.7-1B: W8A8.

Weights, quantization-generation scripts, DSpark, and evaluation datasets are
not included. Vision remains BF16. Use the matching FlagOS quantized checkpoints.
These packages are CPU runtimes, not Metal GPU runtimes.

## macOS arm64 native runtime

Use the native archive on Apple M5 Pro. This preserves the Darwin CPU/SME2
runtime used for qualification. A Docker Desktop Linux VM has a different CPU
and OS environment and is not the qualified Mac performance path.

```sh
mkdir -p "$HOME/flagos-minicpmv47"
tar -xzf flagos-minicpmv47-macos-arm64-20261011.tar.gz -C "$HOME/flagos-minicpmv47"
RT="$HOME/flagos-minicpmv47"
"$RT/bin/flagos-vllm" --profile 1b --version
"$RT/bin/flagos-vllm" --profile 1b serve /absolute/path/to/1b-W8A8 \
  --trust-remote-code --dtype bfloat16 --max-model-len 4096 \
  --max-num-batched-tokens 512 --max-num-seqs 1 \
  --worker-cls vllm_fl.worker.cpu_phase_worker.CPUPhaseWorker \
  --distributed-executor-backend uni --enforce-eager \
  --mamba-ssm-cache-dtype float32 --generation-config vllm \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --host 127.0.0.1 --port 8000
```

For 35B use `--profile 35b` and its checkpoint path. Cache defaults to
`$RT/cache`; set `FLAGOS_CACHE_DIR` to a writable location to keep caches separate.
The first request compiles kernels and may create a large expert packing cache.
Keep sufficient free disk and memory. Apple Command Line Tools (`clang`) are
needed for JIT linking. The archive supplies CPython and OpenMP; no Homebrew
runtime library path is required.

## Cix P1 Linux arm64 Docker image

```sh
gzip -dc flagos-minicpmv47-cix-p1-0.3.1-20261011.docker.tar.gz | sudo docker load
sudo docker run --rm --network host --cpuset-cpus 0,1,6-11 \
  -v /absolute/path/to/35b-W4A8-G32:/model:ro \
  -v /absolute/path/to/writable-cache:/cache \
  -e FLAGOS_PROFILE=35b \
  flagos/minicpm-v47-cix-p1:0.3.1-20261011 \
  serve /model --trust-remote-code --dtype bfloat16 --max-model-len 4096 \
  --max-num-batched-tokens 512 --max-num-seqs 1 \
  --worker-cls vllm_fl.worker.cpu_phase_worker.CPUPhaseWorker \
  --distributed-executor-backend uni --enforce-eager \
  --kv-cache-memory-bytes 268435456 --mamba-ssm-cache-dtype float32 \
  --generation-config vllm --limit-mm-per-prompt '{"image":1,"video":0}' \
  --host 127.0.0.1 --port 8000
```

For 1B change `FLAGOS_PROFILE=1b` and the checkpoint mount. CPU placement above
matches the tested P1 core numbering; verify topology before reusing it on
another board. Host networking also works where Docker bridge forwarding is
disabled. No GPU devices or privileged container are needed.
The tested P1 KV budget is 256 MiB. Keep `VLLM_CPU_KVCACHE_SPACE` unset: this
deprecated GiB environment setting overrides `--kv-cache-memory-bytes` in the
CPU platform configuration. The launcher removes an empty inherited value.

## Execution policy

| Platform | Model | Prefill threads | Decode threads |
|---|---|---:|---:|
| M5 Pro | 35B | 18 | 8 |
| M5 Pro | 1B | 8 | 8 |
| Cix P1 | 35B | 8 | 6 |
| Cix P1 | 1B | 8 | 4 |

Profiles set defaults; explicit environment values override them. The packaged
CPUPhaseWorker switches threads at prefill/decode boundaries. Native fused
operators are the default; Gemma RMSNorm uses ordinary CPU Triton. To select
CPU TLE Raw for supported native operators, set both
`FL_ARM_OPERATOR_BACKEND=raw` and `FLAGGEMS_ARM_GDN_BACKEND=raw`.
TLE calls retain the C++ arithmetic implementation; host preparation, weight
packing, and framework operations are not all Triton kernels. No universal
whole-model 0.5% regression claim is made.

On the tested P1, both profiles use `OMP_PROC_BIND=spread`. The 1B profile's
four decode workers occupy all four big-core frequency policies; the 35B
profile spreads its six decode workers. The 1B profile also keeps
warm allocator pages with `MIMALLOC_PURGE_DELAY=-1` and
`TCMALLOC_RELEASE_RATE=0`. Torch tensor storage uses embedded mimalloc, so
TCMalloc alone does not control tensor page release. The 35B model uses
`MIMALLOC_PURGE_DELAY=100` milliseconds with `TCMALLOC_RELEASE_RATE=0`, keeping
explicit startup/prefill cleanup enabled. No profile changes host governors.

For controlled 1B benchmarks on this P1, the host can explicitly hold each
big-core frequency policy's minimum at its existing maximum while a command
runs. This reduced the observed prefill gap; residual timing variation is
documented in the regression report. The host tool preserves the governor,
records the original settings, and restores them after command failure or
SIGINT/SIGTERM. It waits for asynchronous sysfs readback to settle.

```sh
sudo python3 minicpmv47/p1_performance_run.py \
  --state /absolute/path/to/new-frequency-journal.json -- \
  docker run ...
```

Run this tool on the Linux host. The container needs no extra capabilities.
If the controller is killed ungracefully, the same journal can be restored with
`sudo python3 minicpmv47/p1_performance_run.py --state /path/to/journal.json --restore`.
The 35B control did not benefit from this frequency floor; retain its default
host policy. This host utility is deployment/benchmark tooling, not a model
adapter or an arithmetic kernel.

## Validate an extracted package

Use a fresh writable cache and a new report filename. Reports are flushed and
atomically saved after every request, so completed results survive interruption.

```sh
FLAGOS_CACHE_DIR=/absolute/path/to/fresh-cache \
  "$RT/bin/flagos-python" --profile 1b \
  "$RT/share/flagos-minicpmv47/validation/package_ops.py" \
  --prefix "$RT/python/lib/python3.11/site-packages" --output operators.json
"$RT/bin/flagos-python" --profile 1b \
  "$RT/share/flagos-minicpmv47/validation/model_smoke.py" \
  --model /absolute/path/to/1b-W8A8 --worker phase --vision --repeats 3 \
  --package-site "$RT/python/lib/python3.11/site-packages" --output model.json
```

For Docker override the entrypoint to `/opt/flagos-runtime/bin/flagos-python`,
use `/opt/flagos-runtime/venv/lib/python3.11/site-packages` as the prefix, and
mount a writable results directory. Tests check native/Raw/Triton operator
results, package-only imports, text, image and OCR generation, and PP512/TG128.
TG128 is `127 / (last-token time - first-token time)`. Total is
`640 / (prefill time + 127 decode intervals)` for one combined request.
The current harness measures retained warm heap state by default. Use
`--heap-policy trim` to reproduce the earlier explicit cleanup before every
benchmark request; do not silently mix these protocols. It records allocator
and OpenMP settings, interpreter provenance, governors, medians and ranges.

## Build policy and provenance

`scripts/stage_runtime.py` creates a new output directory from a verified
dependency runtime, matched FlagTree compiler artifacts, two component wheels,
and installed FlagGems native libraries. It never modifies the source runtime.
This is a snapshot release, not a fresh source build of Torch/vLLM/LLVM. The
source lock and staging manifest record the component commits and binary hashes.
The Linux Docker base is the existing CPU image identified in the lock; staging
replaces its Python packages with the tested versions and clears obsolete base
experiment flags. The exported image is independently loadable without that
base image being preinstalled.
JIT launchers use GCC on P1, matching the verified host runtime and Torch's
GNU OpenMP. Inheriting the earlier base image's Clang configuration can load a
second OpenMP runtime and give different thread behavior. Use a fresh kernel
cache when changing compiler selection.
Linux staging also requires a CPython snapshot created with the interpreter
used for host qualification. Python version alone is insufficient: the older
Docker base's custom CPython 3.11.2 build differed from the qualified host build.
The packaged launcher isolates the snapshot's stdlib, headers, and library paths.

```sh
python3 minicpmv47/scripts/stage_runtime.py \
  --runtime /verified/runtime --triton /verified/triton \
  --installed /verified/installed-prefix --wheels /verified/two-wheels \
  --cpu-tle-source /locked/flagtree/third_party/cpu/language/cpu/tle_ops.py \
  --output /new/staging/runtime
# Linux: run this with the qualified host interpreter first:
/verified/host/python3.11 minicpmv47/scripts/snapshot_cpython.py \
  --output /new/qualified-cpython
# Linux staging also requires --vllm /verified/vllm-package-directory
# and --cpython /new/qualified-cpython.
# macOS: relocate and ad-hoc sign the staged Mach-O files, then validate.
python3 scripts/relocate_macho.py /new/staging/runtime
python3 minicpmv47/scripts/export_macos.py --runtime /new/staging/runtime \
  --output /new/artifacts/flagos-minicpmv47-macos-arm64-20261011.tar.gz
# Linux: put Dockerfile.cix-p1 next to runtime/ in a build context.
sudo docker build --network host \
  --build-arg FLAGGEMS_COMMIT=ae1a81c716445ff71d0a8b2a979fa9ded6f3d2c7 \
  --build-arg FLAGTREE_COMMIT=36133df9540cdd9eb5b09209ee41ce1697b27126 \
  --build-arg PLUGIN_COMMIT=f0de702335902facfffe625c0690f7a3df7f617d \
  -t flagos/minicpm-v47-cix-p1:0.3.1-20261011 /new/staging
```

The legacy root build scripts and release lock target an earlier runtime; they
are not the build entrypoint for this MiniCPM V4.7 release.
