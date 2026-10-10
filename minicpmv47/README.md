# FlagOS MiniCPM V4.7 on Apple M5 and Cix P1

This release packages vLLM 0.24.0+cpu, Torch 2.11.0, Transformers 5.15.1,
FlagGems, FlagTree CPU (Triton 3.7.2), and vllm-plugin-FL. Component commits
are pinned in `sources.lock.json`. The vLLM core is unchanged. Model and
processor adaptation lives in the plugin; operators live in FlagGems.

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
gzip -dc flagos-minicpmv47-cix-p1-20261011.docker.tar.gz | sudo docker load
sudo docker run --rm --network host --cpuset-cpus 0,1,6-11 \
  -v /absolute/path/to/35b-W4A8-G32:/model:ro \
  -v /absolute/path/to/writable-cache:/cache \
  -e FLAGOS_PROFILE=35b \
  flagos/minicpm-v47-cix-p1:0.3.0-20261011 \
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

```sh
python3 minicpmv47/scripts/stage_runtime.py \
  --runtime /verified/runtime --triton /verified/triton \
  --installed /verified/installed-prefix --wheels /verified/two-wheels \
  --cpu-tle-source /locked/flagtree/third_party/cpu/language/cpu/tle_ops.py \
  --output /new/staging/runtime
# Linux also requires --vllm /verified/vllm-package-directory.
# macOS: relocate and ad-hoc sign the staged Mach-O files, then validate.
python3 scripts/relocate_macho.py /new/staging/runtime
python3 minicpmv47/scripts/export_macos.py --runtime /new/staging/runtime \
  --output /new/artifacts/flagos-minicpmv47-macos-arm64-20261011.tar.gz
# Linux: put Dockerfile.cix-p1 next to runtime/ in a build context.
sudo docker build --network host \
  --build-arg FLAGGEMS_COMMIT=ae1a81c716445ff71d0a8b2a979fa9ded6f3d2c7 \
  --build-arg FLAGTREE_COMMIT=36133df9540cdd9eb5b09209ee41ce1697b27126 \
  --build-arg PLUGIN_COMMIT=f0de702335902facfffe625c0690f7a3df7f617d \
  -t flagos/minicpm-v47-cix-p1:0.3.0-20261011 /new/staging
```

The legacy root build scripts and release lock target an earlier runtime; they
are not the build entrypoint for this MiniCPM V4.7 release.
