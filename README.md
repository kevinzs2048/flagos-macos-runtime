# FlagOS Express Runtime for macOS

One native Apple M5 Pro Runtime for validated W4A8 and W8A8 models through the
standard vLLM CLI. This release line started with the published Qwen3.8-27B
W4A8 Runtime at `v0.1.0-alpha.1` and extends the same source and ABI lineage to
MiniCPM5-2.6B W4A8 G128 and channel-wise W8A8. It packages the validated Python
environment, vLLM 0.20.2, Triton CPU, FlagGems, vLLM-Plugin-FL,
libtriton_jit and the required native libraries. Model weights are downloaded
separately; a model release does not create a second Runtime fork.

This developer release targets Mac17,9 / Apple M5 Pro / 64 GiB. It uses the
Arm CPU only; Metal is not used. The production G128 route uses SDOT/I8MM and
does not require SME2.

## Validated batch-one performance

Tested with a real OpenAI-compatible `vllm serve`, concurrency 1, a 512-token
prompt and 128 generated tokens. Prefix caching was disabled, one complete
prime request was discarded, and each of three retained samples followed a
90-second idle interval.

| Model | PP512 | TG128 | Total 512+128 |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B W4A8 G128 | 76.47 tok/s | 13.07 tok/s | 38.98 tok/s |
| MiniCPM5-2.6B W4A8 G128 | 942.63 tok/s | 90.07 tok/s | 327.80 tok/s |
| MiniCPM5-2.6B W8A8 Channel | 1166.56 tok/s | 65.58 tok/s | 268.29 tok/s |

The Qwen values are the retained thermally cooled acceptance medians for this
Runtime lineage. Its public `v0.1.0-alpha.1` model card records the earlier
74.93/12.73/38.07 tok/s release measurements. The current branch preserves the
Qwen GDN, G128 and W8 `lm_head` routes while adding the MiniCPM kernels.

Under the same HTTP workload, llama.cpp built with KleidiAI and its default
runtime kernel selection measured Q4_0 at 776.53/74.21/269.66 tok/s and Q8_0
at 769.51/51.94/205.74 tok/s. See
[`benchmarks/minicpm5-express-comparison-20260822.json`](benchmarks/minicpm5-express-comparison-20260822.json)
for the retained samples, definitions and build evidence. The quantization
formats are not numerically identical, so this is a serving comparison rather
than a claim that the checkpoints have identical quantization error.

## Supported model weights

Install the ModelScope CLI. The Runtime archive itself does not contain model
weights.

```bash
python3 -m pip install --user modelscope

MODEL_REPO_QWEN="FlagRelease/Qwen3.8-27B-W4A8-arm-FlagOS-Express"
MODEL_REPO_MINICPM="<MiniCPM5 FlagOS Express ModelScope repository ID>"

modelscope download --model "$MODEL_REPO_QWEN" \
  --local_dir "$HOME/Models/Qwen3.8-27B-W4A8-arm-FlagOS-Express"
modelscope download --model "$MODEL_REPO_MINICPM" \
  --local_dir "$HOME/Models/MiniCPM5-2.6B-arm-FlagOS-Express"
```

## Install a prebuilt Runtime

```bash
VERSION=0.1.0-alpha.2
BASE="https://github.com/kevinzs2048/flagos-macos-runtime/releases/download/v$VERSION"

curl -fLO "$BASE/install.sh"
curl -fLO "$BASE/install.sh.sha256"
shasum -a 256 -c install.sh.sha256
bash install.sh

export PATH="$HOME/Library/FlagOS/current/bin:$PATH"
vllm --version
```

For a locally built archive:

```bash
bash install.sh \
  --asset artifacts/flagos-runtime-0.1.0-alpha.2-darwin-arm64-m5pro.tar.gz
export PATH="$HOME/Library/FlagOS/current/bin:$PATH"
```

The installer uses no `sudo`. It verifies the archive checksum, platform,
required Arm features and the packaged vLLM import before activating the
Runtime. The prebuilt Runtime is downloaded as independently checksummed
50 MiB parts (four downloads at a time) and reconstructed transparently; this
avoids unreliable long-lived GitHub upload/download connections. The default
install root is `~/Library/FlagOS`; paths containing whitespace are rejected
because PyTorch Inductor cannot compile its CPU sampler against them.

## Run a supported model

The same Runtime serves Qwen and MiniCPM. The MiniCPM publication contains
validated W4A8 G128 and channel-wise W8A8 checkpoint variants. Select the
checkpoint directory for the variant being served. W4A8 example:

```bash
MODEL="$HOME/Models/MiniCPM5-2.6B-arm-FlagOS-Express/<W4A8 checkpoint>"
NAME=minicpm5-w4a8

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name "$NAME" \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --language-model-only \
  --generation-config vllm \
  --distributed-executor-backend uni \
  --disable-log-stats \
  --compilation-config '{"mode":3}'
```

For W8A8, select the W8A8 checkpoint under the same MiniCPM publication and use
`NAME=minicpm5-w8a8`. No GPU or Metal memory option is required; this is the
Arm CPU path.

Qwen3.8-27B uses the same launcher and Runtime. Its validated release command
keeps the text-only and reasoning-parser options explicit:

```bash
MODEL="$HOME/Models/Qwen3.8-27B-W4A8-arm-FlagOS-Express"

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen38 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1 \
  --enforce-eager \
  --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --distributed-executor-backend uni \
  --disable-log-stats
```

The launcher is intentionally thin: it sets the self-contained Runtime paths,
sources `share/flagos/m5-pro.env` and dispatches to the standard vLLM CLI. The
version flag has a metadata-only fast path; inference arguments are passed
through unchanged. There is no separate FlagOS inference API and no `doctor`,
benchmark or model-download product command.

`runtime-manifest.json` is the authoritative supported-model registry. See
[MODEL_SUPPORT.md](MODEL_SUPPORT.md) for the rule used to add future models to
this branch without cloning or replacing the Runtime repository.

## Build everything locally

The one-command build consumes a clean vLLM v0.20.2 build environment and a
relocatable libomp prefix. It fetches and verifies the exact component commits
recorded by `sources.lock.json`; clean local checkout overrides are supported
for offline and developer builds. The unified Express component commits are
locally committed but not yet published, so local source overrides are
required until maintainers publish the refs. The vLLM CPU extension is rebuilt
with the audited Darwin OpenMP and AOT-cache compatibility patches; the
upstream source export is not modified.

```bash
./build.sh /path/to/vllm-0.20.2 /path/to/relocatable-libomp
```

Equivalent explicit environment form:

```bash
export FLAGOS_VLLM_SOURCE=/path/to/vllm-0.20.2
export FLAGOS_LIBOMP_ROOT=/path/to/relocatable-libomp
export FLAGOS_TRITON_SOURCE=/path/to/triton-cpu
export FLAGOS_FLAGGEMS_SOURCE=/path/to/FlagGems
export FLAGOS_PLUGIN_SOURCE=/path/to/vllm-plugin-FL
export FLAGOS_LIBTRITON_JIT_SOURCE=/path/to/libtriton_jit
./build.sh
```

The build produces:

- `artifacts/flagos-runtime-0.1.0-alpha.2-darwin-arm64-m5pro.tar.gz` —
  validated self-contained end-user Runtime; Release metadata also produces
  its 50 MiB transport parts and parts manifest.
- `artifacts/flagos-wheelhouse-0.1.0-alpha.2-cp311-darwin-arm64.tar.gz` —
  four component wheels for developers.
- SHA256 sidecars and `SHA256SUMS`.

See [BUILDING.md](BUILDING.md) for the step-by-step process and local source
override variables.
