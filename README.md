# MiniCPM5 FlagOS Express Runtime for macOS

Native Apple M5 Pro Runtime for W4A8 G128 and channel-wise W8A8 inference with
the standard vLLM CLI. The validated text-only models are MiniCPM5-2.6B W4A8
G128 and MiniCPM5-2.6B channel-wise W8A8. It packages the validated Python
environment, vLLM 0.20.2, Triton CPU, FlagGems, vLLM-Plugin-FL,
libtriton_jit and the required native libraries. Model weights are downloaded
separately.

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
| MiniCPM5-2.6B W4A8 G128 | 942.63 tok/s | 90.07 tok/s | 327.80 tok/s |
| MiniCPM5-2.6B W8A8 Channel | 1166.56 tok/s | 65.58 tok/s | 268.29 tok/s |

Under the same HTTP workload, llama.cpp built with KleidiAI and its default
runtime kernel selection measured Q4_0 at 776.53/74.21/269.66 tok/s and Q8_0
at 769.51/51.94/205.74 tok/s. See
[`benchmarks/minicpm5-express-comparison-20260822.json`](benchmarks/minicpm5-express-comparison-20260822.json)
for the retained samples, definitions and build evidence. The quantization
formats are not numerically identical, so this is a serving comparison rather
than a claim that the checkpoints have identical quantization error.

## Download the model weights

Install the ModelScope CLI and set each repository ID after the two model
repositories are published. The Runtime archive itself does not contain model
weights.

```bash
python3 -m pip install --user modelscope

MODEL_REPO_W4="<MiniCPM5 W4A8 G128 ModelScope repository ID>"
MODEL_REPO_W8="<MiniCPM5 W8A8 Channel ModelScope repository ID>"

modelscope download --model "$MODEL_REPO_W4" \
  --local_dir "$HOME/Models/MiniCPM5-2.6B-W4A8-G128-FlagOS"
modelscope download --model "$MODEL_REPO_W8" \
  --local_dir "$HOME/Models/MiniCPM5-2.6B-W8A8-Channel-FlagOS"
```

## Install a prebuilt Runtime

```bash
VERSION=0.1.0-alpha.1
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
  --asset artifacts/flagos-runtime-0.1.0-alpha.1-darwin-arm64-m5pro.tar.gz
export PATH="$HOME/Library/FlagOS/current/bin:$PATH"
```

The installer uses no `sudo`. It verifies the archive checksum, platform,
required Arm features and the packaged vLLM import before activating the
Runtime. The prebuilt Runtime is downloaded as independently checksummed
50 MiB parts (four downloads at a time) and reconstructed transparently; this
avoids unreliable long-lived GitHub upload/download connections. The default
install root is `~/Library/FlagOS`; paths containing whitespace are rejected
because PyTorch Inductor cannot compile its CPU sampler against them.

## Run MiniCPM5-2.6B

The same Runtime serves both validated MiniCPM checkpoints. Select either the
W4A8 G128 or channel-wise W8A8 directory and change only the served name.

```bash
MODEL="$HOME/Models/MiniCPM5-2.6B-W4A8-G128-FlagOS"
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

For W8A8, use `MiniCPM5-2.6B-W8A8-Channel-FlagOS` and
`NAME=minicpm5-w8a8`. No GPU or Metal memory option is required; this is the
Arm CPU path.

The launcher is intentionally thin: it sets the self-contained Runtime paths,
sources `share/flagos/m5-pro.env` and dispatches to the standard vLLM CLI. The
version flag has a metadata-only fast path; inference arguments are passed
through unchanged. There is no separate FlagOS inference API and no `doctor`,
benchmark or model-download product command.

## Build everything locally

The one-command build consumes a clean vLLM v0.20.2 build environment and a
relocatable libomp prefix. It fetches and verifies the exact component commits
recorded by `sources.lock.json`; clean local checkout overrides are supported
for offline and developer builds. The MiniCPM Express component commits are
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

- `artifacts/flagos-runtime-0.1.0-alpha.1-darwin-arm64-m5pro.tar.gz` —
  validated self-contained end-user Runtime; Release metadata also produces
  its 50 MiB transport parts and parts manifest.
- `artifacts/flagos-wheelhouse-0.1.0-alpha.1-cp311-darwin-arm64.tar.gz` —
  four component wheels for developers.
- SHA256 sidecars and `SHA256SUMS`.

See [BUILDING.md](BUILDING.md) for the step-by-step process and local source
override variables.
