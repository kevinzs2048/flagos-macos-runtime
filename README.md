# FlagOS macOS W4A8/W8A8 Runtime

Native Apple M5 Pro Runtime for W4A8 G128 and channel-wise W8A8 inference with
the standard vLLM CLI. The validated text-only models include Qwen3.8-27B W4A8
and MiniCPM5-2.6B W4A8/W8A8. It packages the validated Python environment,
vLLM 0.20.2, Triton CPU, FlagGems, vLLM-Plugin-FL, libtriton_jit and the
required native libraries. Model weights are downloaded separately.

This developer release targets Mac17,9 / Apple M5 Pro / 64 GiB. It uses the
Arm CPU only; Metal is not used. The production G128 route uses SDOT/I8MM and
does not require SME2.

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

## Run Qwen3.8-27B

Download the model from its ModelScope or Hugging Face repository, then use the
normal vLLM command. The packaged `vllm` launcher automatically applies the
validated M5 Pro FlagGems/Triton/OpenMP profile and strict kernel routing.

```bash
MODEL="$HOME/Models/Qwen3.8-27B-W4A8-GPTQ-G128-packed"

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

## Build everything locally

The one-command build consumes a clean vLLM v0.20.2 build environment and a
relocatable libomp prefix. It fetches and verifies the exact component commits
recorded by `sources.lock.json`; clean local checkout overrides are supported
for offline and developer builds. The current Triton CPU and FlagGems candidate
commits are locally committed but not yet published, so those two overrides are
required until their branches are pushed. The vLLM CPU extension is rebuilt
with the audited Darwin OpenMP and AOT-cache compatibility patches; the
upstream source export is not modified.

```bash
./build.sh /path/to/vllm-0.20.2 /path/to/relocatable-libomp
```

Equivalent explicit environment form:

```bash
export FLAGOS_VLLM_SOURCE=/path/to/vllm-0.20.2
export FLAGOS_LIBOMP_ROOT=/path/to/relocatable-libomp
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
