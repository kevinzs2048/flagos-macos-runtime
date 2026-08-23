# FlagOS Express Runtime for Apple M5 Pro

FlagOS Express is a native, CPU-only macOS Runtime for serving validated W4A8
and W8A8 language models through the standard vLLM CLI. One Runtime supports
both Qwen3.8-27B W4A8 G128 and MiniCPM5-2.6B W4A8 G128/W8A8 Channel; model
weights are distributed separately and do not require model-specific Runtime
forks.

The Runtime packages Python 3.11, PyTorch, vLLM 0.20.2, Triton CPU, FlagGems,
vLLM-Plugin-FL, libtriton_jit and their native dependencies. The production
Arm kernels use SDOT and I8MM. Metal, GPU offload, Docker and virtual machines
are not used.

## Platform and scope

| Item | Validated configuration |
| --- | --- |
| Host | Mac17,9 / Apple M5 Pro / 64 GiB unified memory |
| OS | macOS 26.5.1, arm64 |
| Required Arm features | DotProd and I8MM |
| Runtime | FlagOS `0.1.0-alpha.2`, vLLM `0.20.2+cpu` |
| Serving mode | Text-only, OpenAI-compatible HTTP API |
| Accelerator | CPU only; Metal disabled |

The installer intentionally rejects other Mac models. This is a validated
developer release, not a claim of compatibility with every Apple Silicon Mac.

## Source and release layout

The source branch, binary release and model weights have separate locations:

| Content | Location |
| --- | --- |
| Unified Runtime source | [`kevinzs2048/flagos-macos-runtime`, branch `minicpm-express`](https://github.com/kevinzs2048/flagos-macos-runtime/tree/minicpm-express) |
| Prebuilt Runtime target | [`v0.1.0-alpha.2` GitHub Release](https://github.com/kevinzs2048/flagos-macos-runtime/releases/tag/v0.1.0-alpha.2) |
| Qwen model weights | [`FlagRelease/Qwen3.8-27B-W4A8-arm-FlagOS-Express`](https://modelscope.cn/models/FlagRelease/Qwen3.8-27B-W4A8-arm-FlagOS-Express) |
| MiniCPM model weights | Distributed separately through the corresponding model publication |

The `v0.1.0-alpha.2` Release URL becomes downloadable after the tag and assets
are published. Until then, use the source branch and locally built artifacts.

## Validated performance

All measurements use a real HTTP server, concurrency 1, exactly 512 input
tokens and 128 generated tokens. Prefix caching was disabled. One full-shape
prime was discarded, followed by three retained samples with a 90-second idle
interval before each sample.

| Model | Prefill PP512 | Decode TG128 | Total 512+128 |
| --- | ---: | ---: | ---: |
| MiniCPM5-2.6B W4A8 G128, FlagOS | 1017.23 tok/s | 90.03 tok/s | 334.22 tok/s |
| MiniCPM5-2.6B Q4_0, llama.cpp + KleidiAI | 780.95 tok/s | 76.37 tok/s | 275.63 tok/s |
| MiniCPM5-2.6B W8A8 Channel, FlagOS | 1134.93 tok/s | 64.33 tok/s | 263.62 tok/s |
| MiniCPM5-2.6B Q8_0, llama.cpp + KleidiAI | 425.36 tok/s | 53.36 tok/s | 178.57 tok/s |
| Qwen3.8-27B W4A8 G128, FlagOS | 76.47 tok/s | 13.07 tok/s | 38.98 tok/s |

The latest llama.cpp Q4 Prefill samples were bimodal, and its Q8 Prefill
working set was colder than in the preceding run. No sample was removed or
replaced. W4A8 versus Q4_0 and W8A8 versus Q8_0 are serving comparisons between
different quantization formats; they do not imply identical quantization
error. Raw samples, metric definitions and build evidence are recorded in
[`benchmarks/minicpm5-express-comparison-20260823.json`](benchmarks/minicpm5-express-comparison-20260823.json).

## Install the prebuilt Runtime

After the `v0.1.0-alpha.2` assets are published, download and verify the
installer before running it:

```bash
VERSION=0.1.0-alpha.2
BASE="https://github.com/kevinzs2048/flagos-macos-runtime/releases/download/v$VERSION"

curl --fail --location --remote-name "$BASE/install.sh"
curl --fail --location --remote-name "$BASE/install.sh.sha256"
shasum -a 256 -c install.sh.sha256
bash install.sh
```

Expose the packaged standard vLLM command in the current shell:

```bash
export PATH="$HOME/Library/FlagOS/current/bin:$PATH"
vllm --version
```

The installer requires no `sudo`. It downloads ten independently checksummed
Runtime parts, four at a time, verifies and reconstructs the 476 MiB archive,
installs it below `~/Library/FlagOS/`, and atomically activates the new version.
The installation path must not contain whitespace because PyTorch Inductor
cannot compile the CPU sampler against such a library path.

To install a locally built archive instead:

```bash
bash install.sh \
  --asset artifacts/flagos-runtime-0.1.0-alpha.2-darwin-arm64-m5pro.tar.gz
export PATH="$HOME/Library/FlagOS/current/bin:$PATH"
```

## Obtain model weights

The Runtime does not contain model weights. Install the ModelScope CLI to
download a published checkpoint:

```bash
python3 -m pip install --user modelscope
```

For Qwen3.8-27B:

```bash
MODEL_REPO="FlagRelease/Qwen3.8-27B-W4A8-arm-FlagOS-Express"
MODEL_DIR="$HOME/Models/Qwen3.8-27B-W4A8-arm-FlagOS-Express"

modelscope download --model "$MODEL_REPO" --local_dir "$MODEL_DIR"
```

For MiniCPM5-2.6B, download the corresponding model publication and keep the
W4A8 and W8A8 checkpoint directories separately. The serving examples below
accept any absolute local path and therefore do not require a placeholder
ModelScope repository ID in the Runtime.

## Serve MiniCPM5-2.6B

### W4A8 G128

```bash
MODEL="/absolute/path/to/MiniCPM5-2.6B-W4A8-G128-FlagOS"

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name minicpm5-w4a8 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --language-model-only \
  --generation-config vllm \
  --distributed-executor-backend uni \
  --disable-log-stats \
  --compilation-config '{"mode":3}'
```

### W8A8 Channel

Use the same command with the W8A8 checkpoint and a different served name:

```bash
MODEL="/absolute/path/to/MiniCPM5-2.6B-W8A8-Channel-FlagOS"

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name minicpm5-w8a8 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --language-model-only \
  --generation-config vllm \
  --distributed-executor-backend uni \
  --disable-log-stats \
  --compilation-config '{"mode":3}'
```

No GPU-memory option is needed. The packaged launcher loads the validated M5
Pro profile, enables vLLM-Plugin-FL and forwards all inference arguments to the
standard vLLM CLI.

## Send a request

In another terminal, use the served model name selected above:

```bash
curl --fail --max-time 300 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minicpm5-w4a8",
    "messages": [
      {"role": "user", "content": "Introduce yourself briefly."}
    ],
    "max_tokens": 128,
    "temperature": 0.6
  }'
```

## Serve Qwen3.8-27B with the same Runtime

The Qwen route inherited from `v0.1.0-alpha.1` remains supported:

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

## Build the Runtime from source

Skip this section when using the prebuilt Runtime. A source build requires
Xcode Command Line Tools, Homebrew, a clean stock vLLM `v0.20.2` checkout with
a Python 3.11 environment at `.venv311`, and a relocatable libomp prefix.

The following is a complete setup from clean source checkouts:

```bash
xcode-select -p
brew install python@3.11 cmake ninja libomp

WORK="$HOME/flagos-express-build"
mkdir -p "$WORK"

git clone --branch v0.20.2 --depth 1 \
  https://github.com/vllm-project/vllm.git "$WORK/vllm-0.20.2"

"$(brew --prefix python@3.11)/bin/python3.11" -m venv \
  "$WORK/vllm-0.20.2/.venv311"
. "$WORK/vllm-0.20.2/.venv311/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  -r "$WORK/vllm-0.20.2/requirements/build/cpu.txt" \
  -r "$WORK/vllm-0.20.2/requirements/cpu.txt"

git clone --branch minicpm-express --single-branch \
  https://github.com/kevinzs2048/flagos-macos-runtime.git \
  "$WORK/flagos-macos-runtime"
cd "$WORK/flagos-macos-runtime"

./build.sh "$WORK/vllm-0.20.2" "$(brew --prefix libomp)"
```

For an immutable release rebuild, replace `--branch minicpm-express` with
`--branch v0.1.0-alpha.2` after the tag is published. The build materializes
and verifies the exact component commits and Git tree IDs recorded in
[`sources.lock.json`](sources.lock.json). Local source overrides are supported
for fully offline builds; see [`BUILDING.md`](BUILDING.md).

The build produces:

- `artifacts/flagos-runtime-0.1.0-alpha.2-darwin-arm64-m5pro.tar.gz`: the
  self-contained end-user Runtime.
- Ten `*.part-NNN` files plus a parts manifest and SHA256 sidecars for GitHub
  Release transport.
- `artifacts/flagos-wheelhouse-0.1.0-alpha.2-cp311-darwin-arm64.tar.gz`: four
  developer component wheels.
- `install.sh`, `install.sh.sha256` and `SHA256SUMS`.

Run the release verifier independently with:

```bash
python3 scripts/verify_release.py
```

The verified alpha.2 archive contains 42,580 hashed files and 529 Mach-O
objects, uses one packaged OpenMP Runtime, and passes direct W4/W8 native
operator numerical smoke tests. See [`RELEASE_ACCEPTANCE.md`](RELEASE_ACCEPTANCE.md)
for the complete acceptance record.

## Runtime architecture

```text
standard vllm CLI
  -> FlagOS M5 Pro profile
  -> vLLM 0.20.2 CPU backend
  -> vLLM-Plugin-FL
  -> FlagGems Triton W4/W8/GDN operators
  -> Triton CPU Arm lowering
  -> libtriton_jit + SDOT/I8MM
```

The MiniCPM optimization does not embed or link KleidiAI/TLE compute kernels.
FlagGems supplies the Triton kernels, Triton CPU lowers the relevant Arm dot
operations, and libtriton_jit provides the launch ABI. The exact repository
commits and their responsibilities are documented in
[`UPSTREAM_STATUS.md`](UPSTREAM_STATUS.md).

`runtime-manifest.json` is the authoritative supported-model registry. The
policy for adding future models without creating another Runtime fork is
documented in [`MODEL_SUPPORT.md`](MODEL_SUPPORT.md).

## Known limitations

- The release is validated only on Mac17,9 / Apple M5 Pro / 64 GiB.
- The packaged model routes are text-only; vision and MTP are not enabled.
- Model weights are not included in the Runtime archive.
- The developer release is unsigned and is not notarized with an Apple
  Developer ID.
- Performance depends on temperature, memory residency and other host load;
  retained benchmark samples are published without outlier substitution.
