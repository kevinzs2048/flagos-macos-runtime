---
library_name: vllm
tags:
- flagos
- macos
- apple-silicon
- w4a8
- moe
- xingchen4
---

# XingChen4 W4A8 G128 — FlagOS for Mac

CPU inference for XingChen4 (`Xing4_0ForCausalLM`) on Apple M5 Pro.
The standard deployment uses the W4A8 target only, without MTP.

## Requirements

- Apple M5 Pro (Mac17,9), 64 GiB memory, native macOS arm64.
- Tested on macOS 26.5.1 with Xcode Command Line Tools.
- Matching FlagOS Runtime: Python 3.11.9, PyTorch 2.11.0, vLLM **0.24.0+cpu**,
  Triton CPU 3.7.2+gitbdada74a, FlagGems 5.0.2.
- Use the supplied Runtime, not a stock vLLM installation.

## Operation steps

### 1. Download the W4A8 model weights

Model repository: [FlagRelease/Xing4.0-29B-A4B-W4A8-arm-FlagOS](https://modelscope.cn/models/FlagRelease/Xing4.0-29B-A4B-W4A8-arm-FlagOS).

```bash
python3 -m pip install --user modelscope
modelscope download \
  --model FlagRelease/Xing4.0-29B-A4B-W4A8-arm-FlagOS \
  --local_dir ./xingchen4-w4a8
```

Keep all 41 shards, the index, configuration, tokenizer, and custom Python
files together. Model weights are separate from the Runtime.

### 2. Install the prebuilt macOS Runtime (recommended)

Binary release: [v0.2.0-alpha.2-xing4-0](https://github.com/kevinzs2048/flagos-macos-runtime/releases/tag/v0.2.0-alpha.2-xing4-0).
[Runtime archive](https://github.com/kevinzs2048/flagos-macos-runtime/releases/download/v0.2.0-alpha.2-xing4-0/flagos-runtime-0.2.0-alpha.2-xing4-0-darwin-arm64-m5pro.tar.gz)
and [SHA256 checksum](https://github.com/kevinzs2048/flagos-macos-runtime/releases/download/v0.2.0-alpha.2-xing4-0/flagos-runtime-0.2.0-alpha.2-xing4-0-darwin-arm64-m5pro.tar.gz.sha256).

Install Xcode Command Line Tools if needed:

```bash
xcode-select --install
```

Download the installer and verify it before execution:

```bash
cd xingchen4-w4a8
curl --fail --location --remote-name \
  https://github.com/kevinzs2048/flagos-macos-runtime/releases/download/v0.2.0-alpha.2-xing4-0/install.sh
curl --fail --location --remote-name \
  https://github.com/kevinzs2048/flagos-macos-runtime/releases/download/v0.2.0-alpha.2-xing4-0/install.sh.sha256
shasum -a 256 -c install.sh.sha256 &&
bash install.sh
./runtime/bin/vllm --version
```

Expected version: **0.24.0+cpu**. The installer downloads and verifies the
Runtime, installs it into a new `./runtime` directory, and refuses to overwrite
an existing installation. It requires no `sudo`, PATH changes, or environment
setup. This developer alpha is unsigned and non-notarized.

To remove this isolated installation (recoverable in macOS Trash):

```bash
bash install.sh --uninstall ./runtime
```

### 3. Build the Runtime from source (developer alternative)

Skip this section when using the prebuilt Runtime. Use the matching developer
source bundle and the Runtime from step 2 as the bootstrap. Ninja must be
available on PATH. From the model directory:

```bash
cd ..
git clone --branch v0.2.0-alpha.2-xing4-0 \
  https://github.com/kevinzs2048/flagos-macos-runtime.git xingchen4-runtime-source
curl --fail --location --remote-name \
  https://github.com/kevinzs2048/flagos-macos-runtime/releases/download/v0.2.0-alpha.2-xing4-0/xing4-0-developer-sources.tar.gz
curl --fail --location --remote-name \
  https://github.com/kevinzs2048/flagos-macos-runtime/releases/download/v0.2.0-alpha.2-xing4-0/xing4-0-developer-sources.tar.gz.sha256
shasum -a 256 -c xing4-0-developer-sources.tar.gz.sha256 &&
tar -xzf xing4-0-developer-sources.tar.gz
cd xingchen4-w4a8
bash ../xingchen4-runtime-source/build_from_source.sh \
  ../developer-sources ./runtime ./runtime-source
./runtime-source/bin/vllm --version
```

This rebuilds the native Arm operators and plugin, then applies the Xing4_0
overlay. It reuses validated Python/PyTorch/Triton CPU and the vLLM CPU ABI;
it does not rebuild every dependency from zero.
[BUILDING_MAC.md](BUILDING_MAC.md) describes the source inventory and validation.
Validate any further rebuilt Runtime separately; the table below is from the
installed alpha.2 release candidate.

### 4. Start W4A8 inference with vLLM

In the same directory:

For a source-built Runtime, replace `./runtime/bin/vllm` with
`./runtime-source/bin/vllm` in the command below.

```bash
./runtime/bin/vllm serve . \
  --served-model-name xingchen4-w4a8 \
  --trust-remote-code --dtype bfloat16 \
  --distributed-executor-backend uni \
  --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --enforce-eager --no-enable-prefix-caching \
  --host 127.0.0.1 --port 8000
```

Run from a fresh terminal without inherited MTP/draft settings. Do not add
`--speculative-config`. First use may take longer for JIT compilation.
The tested serving context is 1024 tokens; longer contexts are unvalidated.

### 5. Send a request

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"xingchen4-w4a8","messages":[{"role":"user","content":"What is 6 times 7? Answer with the number only."}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

Read `choices[0].message.content`.

## Measured performance

M5 Pro / 64 GiB, CPU-only, batch one, September 17, 2026.
Three-round medians, tokens/s:

| Metric | MTP off |
| --- | ---: |
| PP512 | 312.48 |
| TG128 pure decode | 39.88 |
| TG128 full request | 38.97 |
| Combined PP512 + TG128 | 118.97 |

Synthetic random-token prompts (seed 20260820), greedy decoding, EOS ignored.
PP512 = 512 inputs / request time. Pure TG128 = 127 outputs / (128-output
request time minus a separate first-token baseline). Full-request TG128 =
128 outputs / request time. Combined = 640 input/output tokens / request time.

Each round started after 180 seconds of cooldown following warmup.
Background storage activity remained, and chip temperature was not measured.
These are in-process engine results, not chat or HTTP serving throughput.

All measured outputs repeated and matched the target-only reference.
These measurements are accepted for this release; they do not establish full
performance equivalence to earlier runs. MTP HTTP serving is unvalidated.

## Quantization and validation

- W4: symmetric G128, MSE scale search; A8: dynamic per-token INT8.
- W4 values are stored in INT8 tensors on disk and repacked during loading.
  Indexed tensor size: 28.80 GiB; this is not a nibble-packed GPTQ/MLX/GGUF model.
- The 212 MTP tensors are excluded from this target export.
- Target-only HTTP smoke tests passed Chinese, English, and arithmetic requests,
  repeated twice. This is not a comprehensive accuracy evaluation.

## License

Model weights remain subject to the original publisher's license.
Runtime component notices are included in the Runtime.
