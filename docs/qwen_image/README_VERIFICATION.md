# README command verification

## Result

The English model card's **original validation Mac environment setup** and
unmodified image-generation example completed successfully on September 20,
2026. The process exited with status 0 and saved a valid 1024 × 1024 PNG.

| Setting / measurement | Result |
| --- | --- |
| Prompt | A capybara reading a book by candlelight, watercolor painting |
| Resolution / steps / seed | 1024 × 1024 / 40 / 42 |
| Runtime / backend | PyTorch CPU / native SME2 |
| Checkpoint | September 20 W8A8, same112 |
| Warmup | None; the README example does not request it |
| Resident request | 767.821 seconds |
| Model load and prepack inside loader | 2.692 seconds |
| Whole command, measured with `/usr/bin/time -p` | 785.01 seconds |
| W8 coverage | 112 distinct modules, 40 calls each; 4,480 total |

Resident time includes text encoding, denoising, VAE decoding and PIL conversion;
it excludes loading, prepacking, initial latent creation and PNG writing. Whole
command time also includes Python imports, startup, output writing and teardown.
This is a fresh process with existing dependency/build caches, not a cold-machine
or cold-compilation benchmark. Do not compare its single-prompt timing with the
paper-lantern benchmark as evidence of an optimization.

The image was opened and inspected: a capybara, books, a lit candle and watercolor
appearance are visible. Dimensions, decoded-pixel hash, PNG hash and operator
coverage were checked programmatically. This checks the documented workflow;
it does not establish quantization parity against BF16 or broad image quality.

## Reproduction

Run in an output directory of your choice on the original validation Mac:

```bash
export MODEL_DIR="$HOME/Qwen-Image-2.1-0920-W8A8-PerChannel-112"
source "$HOME/qwen-image-2.1-repos/env-upstream.sh"
export QWEN_RUNTIME_DIR="$HOME/flagos-macos-runtime"
bash "$QWEN_RUNTIME_DIR/bin/qwen-image" \
  --model "$MODEL_DIR" \
  --prompt 'A capybara reading a book by candlelight, watercolor painting' \
  --width 1024 --height 1024 \
  --steps 40 --seed 42 \
  --output result.png
```

The recorded application revision is `78e5ce228b77040babfc218cc050a1f1940da546`;
FlagGems is `4fe54663c8a70949ae2346d48a209bc49bedc1e9`. Documentation and evidence
updates after this run do not change the inference implementation or commands.

- [Machine-readable evidence](../../benchmarks/qwen-image-readme-1024-40.json)
  includes revisions, dependency versions, command/environment hashes and dispatch.
- Local complete artifacts: `~/qwen-image-2.1-repos/validation/readme-1024-40/`.
- Model-directory copy: `validation/readme-1024-40/`.

## Installation boundary

This run reused the documented local environment: Python packages, reference
Diffusers source, FlagTree-CPU, KleidiAI and OpenMP were already prepared. It did
not execute the alternative fresh-clone/editable-install instructions on a new
Mac. The `/path/to/...` entries in that alternative are placeholders that users
must configure. No self-contained Mac runtime installer for this profile has
been published. Therefore this result validates the local inference instructions,
not an end-to-end clean-machine installation.
