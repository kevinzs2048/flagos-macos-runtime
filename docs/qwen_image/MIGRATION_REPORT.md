# Qwen-Image code ownership migration

## Result

Qwen-Image model integration now belongs to `flagos-macos-runtime`. Reusable
Arm computation belongs to the FlagGems source package. The former FlagGems
example contains only `README.md` and the four-line compatibility launcher.

| Repository | Changes |
| --- | --- |
| FlagGems | Move W8/BF16 Linear, packing/cache helpers and execution policies into `src/flag_gems/runtime/backend/_arm/quantized_linear/sme2/`; move the remaining experimental Triton kernels into `_arm/ops/`; relocate generic tests to `tests/arm/sme2/` |
| flagos-macos-runtime | Own checkpoint loading, CPU compatibility, Qwen model fusion wiring, CLI, offline validation/export tools and model tests; add Python package metadata, `bin/qwen-image`, a dedicated profile and dependency lock |
| FlagTree-CPU | No changes; retain the existing validated compiler revision |
| PyTorch / reference Diffusers | No source changes; model-specific runtime adapters are maintained in the runtime repository |

The native C++/assembly implementation remains in FlagGems. The main W8 Triton
kernels were already in FlagGems. No kernel arithmetic or quantization policy
was changed. The default remains native SME2/KleidiAI, 112 W8 Linear modules,
CFG=1, prefix KV caching, 18 main threads and 12 MLP workers.

The plain PyTorch baseline remains independently usable: its ATen precision
compatibility module is in the runtime, and a subprocess regression verifies
that preparing the baseline imports no FlagGems and loads none of our custom
operator libraries. Reference Diffusers may import Triton through its own
dependencies; importing a module does not mean that the baseline executes
Triton kernels.

## Source identity

- FlagGems computation revision: `4fe54663c8a70949ae2346d48a209bc49bedc1e9`.
- Runtime application revision: `639acd38459a4d58abf4a8374aaa31734a668a20`.
- Runtime branch: `codex/qwen-image-pytorch-cpu`.
- Dependency revisions and component hashes: `qwen-image.sources.lock.json`.
- The migration is committed locally and has not been pushed or published.

An AST comparison checked all 104 original top-level functions/classes across
the migrated modules: their bodies are unchanged after excluding import nodes.
Native source diffs are empty. This structural check complements execution tests;
it does not replace them.

## Tests and package

- FlagGems standalone Arm operator suite: **209 passed**.
- Runtime model/ownership/baseline suite: **171 passed**.
- Total: **380 passed**. Existing CPU device-property/capability fallback warnings
  remain; there were no failed tests in the final run.
- Shell launcher syntax and CLI help checks passed.
- Adapter wheel: `artifacts/qwen-image/flagos_macos_qwen_image-0.1.0-py3-none-any.whl`.
- Wheel size: 58,229 bytes; all 24 Python module files match the committed source.
- Wheel SHA-256: `18bd57ed96e4023796de8606993fb502b3c479e03e4c4e36588d83a778d23030`.

The wheel contains model integration only. It does not duplicate FlagGems
operators and is not a self-contained Python/native runtime. The existing vLLM
installer and release archives are unchanged. The prepared dependency environment
is still required; public runtime packaging is a separate release step.

## Full image regression

The replay uses the September 20 same112 checkpoint, the previous case 001
prompt, seed 42 and a two-step warmup. Pixel SHA-256 and W8 module/call coverage
are checked automatically. Timing includes text encoding, denoising, VAE and
PIL conversion; it excludes loading, prepacking, warmup, initial latent creation
and PNG writing.

| Request | Before migration | After migration | Pixel parity | W8 GEMMs |
| --- | ---: | ---: | --- | ---: |
| 512×512, 40 steps | 158.760 s | 157.052 s | Exact | 4480 |
| 1024×1024, 40 steps | 811.285 s | 770.919 s | Exact | 4480 |

These are single requests compared with earlier same-machine measurements,
not an interleaved performance experiment. Small timing differences are not
evidence of a speedup from moving files.

The first 512 run also matched pixels at 155.504 s. Its following 1024 run was
intentionally interrupted to correct the pure-ATen baseline dependency boundary.
The final run above starts from the corrected application commit. Both final
requests completed successfully and matched the previous image pixels exactly.
No latency regression was observed in these two requests.

Evidence on the validation machine:

- `~/qwen-image-2.1-repos/validation/runtime-migration-final/`: final images,
  request JSON, logs and replay report.
- `~/qwen-image-2.1-repos/validation/runtime-migration-gems.xml`.
- `~/qwen-image-2.1-repos/validation/runtime-migration-model.xml`.
- `benchmarks/qwen-image-migration-static.json`: structural audit and wheel hash.

See [integration and setup](INTEGRATION.md) and [the English model card](MODEL_CARD_MAC.md).
