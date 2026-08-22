# Multi-model release policy

`minicpm-express` is the integration branch for one FlagOS macOS Runtime. It
descends from the Qwen3.8-27B release tag `v0.1.0-alpha.1`; MiniCPM and future
model releases extend that Runtime instead of creating model-specific Runtime
forks.

## Compatibility contract

- `runtime-manifest.json` is the authoritative registry of models validated by
  the release. A model card may link to the shared Runtime, but it must not
  maintain a private copy of the Runtime source or binaries.
- `sources.lock.json` pins one coherent vLLM, Triton CPU, FlagGems,
  vLLM-Plugin-FL and libtriton_jit stack for every listed model.
- `profiles/m5-pro.env` is a hardware/runtime profile, not a model selector.
  Kernel routing must be guarded by architecture, quantization contract, tensor
  shape or execution phase so an optimization for one model cannot silently
  replace an incompatible route for another.
- Model-specific serving arguments belong in the README and model card. They
  are passed to the standard `vllm` CLI; the launcher must not guess a model
  from a directory name.
- Existing release tags remain immutable. A changed source lock, profile or
  supported-model set is published under a new Runtime version.

## Adding a model

Before a model is added to `supported_models`:

1. Record architecture, checkpoint quantization contract, ModelScope ID and
   inference boundaries in `runtime-manifest.json`.
2. Add a complete `vllm serve` command to the README and model card.
3. Run a real OpenAI-compatible HTTP correctness smoke, including the model's
   chat/reasoning behavior where applicable.
4. Measure cold concurrency-one PP512, TG128 and Total throughput with prefix
   caching disabled and retain the raw results under `benchmarks/`.
5. Capture kernel-route coverage and confirm that prohibited fallbacks remain
   zero.
6. Re-run targeted correctness and performance regression checks for every
   existing model whose shared kernels or profile settings changed.
7. Rebuild the Runtime and wheelhouse, then run release, install, relocation
   and native-operator verification.

A documentation-only model registration is not sufficient. Multiple
quantization checkpoints of the same model are recorded as variants under one
model entry, not as separate model publications.

## Current support

| Model | Quantization | Runtime path |
| --- | --- | --- |
| Qwen3.8-27B | W4A8 GPTQ G128; W8A8-prepared head | Q4 G128 + GDN + W8 head |
| MiniCPM5-2.6B | W4A8 GPTQ G128 | Q4 G128 + Llama attention |
| MiniCPM5-2.6B | channel-wise W8A8 | W8 body/head + Llama attention |

All three use the same installed `vllm` executable and M5 Pro profile.
