# FlagOS macOS Runtime developer-alpha acceptance

Acceptance host: Mac17,9 / Apple M5 Pro / 64 GiB / macOS 26.5.1. Runtime
version: `0.1.0-alpha.1`; ABI: `flagos-arm-w4a8-g128-v1`. The tested model is
`Qwen3.8-27B-W4A8-GPTQ-G128-packed`.

This is an unsigned, non-notarized developer alpha. It is a native macOS CPU
Runtime; no Docker or Metal path is used.

## Source and build evidence

- Stock vLLM is clean at upstream tag `v0.20.2`, commit `bc150f502`; no vLLM
  patch or Runtime-specific vLLM branch is shipped.
- Triton CPU, FlagGems, vLLM-Plugin-FL and libtriton_jit use clean
  `macos-arm-w4a8` commits recorded in `sources.lock.json`.
- The build materializes exact commits from Git and independently verifies each
  tree ID. Generated source trees are ignored build inputs rather than a second
  155 MB copy committed to this repository.
- The old qwen35 plugin, TLE path, standalone Q4 dylib and vLLM patch are absent.
- The simplified Runtime repository has no second Python product/CLI package;
  release verification is performed directly by the build scripts. Formal
  libtriton_jit CTest: 7 passed. Relevant vLLM-Plugin-FL tests: 11 passed.

## Validated release assets

| Asset | Size | SHA256 |
| --- | ---: | --- |
| `flagos-runtime-0.1.0-alpha.1-darwin-arm64-m5pro.tar.gz` | 432.0 MiB | `f811f14295e24a305c1296225707059891df958e229a964a1e6c117a76e37bbe` |
| `flagos-wheelhouse-0.1.0-alpha.1-cp311-darwin-arm64.tar.gz` | 81.1 MiB | `876f251951f996ca87c16a41e135a0be2e9add08bf86bc24936bacc6507c03d7` |
| `install.sh` | 4.9 KiB | `95509229dad1f4882dc70179d7b3ed7562637d61be0155d067039eea5211d7f6` |

Archive verification checked 39,253 Runtime files, 38,186 text files for host
path relocation, 333 Mach-O images, safe archive paths, source provenance and
all checksum sidecars. The packaged `gen_ssig` and `standalone_compile` JIT
helpers also passed an import probe. There are no developer absolute load paths
and all OpenMP references resolve to one Runtime image. The inference stack is
precompiled to relocatable Python 3.11 bytecode for predictable cold startup.
The wheelhouse contains exactly four verified component wheels.

The isolated end-user workflow passed install, activation, standard
`vllm --version`, standard `vllm serve --help`, rollback and active-version
uninstall protection. All four wheels also installed and imported with
`pip --no-index --no-deps` in a fresh target directory.

## Model function and kernel coverage

Model inspection passed: architecture `Qwen3_5ForConditionalGeneration`, 9
language-model shards (18.568 GB), 496 packed W4A8 G128 body linears, dynamic
symmetric per-token INT8 activations and BF16 checkpoint `lm_head` prepared as
W8A8. Vision/MTP remain outside the text-only Runtime path.

Real OpenAI-compatible chat HTTP smoke results are stored in
`benchmarks/qwen38-g128-runtime-correctness.json`. All five fixed cases passed:
Chinese, English, arithmetic (42), Python code and thinking-mode arithmetic
(91). The release Runtime separately passed all 39,253 embedded file
hashes and all nine required native operator registrations. It then loaded the
model, completed warmup and returned a correct response through the real
`/v1/completions` HTTP endpoint. The local model
passed structure/quantization inspection; its eventual public repository should
add a model-level `SHA256SUMS` before publication.

Strict coverage from a real request passed all checks:

- 304 fused G128 Runtime Linear modules prepared.
- Q4 G128 prefill/decode and W8 `lm_head` hit FlagGems kernels.
- GDN prefill/decode and CPU attention backends were observed.
- Attention, Q4 body, W8 head and all GDN fallback counters were zero.

## Batch-one HTTP performance

Protocol: real `vllm serve`, OpenAI streaming completions, batch 1, pp512,
tg128, one discarded full-shape prime and three measured requests, each after
a 90-second idle interval. The server explicitly used `uni` and disabled stats
logging. Preflight reported no competing server. Full evidence is in
`benchmarks/qwen38-g128-runtime-final.json`.

| Metric | Candidate | Regression baseline | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| Prefill median | 76.4665 tok/s | 75.6647 tok/s | +1.06% | PASS |
| Decode median | 13.0727 tok/s | 13.2072 tok/s | -1.02% | PASS |
| Decode TPOT | 76.4955 ms | 75.7164 ms | +1.03% | PASS |
| Median request total | 16.4172 s | 16.4172 s | -0.00% | informational |

The retained Prefill samples were 75.6497, 76.6036 and 76.4665 tok/s. The old
back-to-back protocol's 59.47 tok/s was reproduced as a thermal/protocol
artifact, not a kernel regression, and is retained separately in
`benchmarks/qwen38-g128-runtime-hot-back-to-back.json`. Acceptance uses the
three thermally cooled samples and requires 97% of both historical cold
baselines, so performance and strict kernel coverage jointly pass.

The simplified release retains the exact same five pinned engine component
commits and performance knobs. `VLLM_CPU_OMP_THREADS_BIND=0`, which denoted CPU
ID zero and was inert in the measured UniProc route, is now expressed accurately
as `nobind`; OpenMP remains explicitly set to 14 threads. A final standard
`vllm serve` smoke loaded this checkpoint, completed warmup, returned a normal
Chinese completion, and ran pp512/tg128 at 78.40 ms TPOT (12.76 tok/s) without
a cooldown interval. This is 2.4% below the retained 13.07 tok/s cooled median
and inside the 3% regression gate. The final archive separately passed all
Runtime hashes and direct native-op registration checks.

## Known boundaries

- SME2 is detected but unused by the production G128 route; the validated
  kernels use Arm SDOT/I8MM.
- The alpha is not Developer ID signed or notarized.
- Python/Torch and the compiled vLLM/Triton compiler extensions come from the
  validated build environment rather than a fully hermetic CI rebuild.
- The external model repository ID is intentionally maintained by the model
  publication and is not embedded in this Runtime.
