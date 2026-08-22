# FlagOS macOS Runtime developer-alpha acceptance

Acceptance host: Mac17,9 / Apple M5 Pro / 64 GiB / macOS 26.5.1. Runtime
version: `0.1.0-alpha.2`; ABI: `flagos-arm-w4a8-g128-v1`. This unified Express
release formally supports Qwen3.8-27B W4A8, MiniCPM5-2.6B W4A8 G128 and
MiniCPM5-2.6B channel-wise W8A8 text inference from one Runtime. It descends
from the immutable Qwen release tag `v0.1.0-alpha.1`.

This is an unsigned, non-notarized developer alpha. It is a native macOS CPU
Runtime; no Docker or Metal path is used.

## Source and build evidence

- Stock vLLM is clean at upstream tag `v0.20.2`, commit `bc150f502`; no vLLM
  patch or Runtime-specific vLLM branch is shipped.
- Triton CPU and FlagGems use the clean unified candidate commits recorded in
  `sources.lock.json`; vLLM-Plugin-FL and libtriton_jit remain on their clean
  `macos-arm-w4a8` commits.
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
| Runtime logical archive (10 checksummed Release parts) | 476.0 MiB | `55581da456950e6707285af162a56f4479773288cfa43ee97923b5a2491dc95b` |
| `flagos-wheelhouse-0.1.0-alpha.2-cp311-darwin-arm64.tar.gz` | 81.1 MiB | `6eb7d9b1db3f6ba13068573f20ba8fb7edbcaeefa7e53491f6a8dc8d537bab0b` |
| `install.sh` | 6.6 KiB | `10a3d3e7997d19dc6e002c70c6fd92368b89d409033dace84318c41792a1e08d` |

Archive verification checked 42,580 Runtime files, 40,735 text files for host
path relocation, 529 Mach-O images, safe archive paths, source provenance and
all checksum sidecars. The packaged `gen_ssig` and `standalone_compile` JIT
helpers also passed an import probe. There are no developer absolute load paths
and all OpenMP references resolve to one Runtime image. The inference stack is
precompiled to relocatable Python 3.11 bytecode for predictable cold startup.
Nested dependency test suites, `.dSYM` bundles and the 63 MiB XGrammar static
development library are absent. `libtvm_ffi_testing.dylib` is retained because
the published `tvm_ffi/core` extension links it as a Runtime dependency; the
release verifier imports both `tvm_ffi` and XGrammar explicitly.
The wheelhouse contains exactly four verified component wheels. NumPy/SciPy
informational build metadata contains no ephemeral host paths, and dependencies
from Torch and scikit-learn resolve to the same single Runtime OpenMP image.

The isolated end-user workflow passed install, activation, standard
`vllm --version`, standard `vllm serve --help`, rollback and active-version
uninstall protection. All four wheels also installed and imported with
`pip --no-index --no-deps` in a fresh target directory.

A README-driven deployment audit additionally caught that PyTorch Inductor
cannot compile its CPU sampler when Runtime library paths contain whitespace.
The default install root is therefore `~/Library/FlagOS`, and the installer
rejects whitespace-bearing overrides explicitly. A fresh install at that path
loaded the full Qwen3.8 checkpoint, completed sampler/model warmup, activated
the `qwen3` reasoning parser and returned a normal chat completion with
reasoning and final content separated.

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

For MiniCPM5, the final packaged Arm operator bundle passed all nine numerical
W4/W8 runtime tests: regular versus coarse-stripe equality, decode
repeatability, thread-scope restoration, AOT Parameter identity and the
symmetric W8 activation/compact-RHS contract. Triton CPU was rebuilt before
testing; all three related `TritonCPU/kai-layout` lowering tests passed. The
FlagGems source file used by the clean repository, Runtime archive and
developer wheel is byte-identical. The final dylib links only libtriton_jit,
Torch, OpenMP and system libraries; it has no KleidiAI/TLE compute dependency.

The latest MiniCPM5 single-stream medians are W4A8 PP512 942.63 tok/s, TG128
90.07 tok/s and Total 327.80 tok/s; W8A8 is PP512 1166.56 tok/s, TG128 65.58
tok/s and Total 268.29 tok/s. These measurements use `vllm serve`, concurrency
1, disabled prefix caching, one discarded full-shape prime and three retained
samples separated by 90-second idle intervals. The same HTTP workload on
llama.cpp with KleidiAI measured Q4_0 at 776.53/74.21/269.66 tok/s and Q8_0 at
769.51/51.94/205.74 tok/s. Full retained values and the comparison caveat are
recorded in `benchmarks/minicpm5-express-comparison-20260822.json`; the earlier
packaged-runtime run remains in `benchmarks/minicpm5-packaged-runtime-final.json`.

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

The unified release retains the stock vLLM, vLLM-Plugin-FL and libtriton_jit
commits from the Qwen base. Triton CPU and FlagGems advance to the commits in
`sources.lock.json`; those commits preserve the Qwen G128/GDN routes while
adding the MiniCPM W4/W8 routes. `VLLM_CPU_OMP_THREADS_BIND=0`, which denoted
CPU ID zero and was inert in the measured UniProc route, is now expressed
accurately as `nobind`; OpenMP remains explicitly set to 14 threads. A final
standard `vllm serve` smoke loaded the Qwen checkpoint, completed warmup,
returned a normal Chinese completion, and ran pp512/tg128 at 78.40 ms TPOT
(12.76 tok/s) without a cooldown interval. This is 2.4% below the retained
13.07 tok/s cooled median and inside the 3% regression gate. The final archive
separately passed all Runtime hashes and direct native-op registration checks.

## Known boundaries

- SME2 is detected but unused by the production G128 route; the validated
  kernels use Arm SDOT/I8MM.
- The alpha is not Developer ID signed or notarized.
- The locked Triton CPU and FlagGems candidate commits are locally committed
  but not yet published on their configured remotes; source rebuilds currently
  require the two clean local checkout overrides documented in `BUILDING.md`.
- Python/Torch and the compiled vLLM/Triton compiler extensions come from the
  validated build environment rather than a fully hermetic CI rebuild.
- The published Qwen ModelScope ID is embedded in the supported-model registry;
  the single MiniCPM publication ID must replace its placeholder before the
  `v0.1.0-alpha.2` release is tagged.
