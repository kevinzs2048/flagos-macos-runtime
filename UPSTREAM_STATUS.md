# Upstream and local status

`sources.lock.json` records an exact commit and tree for every component. The
build materializes those commits from Git; local overrides must be clean and
match both locked identifiers.  The W4/W8 compiler and kernel work is isolated
on `codex/minicpm5-g128-m16-prefill`; the unchanged adapter/JIT repositories
remain on their existing `macos-arm-w4a8` line.

As of 2026-08-22, the two `codex/minicpm5-g128-m16-prefill` refs are committed
locally but are not present on the configured GitHub remotes. A source build
must therefore set `FLAGOS_TRITON_SOURCE` and `FLAGOS_FLAGGEMS_SOURCE` to these
clean checkouts. No remote push is performed implicitly by the release build.
Once maintainers publish the refs, the same commit/tree lock can use network
materialization without changing the candidate source.

| Component | Responsibility | Candidate state |
| --- | --- | --- |
| vLLM 0.20.2 | Framework, CPU attention and compressed-tensors model loading | Stock `v0.20.2` commit `bc150f502`, plus audited Darwin OpenMP and stock-Inductor AOT patches during the isolated build |
| vLLM-Plugin-FL | Thin vLLM platform/kernel registration and version-locked compatibility hooks | `macos-arm-w4a8` at `2ccd0485`; unchanged by the MiniCPM optimization |
| FlagGems 5.0.2 | Triton W4/W8 pack and kernels, Arm operator binding and process-local JIT cache locks | `codex/minicpm5-g128-m16-prefill` at `4e284f0d` |
| Triton CPU 3.7.2 | Apple Arm CPU lowering, SDOT/I8MM and native M16 accumulator preservation | `codex/minicpm5-g128-m16-prefill` at `1feeab7e` |
| libtriton_jit 0.1.0 | CPU JIT launch ABI and OpenMP scheduling | `macos-arm-w4a8` at `a4eb4db9`; no local change |
| Runtime wrapper/profile | Source materialization, M5 Pro W4/W8 policy, standard vLLM launcher and installer | This repository; current branch `codex/minicpm5-arm-runtime` |

## Review order and measured effect

Review the compiler change before the FlagGems series:

1. Triton CPU `1feeab7e`: preserve native I8MM accumulators for the G128 M16
   lowering. This is the compiler prerequisite for the W4 M16 route.
2. FlagGems `9d54e3f1`: schedule eligible W4 G128 Prefill work as coarse,
   contiguous N stripes.
3. FlagGems `abb79dfd`: keep the W4 fast apply wrapper data-only for stable AOT
   cache keys.
4. FlagGems `ce34ab00`: apply the same coarse N-stripe scheduling principle to
   W8 Prefill.
5. FlagGems `3610e17c`: scope JIT cache locks to a process instead of serializing
   unrelated processes through a shared filesystem lock.
6. FlagGems `4e284f0d`: document routing, tuning variables and the compute/launch
   boundary.

For W8 on the M5 Pro, three pp512/tg128 HTTP runs compared the regular Triton
grid with `FLAGGEMS_W8_STEALING_PREFILL=1` and chunk size 2. TTFT changed from
473.981/479.839/495.355 ms to 447.290/439.538/454.783 ms. The corresponding
Prefill median changed from 1067.0 to 1144.7 tok/s (+7.3%). Median total
throughput changed from 263.85 to 268.22 tok/s (+1.7%). Decode stayed within
run-to-run noise at about 65.8 tok/s, as expected because this patch changes
the M>1 I8MM schedule rather than the M=1 SDOT route.

The arithmetic remains in `.py` Triton kernels. `q4_op.cpp` and `w8_op.cpp`
select routes, partition the logical grid, launch through libtriton_jit and
maintain counters; they do not implement a second C/C++ GEMM or embed
KleidiAI. This boundary is intentional: the optimization copies the coarse
stripe scheduling principle, not the external compute library.

The retired `vllm_triton_cpu_qwen35` package, standalone
`libtriton_jit_q4_op.dylib` and TLE/KleidiAI compute routes are deliberately not
part of this candidate. Q4/W8 ship as Triton kernels through the versioned
FlagGems tree and one thin `libflag_gems_arm_ops.dylib` launcher linked to the
versioned libtriton_jit runtime.  Runtime-specific framework changes are kept
as the three reviewable files under `patches/`: Darwin OpenMP, finalized vLLM
AOT artifact loading, and the Torch Inductor token-parallel guard.

The alpha build is not fully hermetic: Python, Torch and the Triton compiler
extension come from the validated build environment.  vLLM's CPU extension,
libtriton_jit, the FlagGems Arm operator bundle and the plugin wheel are rebuilt
from the locked sources.  The release pins source commits, package versions,
final Mach-O dependencies and every emitted Runtime file hash.  A later CI
productization step should additionally rebuild Python, Torch and the Triton
compiler extension from their pinned sources.
