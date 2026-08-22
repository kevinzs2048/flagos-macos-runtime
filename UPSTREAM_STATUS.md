# Upstream and local status

`sources.lock.json` records an exact commit and tree for every component. The
build materializes those commits from Git; local overrides must be clean and
match both locked identifiers. All four non-vLLM component checkouts used by
the unified Express package are on the local `minicpm-express` branch. This
integration branch descends from the Qwen3.8-27B Runtime base and groups the
exact multi-model package inputs without rewriting their history.

As of 2026-08-23, all four component `minicpm-express` refs are published on
their configured GitHub repositories. A networked source build can materialize
the locked commits directly; local overrides remain available for offline
builds. No remote push is ever performed implicitly by the release build.

| Component | Responsibility | Candidate state |
| --- | --- | --- |
| vLLM 0.20.2 | Framework, CPU attention and compressed-tensors model loading | Stock `v0.20.2` commit `bc150f502`, plus audited Darwin OpenMP and stock-Inductor AOT patches during the isolated build |
| vLLM-Plugin-FL | Thin vLLM platform/kernel registration and version-locked compatibility hooks | published `minicpm-express` at `2ccd0485`; unchanged by the MiniCPM optimization |
| FlagGems 5.0.2 | Triton W4/W8 pack and kernels, Arm operator binding and process-local JIT cache locks | published `minicpm-express` at `09c2947d` |
| Triton CPU 3.7.2 | Apple Arm CPU lowering, SDOT/I8MM and native M16 accumulator preservation | published `minicpm-express` at `1feeab7e` |
| libtriton_jit 0.1.0 | CPU JIT launch ABI and OpenMP scheduling | published `minicpm-express` at `a4eb4db9`; no code change |
| Runtime wrapper/profile | Source materialization, M5 Pro W4/W8 policy, standard vLLM launcher and installer | release candidate branch `minicpm-express` |

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
7. FlagGems `5d7dc1bc`: trim exact-KAI W8 dispatch, allocator and vLLM wrapper
   overhead while keeping all SDOT/I8MM arithmetic in Triton.
8. FlagGems `532d7009`: retain W4 G128 JIT handles on the decode hot path.
9. FlagGems `dea677ef`: replace duplicated manual OpenMP restoration with one
   scoped guard across W4 prefill routes.
10. FlagGems `ac73d54f`: preserve the prepared Q4 Parameter identity through
    joint layer/kernel AOT deepcopy.
11. FlagGems `25657c75`: separate W8 Prefill and Decode scheduling so coarse
    N-stripe I8MM work does not perturb the M=1 SDOT hot path.
12. FlagGems `a892d50c`: honor the checkpoint's symmetric activation contract,
    remove the obsolete asymmetric W8 packing route and cover the compact RHS
    layout with direct runtime tests.
13. FlagGems `09c2947d`: remove retired W8 exports and make the AArch64 runtime
    tests auto-discover packaged libraries, including compact Prefill tails.

The combined candidate builds cleanly as one Arm operator bundle. Direct tests
cover W4 G128, W4 G32 compact/SwiGLU, W8 decode/prefill, regular/coarse-stripe
bit equality, scoped thread restoration and W4/W8 AOT Parameter identity. Its
Mach-O imports libtriton_jit, Torch, OpenMP and system libraries only; it has no
KleidiAI, native GEMM or embedded assembly compute path. Final no-contention
HTTP performance gates remain required before this ref replaces the packaged
alpha Runtime.

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
versioned libtriton_jit runtime. Runtime-specific framework changes are kept
as the three reviewable files under `patches/`: Darwin OpenMP, finalized vLLM
AOT artifact loading, and the Torch Inductor token-parallel guard.

The alpha build is not fully hermetic: Python, Torch and the Triton compiler
extension come from the validated build environment.  vLLM's CPU extension,
libtriton_jit, the FlagGems Arm operator bundle and the plugin wheel are rebuilt
from the locked sources.  The release pins source commits, package versions,
final Mach-O dependencies and every emitted Runtime file hash.  A later CI
productization step should additionally rebuild Python, Torch and the Triton
compiler extension from their pinned sources.
