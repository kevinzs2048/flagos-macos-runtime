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
