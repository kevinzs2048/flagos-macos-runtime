# Upstream and local status

All maintained forks use the `macos-arm-w4a8` branch. Stock vLLM is locked
directly to the upstream `v0.20.2` tag and has no Runtime-specific branch.
`sources.lock.json` records an exact commit and tree for every component. The
build materializes those commits from Git; local overrides must be clean and
match both locked identifiers.

| Component | Responsibility | Candidate state |
| --- | --- | --- |
| vLLM 0.20.2 | Framework, CPU attention and compressed-tensors model loading | Stock upstream tag `v0.20.2`, commit `bc150f502`; no Runtime patch or local source change |
| vLLM-Plugin-FL | Thin vLLM platform/kernel registration, version-locked compatibility hooks and coverage activation | Fork `macos-arm-w4a8`, commit locked |
| FlagGems 5.0.2 | Q4/W8 pack and kernels, GDN math/native bundle, fusion and strict route accounting | Fork `macos-arm-w4a8`, commit locked |
| Triton CPU 3.7.2 | Apple Arm CPU lowering, SDOT/I8MM and CPU backend | Fork `macos-arm-w4a8`, commit locked |
| libtriton_jit 0.1.0 | CPU JIT launch ABI, OpenMP scheduling and thread-safe lazy loading | Fork `macos-arm-w4a8`, commit locked |
| Runtime wrapper/profile | M5 Pro threads, strict-kernel policy, standard vLLM launcher and installer | This repository |

The retired `vllm_triton_cpu_qwen35` package, standalone
`libtriton_jit_q4_op.dylib`, TLE routes and vLLM patch file are deliberately not
part of this candidate. Q4/W8/GDN ship through the versioned FlagGems tree and
one `libflag_gems_arm_ops.dylib` linked to the versioned libtriton_jit runtime.

The alpha build is not fully hermetic: Python, Torch, the stock vLLM compiled
extension and the Triton compiler extension come from the validated vLLM build
environment. The release still pins their source commits, package versions,
final Mach-O dependencies and every emitted Runtime file hash. A later CI
productization step should rebuild those toolchain binaries from scratch.
