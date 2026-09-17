# Build XingChen4 on macOS arm64

Use the source tag and developer archive from the same Xing4_0 GitHub Release
as the prebuilt Runtime. The main [README](README_W4A8.md) includes their URLs
and download commands.

## Inputs and layout

```text
workspace/
  xingchen4-w4a8/
    ...model files...
    runtime/                       # Prebuilt vLLM 0.24.0+cpu bootstrap
  xingchen4-runtime-source/         # Fixed source tag
    build_from_source.sh
    apply_adapter.sh
    adapter/
    launcher/
    scripts/
    sources.lock.json
    sources.public.lock.json
  developer-sources/               # Extracted developer archive
    flagos-macos-runtime-xingchen4-w4a8-v024/
    vllm-0.24.0/
    FlagGems/
    vllm-plugin-FL-v024/
    libtriton_jit/
    KleidiAI-prefill-experiment/
    dependencies/
      fmt/
      json/
```

Requirements: native Apple M5 Pro/macOS arm64, Xcode Command Line Tools,
Ninja on PATH, and whitespace-free paths. The bootstrap supplies Python 3.11,
Torch 2.11.0 and headers, CMake, libomp, Triton CPU, and `vllm/_C.abi3.so`.

Keep the source directory names and sibling layout: the native helper resolves
FlagGems and libtriton_jit from the plugin's parent directory.

## Actual public source snapshots

| Input | Commit |
| --- | --- |
| vLLM 0.24.0 | `ee0da84ab9e04ac7610e28580af62c365e898389` |
| FlagGems recovered snapshot | `ccc563d7894b8542ae51a36a32627a821a832a0b` |
| Plugin recovered snapshot and build helper | `193f7f4cbfdf9e40a3643571994e907e1120916c` |
| libtriton_jit | `a4eb4db996a63afe281f6ac6418182c63105d73a` |
| KleidiAI | `4bc7bd457930de119f8b826ecfdebfbd9c08cf8c` |

Two historical component commits were unavailable. Their package source was
recovered byte-for-byte from the hash-verified accepted Runtime, with root build
scaffolding from separately identified commits. These are new, explicitly
identified snapshots, not a claim that the lost commits were restored.
`sources.public.lock.json` records commits, recovery provenance, and SHA256
inventories. `sources.lock.json` retains the historical input records.

fmt 11.2.0 and nlohmann/json 3.11.3 sources are bundled and pinned by SHA256.
Python/PyTorch/Triton CPU and the vLLM CPU ABI are reused from the bootstrap.
The compiler remains `3.7.2+gitbdada74a`; it is not rebuilt in this workflow.

## Build

From the model directory:

```bash
bash ../xingchen4-runtime-source/build_from_source.sh \
  ../developer-sources ./runtime ./runtime-source
./runtime-source/bin/vllm --version
```

The output directory must not exist. Use a fresh base-builder copy: the wrapper
refuses existing generated Runtime/plugin outputs and the base Runtime archive.
It does not reset the original checkouts or modify the bootstrap.

The wrapper checks source commits/file inventories, builds KleidiAI,
libtriton_jit and FlagGems Arm operators, builds the plugin wheel, assembles
the Runtime, applies the Xing4_0 overlay, and regenerates the integrity manifest.
It marks the new Runtime's validation pending. There is no from-zero rebuild
of every dependency.

## Validate

Use the serving and request commands in the main README, replacing
`./runtime/bin/vllm` with `./runtime-source/bin/vllm`.
Validate correctness and performance before distributing a rebuilt Runtime.
Its binary checksum and performance must not be copied from the prebuilt release.

The local complete native/plugin build passed, including Runtime integrity and
vLLM version checks. Smoke/performance results, when available, are recorded in
the release acceptance evidence; a build pass alone is not an accuracy claim.
