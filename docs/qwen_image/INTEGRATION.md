# Qwen-Image PyTorch CPU integration

## Ownership

The runtime owns model integration, not operator arithmetic. The dependency
direction is runtime -> FlagGems; FlagGems does not import this model package.

| Code | Owner and location |
| --- | --- |
| W8/BF16 Linear, packing, caches, execution policies | FlagGems `src/flag_gems/runtime/backend/_arm/quantized_linear/sme2/` |
| Triton kernels | FlagGems `src/flag_gems/runtime/backend/_arm/ops/` |
| Existing native SME2/KleidiAI kernels | FlagGems `src/flag_gems/csrc/arm/qwen_image/` |
| Compiler/lowering support | FlagTree-CPU, unchanged by this migration |
| Qwen W8 checkpoint schema, layer selection and loading | Runtime `src/qwen_image_cpu/w8a8_sme.py` |
| Reference Diffusers compatibility and CPU adaptation | Runtime `src/qwen_image_cpu/reference.py` |
| Pure ATen FP32 precision compatibility, also used by the baseline | Runtime `src/qwen_image_cpu/cpu_accumulation.py` |
| Model-specific fusion wiring and shared modulation | Runtime `src/qwen_image_cpu/` adapter modules |
| Image request orchestration | Runtime `src/qwen_image_cpu/pipeline.py` and `__main__.py` |
| Mac environment defaults and launcher | Runtime `profiles/qwen-image-m5-pro.env`, `bin/qwen-image` |
| Offline export and precision tooling | Runtime `export_w8.py`, `validate_w8.py` |
| Generic operator tests | FlagGems `tests/arm/sme2/` |
| Model integration tests | Runtime `tests/qwen_image/` |

Experimental QKV scheduling and other opt-in adapters remain available for
regression, but the default is still native SME2, same112 W8A8, 18 main threads
and 12 MLP workers. Existing C++/assembly and production Triton kernel arithmetic
is unchanged. The compiler repository has no new changes in this migration.

The model adapters currently replace selected reference methods at runtime.
They are maintained source in this repository, not user-written startup code.
The supplied reference Diffusers source remains a separate dependency. This
migration does not claim that the adapters have been accepted upstream.

## Install the adapter in a prepared environment

Use the locked PyTorch/FlagGems/FlagTree/KleidiAI environment described in
[the model card](MODEL_CARD_MAC.md). The adapter wheel alone is not a complete
Python/native runtime. Its Python metadata does not install the private
reference Diffusers implementation or build the native dependencies.

```bash
export QWEN_RUNTIME_DIR=/path/to/flagos-macos-runtime
export FLAGGEMS_DIR=/path/to/FlagGems
export QWEN_IMAGE_PYTHON=/path/to/prepared/python3.11

"$QWEN_IMAGE_PYTHON" -m pip install --no-deps --no-build-isolation -e "$FLAGGEMS_DIR"
"$QWEN_IMAGE_PYTHON" -m pip install --no-deps --no-build-isolation -e "$QWEN_RUNTIME_DIR"

bash "$QWEN_RUNTIME_DIR/bin/qwen-image" --help
```

On the validation machine, `~/qwen-image-2.1-repos/env-upstream.sh` supplies the
reference source, KleidiAI, compiler and OpenMP paths. It now resolves the model
package from this runtime repository, without the old FlagGems example path.

The installed Python application is also callable with `python -m qwen_image_cpu`
or `qwen-image` after configuring the environment. `bin/qwen-image` is the source
checkout launcher and applies the dedicated Qwen environment profile.

## Build the Python adapter wheel

```bash
"$QWEN_IMAGE_PYTHON" -m pip wheel --no-deps --no-build-isolation \
  "$QWEN_RUNTIME_DIR" --wheel-dir "$QWEN_RUNTIME_DIR/artifacts/qwen-image"
```

The existing vLLM `build.sh`, installer, release manifests and Mac profiles are
unchanged. They do not implicitly include this new application. The Qwen adapter
uses its own Python package and source lock; a self-contained public Qwen runtime
installer remains separate release work.

## Regression commands

```bash
"$QWEN_IMAGE_PYTHON" -m pytest "$FLAGGEMS_DIR/tests/arm/sme2" \
  --confcutdir="$FLAGGEMS_DIR/tests/arm/sme2" -q
"$QWEN_IMAGE_PYTHON" -m pytest "$QWEN_RUNTIME_DIR/tests/qwen_image" -q
```

The explicit conftest boundary selects the standalone Arm native tests without
the parent generic GPU-oriented FlagGems test configuration. Model image
regression uses the same source weights, prompt, seed and warmup as the previous
512/1024, 40-step runs. Acceptance requires identical pixel hashes and all 112
W8 modules executing once per step, with resident latency reported separately
from loading and warmup.

Earlier reports in this directory record their original paths and commits as
historical evidence. Current runnable code resides at the locations above.
