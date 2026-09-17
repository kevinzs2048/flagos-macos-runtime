"""ARM CPU compatibility layer for the stock vLLM 0.20.2 package.

Keep model/runtime compatibility fixes in vllm-plugin-FL rather than carrying
a permanently modified vLLM source tree.  The integration intentionally fails
closed on a different vLLM version because it patches private 0.20.2 APIs.
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from importlib import metadata

import torch
from vllm.logger import init_logger
from vllm_fl.runtime_families import is_mhc_mla_w4a8_family


_INSTALLED = False
logger = init_logger(__name__)


@contextmanager
def _telechat4_mtp_draft_mla_threads():
    """Temporarily apply the TeleChat4 draft-only MLA worker policy.

    CPU UniProc executes target verification and draft proposal serially. A
    scoped environment override therefore lets the native MLA operator use a
    smaller latency-oriented team for the one-layer draft without changing
    the 40-layer target policy. Restore the caller's value even when draft
    execution raises so later target forwards cannot inherit the override.
    """
    configured = os.environ.get(
        "FL_CPU_TELECHAT4_MTP_DRAFT_MLA_THREADS", "0"
    ).strip()
    if configured.lower() in {"", "0", "off", "false", "no"}:
        yield
        return

    try:
        threads = int(configured)
    except ValueError as exc:
        raise ValueError(
            "FL_CPU_TELECHAT4_MTP_DRAFT_MLA_THREADS must be 0 or an "
            "integer in [1, 256]"
        ) from exc
    if not 1 <= threads <= 256:
        raise ValueError(
            "FL_CPU_TELECHAT4_MTP_DRAFT_MLA_THREADS must be 0 or an "
            "integer in [1, 256]"
        )

    variable = "FLAGGEMS_ARM_MLA_THREADS"
    previous = os.environ.get(variable)
    os.environ[variable] = str(threads)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def _require_vllm_0202() -> None:
    installed = metadata.version("vllm")
    base = installed.split("+", 1)[0]
    if base != "0.20.2":
        raise RuntimeError(
            "The FL ARM CPU compatibility layer requires vLLM 0.20.2; "
            f"found {installed}"
        )


def _install_packed_w4a8() -> None:
    from compressed_tensors.config import CompressionFormat
    from vllm.logger import init_logger
    from vllm.model_executor.kernels.linear import (
        MPLinearLayerConfig,
        choose_mp_linear_kernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as config_module,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a8_int as scheme_module,
    )
    from vllm.model_executor.parameter import (
        BasevLLMParameter,
        ChannelQuantScaleParameter,
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    scheme_cls = scheme_module.CompressedTensorsW4A8Int
    config_cls = config_module.CompressedTensorsConfig
    if getattr(scheme_cls, "_vllm_fl_packed_w4a8", False):
        return

    logger = init_logger(__name__)
    original_init = scheme_cls.__init__
    original_create_weights = scheme_cls.create_weights
    original_get_scheme = config_cls._get_scheme_from_parts

    def scheme_init(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        is_static_input_scheme: bool = False,
        input_symmetric: bool = True,
        packed: bool = False,
    ) -> None:
        original_init(
            self,
            strategy=strategy,
            num_bits=num_bits,
            group_size=group_size,
            is_static_input_scheme=is_static_input_scheme,
            input_symmetric=input_symmetric,
        )
        self._vllm_fl_checkpoint_packed = packed
        self._vllm_fl_pack_factor = 32 // num_bits

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ) -> None:
        if not getattr(self, "_vllm_fl_checkpoint_packed", False):
            return original_create_weights(
                self,
                layer,
                output_size,
                input_size,
                output_partition_sizes,
                input_size_per_partition,
                params_dtype,
                weight_loader,
                **kwargs,
            )

        output_size_per_partition = sum(output_partition_sizes)
        row_parallel = input_size != input_size_per_partition
        effective_group_size = (
            input_size_per_partition if self.group_size == -1 and row_parallel
            else input_size if self.group_size == -1
            else self.group_size
        )
        if input_size_per_partition % effective_group_size:
            raise ValueError(
                f"input partition {input_size_per_partition} is not divisible "
                f"by W4 group size {effective_group_size}"
            )

        kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=effective_group_size,
            zero_points=False,
            has_g_idx=False,
        )
        kernel_type = choose_mp_linear_kernel(kernel_config)
        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info(
                "Using %s for packed CompressedTensorsW4A8Int",
                kernel_type.__name__,
            )
            self._kernel_backends_being_used.add(kernel_type.__name__)

        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self._vllm_fl_pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self._vllm_fl_pack_factor,
            packed_dim=1,
        )
        layer.register_parameter("weight_packed", weight)

        scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                input_size_per_partition // effective_group_size,
                dtype=params_dtype,
            ),
        }
        if self.group_size == -1 and row_parallel:
            weight_scale = ChannelQuantScaleParameter(
                output_dim=0, **scale_args
            )
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **scale_args
            )
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter(
            "weight_shape",
            BasevLLMParameter(
                data=torch.empty(2, dtype=torch.int64),
                weight_loader=weight_loader,
            ),
        )
        self.kernel = kernel_type(
            kernel_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
            w_zp_param_name=None,
            w_gidx_param_name=None,
        )

    def get_scheme_from_parts(
        self,
        weight_quant,
        input_quant,
        format: str | None = None,
        layer_name: str | None = None,
        **kwargs,
    ):
        resolved_format = format if format is not None else self.quant_format
        if (
            resolved_format == CompressionFormat.pack_quantized.value
            and weight_quant is not None
            and input_quant is not None
            and self._is_dynamic_token_w4a8_int(weight_quant, input_quant)
            and kwargs.get("output_quant") is None
        ):
            return scheme_cls(
                num_bits=weight_quant.num_bits,
                strategy=weight_quant.strategy,
                group_size=weight_quant.group_size,
                is_static_input_scheme=False,
                input_symmetric=input_quant.symmetric,
                packed=True,
            )
        return original_get_scheme(
            self,
            weight_quant,
            input_quant,
            format=format,
            layer_name=layer_name,
            **kwargs,
        )

    scheme_cls.__init__ = scheme_init
    scheme_cls.create_weights = create_weights
    config_cls._get_scheme_from_parts = get_scheme_from_parts
    scheme_cls._vllm_fl_packed_w4a8 = True


def _install_cpu_gemm_guard() -> None:
    from vllm.model_executor.layers import utils as layer_utils
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.platforms import current_platform

    original = layer_utils.dispatch_cpu_unquantized_gemm
    if getattr(original, "_vllm_fl_ndim_guard", False):
        return

    def guarded(layer: torch.nn.Module, remove_weight: bool) -> None:
        weight = getattr(layer, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim != 2:
            return
        # vLLM's CPU unquantized GEMM setup normally releases the source
        # matrix after installing the oneDNN/CPU handle.  MLA consumes
        # kv_b_proj once more in MLAAttention.process_weights_after_loading()
        # to build W_UK/W_UV, and the BF16 TeleChat4 MTP block has no
        # quantization method that can reconstruct it.  Keep just this
        # projection materialized until the MLA post-load pass completes.
        if ".kv_b_proj" in str(getattr(layer, "prefix", "")):
            return original(layer, remove_weight=False)
        return original(layer, remove_weight)

    guarded._vllm_fl_ndim_guard = True
    layer_utils.dispatch_cpu_unquantized_gemm = guarded

    # The unquantized method owns the post-load dispatch call.  Keep an
    # explicit class-level guard as well as the utility wrapper above: this
    # remains effective if a platform has captured the utility function before
    # the plugin is imported.
    original_process = UnquantizedLinearMethod.process_weights_after_loading
    if not getattr(original_process, "_vllm_fl_mla_weight_guard", False):

        def process_mla_weight(self, layer: torch.nn.Module) -> None:
            if ".kv_b_proj" in str(getattr(layer, "prefix", "")):
                if current_platform.is_cpu():
                    layer_utils.dispatch_cpu_unquantized_gemm(
                        layer, remove_weight=False
                    )
                return
            return original_process(self, layer)

        process_mla_weight._vllm_fl_mla_weight_guard = True
        UnquantizedLinearMethod.process_weights_after_loading = process_mla_weight


def _install_cpu_moe_native_silu() -> None:
    """Avoid constructing a CustomOp inside the CPU fused-MoE kernel.

    vLLM 0.20.2's generic CPU MoE path creates ``SiluAndMul`` from inside the
    registered C++/Python op.  That construction asks for the current vLLM
    config, which is not set during the MTP warmup callback.  The operation is
    exactly ``silu(gate) * up``; keeping a function in the activation table is
    both cheaper and safe for BF16 draft weights.
    """

    from torch.nn import functional as F
    from vllm.model_executor.layers.fused_moe import cpu_fused_moe

    if getattr(cpu_fused_moe, "_vllm_fl_native_silu", False):
        return

    def silu_and_mul_native(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    cpu_fused_moe._CPU_MOE_ACT_FN[cpu_fused_moe.MoEActivation.SILU] = (
        silu_and_mul_native
    )
    cpu_fused_moe._vllm_fl_native_silu = True


def _install_cpu_mla_metadata_pin_guard() -> None:
    """Disable unsupported MPS pin-memory allocation in CPU MLA metadata.

    The macOS ARM PyTorch build routes ``torch.zeros(pin_memory=True)`` to an
    MPS dispatch stub even when the vLLM device is CPU.  MLA's chunked metadata
    builder is the only CPU path that requests pinned tensors; ordinary CPU
    tensors are sufficient for the in-process executor.
    """

    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadataBuilder,
    )

    current = MLACommonMetadataBuilder.build
    if getattr(current, "_vllm_fl_cpu_pin_guard", False):
        return

    def build_without_mps_pin(self, *args, **kwargs):
        original_empty = torch.empty
        original_zeros = torch.zeros

        def without_pin(factory):
            def wrapped(*fargs, **fkwargs):
                if fkwargs.get("pin_memory", False):
                    fkwargs = dict(fkwargs)
                    fkwargs["pin_memory"] = False
                return factory(*fargs, **fkwargs)

            return wrapped

        torch.empty = without_pin(original_empty)
        torch.zeros = without_pin(original_zeros)
        try:
            return current(self, *args, **kwargs)
        finally:
            torch.empty = original_empty
            torch.zeros = original_zeros

    build_without_mps_pin._vllm_fl_cpu_pin_guard = True
    MLACommonMetadataBuilder.build = build_without_mps_pin


def _install_text_only_vision_guard() -> None:
    from contextvars import ContextVar
    from vllm.model_executor.models import qwen3_5 as qwen

    if getattr(qwen, "_vllm_fl_text_only_vision", False):
        return

    text_only_build = ContextVar("vllm_fl_qwen_text_only_build", default=False)
    original_vision_init = qwen.Qwen3_VisionTransformer.__init__

    def vision_init(self, *args, **kwargs) -> None:
        if text_only_build.get():
            kwargs["quant_config"] = None
        original_vision_init(self, *args, **kwargs)

    qwen.Qwen3_VisionTransformer.__init__ = vision_init

    for model_cls in (
        qwen.Qwen3_5ForConditionalGeneration,
        qwen.Qwen3_5MoeForConditionalGeneration,
    ):
        original_model_init = model_cls.__init__

        def model_init(
            self,
            *,
            vllm_config,
            prefix: str = "model",
            _original=original_model_init,
        ) -> None:
            language_only = bool(
                vllm_config.model_config.multimodal_config.language_model_only
            )
            token = text_only_build.set(language_only)
            try:
                _original(self, vllm_config=vllm_config, prefix=prefix)
            finally:
                text_only_build.reset(token)

        model_cls.__init__ = model_init

    qwen._vllm_fl_text_only_vision = True


def _install_cpu_attention_block_constraint() -> None:
    from vllm.v1.attention.backend import MultipleOf
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

    if hasattr(CPUAttentionBackend, "get_supported_kernel_block_sizes"):
        return

    CPUAttentionBackend.get_supported_kernel_block_sizes = staticmethod(
        lambda: [MultipleOf(32)]
    )


def _install_cpu_cleanup_guard() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner._cleanup_profiling_kv_cache
    if getattr(original, "_vllm_fl_cpu_guard", False):
        return

    def cleanup(self) -> None:
        if self.device.type != "cpu":
            return original(self)
        if hasattr(self, "kv_caches") and self.kv_caches:
            for index in range(len(self.kv_caches)):
                self.kv_caches[index] = None
            self.kv_caches.clear()
        if hasattr(self, "cross_layers_kv_cache"):
            self.cross_layers_kv_cache = None
            self.cross_layers_attn_backend = None
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            delattr(self, "kv_cache_config")
        self.cache_config.num_gpu_blocks = None
        for layer in self.compilation_config.static_forward_context.values():
            if hasattr(layer, "kv_cache"):
                kv_cache = layer.kv_cache
                layer.kv_cache = (
                    torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
                )
            if hasattr(layer, "impl"):
                if hasattr(layer.impl, "_k_scale_cache"):
                    layer.impl._k_scale_cache = None
                if hasattr(layer.impl, "_v_scale_cache"):
                    layer.impl._v_scale_cache = None
        gc.collect()

    cleanup._vllm_fl_cpu_guard = True
    GPUModelRunner._cleanup_profiling_kv_cache = cleanup


def _zero_cpu_attention_blocks(self, block_ids: list[int]) -> None:
    """Clear recycled attention pages for vLLM's hybrid CPU cache.

    vLLM 0.20.2 asks the runner to clear newly allocated pages whenever a
    model has Mamba/GDN layers.  Its CPU runner discards that request, based
    on the assumption that invalid attention positions are always masked.
    With a hybrid page size larger than the CPU attention kernel block, stale
    K/V data can still affect a recycled physical page.  The generic zeroer
    already knows the hybrid page mapping and intentionally skips Mamba state;
    GDN handles fresh state through ``has_initial_state`` during prefill.
    """
    zeroer = getattr(self, "_kv_block_zeroer", None)
    if zeroer is not None:
        zeroer.zero_block_ids(block_ids)


def _install_cpu_hybrid_kv_zeroing() -> None:
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    current = CPUModelRunner._zero_block_ids
    if getattr(current, "_vllm_fl_hybrid_kv_zeroing", False):
        return
    _zero_cpu_attention_blocks._vllm_fl_hybrid_kv_zeroing = True
    CPUModelRunner._zero_block_ids = _zero_cpu_attention_blocks


def install_telechat4_w4a16_post_load() -> None:
    """Prepack TeleChat4 WNA16 MoE experts after the CPU model loads."""
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    original = CPUModelRunner.load_model
    if getattr(original, "_vllm_fl_native_moe_prepack", False):
        return

    def load_with_moe_prepack(self, load_dummy_weights: bool = False):
        result = original(self, load_dummy_weights=load_dummy_weights)
        import os

        if (
            getattr(self, "device", None) is not None
            and self.device.type == "cpu"
            and os.environ.get("FL_CPU_RUNTIME_FAMILY", "qwen").lower()
            == "telechat4"
        ):
            from vllm_fl.patches.cpu_fused_moe import prepack_native_moe_weights
            from vllm_fl.models.telechat4 import (
                install_telechat4_mla_pair_decode,
            )

            prepack_native_moe_weights(self.model)
            install_telechat4_mla_pair_decode(self.model)
        return result

    load_with_moe_prepack._vllm_fl_native_moe_prepack = True
    CPUModelRunner.load_model = load_with_moe_prepack


def install_telechat4_w4a8_post_load() -> None:
    """Install only checkpoint-W4A8 post-load adapters on TeleChat4."""
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    original = CPUModelRunner.load_model
    if getattr(original, "_vllm_fl_w4a8_mla_pair_post_load", False):
        return

    def load_with_w4a8_mla_pair(self, load_dummy_weights: bool = False):
        result = original(self, load_dummy_weights=load_dummy_weights)
        import os

        if (
            getattr(self, "device", None) is not None
            and self.device.type == "cpu"
            and is_mhc_mla_w4a8_family()
        ):
            from vllm_fl.models.telechat4 import (
                install_telechat4_w4a8_gate_m1_direct,
                install_telechat4_w4a8_mla_pair_decode,
                install_telechat4_w4a8_rope_decode,
                install_telechat4_w4a8_shared_mlp_decode,
            )
            from vllm_fl.attention.cpu_mla import (
                install_cpu_mla_fp32_projections,
            )

            install_telechat4_w4a8_mla_pair_decode(self.model)
            install_telechat4_w4a8_gate_m1_direct(self.model)
            install_telechat4_w4a8_shared_mlp_decode(self.model)
            install_telechat4_w4a8_rope_decode(self.model)
            install_cpu_mla_fp32_projections(self.model)
        return result

    load_with_w4a8_mla_pair._vllm_fl_w4a8_mla_pair_post_load = True
    CPUModelRunner.load_model = load_with_w4a8_mla_pair


def _install_telechat4_mtp_draft_post_load(
    draft_model: torch.nn.Module,
) -> dict[str, int]:
    """Install draft-only adapters at the draft model's load boundary.

    ``CPUModelRunner.load_model`` owns both target and proposer loading, but
    reaching back into ``runner.drafter`` from a target post-load wrapper is
    fragile: proposer implementations are free to finish or replace their
    model independently.  Run these adapters immediately after
    ``EagleProposer.load_model`` instead, when the BF16 expert tensors and the
    shared target lm_head are both final.
    """
    existing = getattr(draft_model, "_vllm_fl_mtp_post_load_state", None)
    if existing is not None:
        return existing

    from vllm_fl.attention.cpu_mla import install_cpu_mla_fp32_projections
    from vllm_fl.spec_decode.telechat4_mtp import (
        install_telechat4_mtp_bf16_moe,
        install_telechat4_mtp_bf16_shared_mlp,
        install_telechat4_mtp_draft_lmhead_q4,
        install_telechat4_mtp_w4a8_moe,
        install_telechat4_mtp_w8_moe,
    )

    state = {
        "draft_head_q4": install_telechat4_mtp_draft_lmhead_q4(draft_model),
        "draft_moe_w4a8": install_telechat4_mtp_w4a8_moe(draft_model),
        "draft_moe_w8": install_telechat4_mtp_w8_moe(draft_model),
        "draft_moe_bf16": install_telechat4_mtp_bf16_moe(draft_model),
        "draft_shared_mlp_bf16": install_telechat4_mtp_bf16_shared_mlp(
            draft_model
        ),
        "draft_mla": install_cpu_mla_fp32_projections(draft_model),
    }
    draft_model._vllm_fl_mtp_post_load_state = state
    if any(state.values()):
        logger.info("TeleChat4 MTP draft post-load state: %s", state)
    return state


def _telechat4_mtp_greedy_sample(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Greedy-sample one TeleChat4 MTP proposal from its draft head.

    The draft-only Q4/W8-refine and the ordinary W8 head both expose their
    per-partition winners in a guarded storage header.  Calling the registered
    reduction is safe for every logits tensor: the operator validates the
    complete header and falls back to ``argmax`` when it is absent or stale.
    Keeping this helper outside the proposer class makes the plugin-owned
    contract independently testable without changing vLLM's global
    ``use_local_argmax_reduction`` setting.
    """
    model._vllm_fl_mtp_last_draft_head_w8_refined = False
    logits = model.compute_logits(hidden_states)
    refined = bool(
        getattr(model, "_vllm_fl_mtp_last_draft_head_w8_refined", False)
    )
    active = os.environ.get(
        "FL_CPU_TELECHAT4_MTP_W8_CACHED_ARGMAX_ACTIVE", "0"
    ).lower() in {"1", "true", "on"}
    if (
        active
        and hasattr(torch.ops.triton_jit_cpu, "w8_cached_argmax")
        and logits.device.type == "cpu"
        and logits.dtype == torch.bfloat16
        and logits.dim() == 2
        and logits.shape[0] == 1
        and logits.is_contiguous()
        and logits.storage_offset() * logits.element_size() >= 64
    ):
        sampled = torch.ops.triton_jit_cpu.w8_cached_argmax(logits)
        # Count only completed reductions.  The C++ operator validates the
        # header before consuming it and otherwise returns the ordinary
        # full-logits argmax, so proposal correctness is fail-closed.
        from vllm_fl.ops.cpu_w8_greedy import (
            record_w8_cached_draft_greedy_hit,
        )

        record_w8_cached_draft_greedy_hit(refined=refined)
        return sampled
    return logits.argmax(dim=-1)


def _install_cpu_dflash2_proposer() -> None:
    """Select the plugin DFlash2 proposer in vLLM's standalone CPU runner."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.cpu_attn import (
        CPUAttentionBackendImpl,
        _get_attn_isa,
    )
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    if not hasattr(CPUAttentionBackendImpl, "do_kv_cache_update"):

        def do_kv_cache_update(
            self,
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
        ) -> None:
            del layer
            key_cache, value_cache = kv_cache.unbind(0)
            block_size = key_cache.shape[-2]
            head_size = key_cache.shape[-1]
            ops.cpu_attn_reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                _get_attn_isa(key.dtype, block_size, head_size),
            )

        CPUAttentionBackendImpl.do_kv_cache_update = do_kv_cache_update

    original_init = CPUModelRunner.__init__
    if getattr(original_init, "_vllm_fl_dflash2_proposer", False):
        return

    def init_with_dflash2(self, vllm_config, device) -> None:
        original_init(self, vllm_config, device)
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or not speculative_config.use_dflash():
            return
        draft_architectures = getattr(
            speculative_config.draft_model_config.hf_config,
            "architectures",
            (),
        ) or ()
        if "DFlash2DraftModel" not in draft_architectures:
            return

        from vllm_fl.spec_decode.dflash2 import DFlash2Proposer

        self.drafter = DFlash2Proposer(vllm_config, device, self)
        self.use_aux_hidden_state_outputs = True

    init_with_dflash2._vllm_fl_dflash2_proposer = True
    CPUModelRunner.__init__ = init_with_dflash2


def _install_cpu_telechat4_mtp_proposer() -> None:
    """Use a BF16-only TeleChat4 MTP proposer on the ARM CPU runner.

    vLLM routes ``method=mtp`` through ``EagleProposer`` because MTP consumes
    the target hidden state directly.  The generic proposer is correct, but
    its draft vllm_config still carries the target W4A8 quantization config.
    TeleChat4's released MTP layer is BF16, so construct the draft model with
    a copied config whose model_config is the draft config and whose
    quant_config is explicitly None.
    """
    from vllm.config.utils import replace
    from vllm.v1.spec_decode.eagle import EagleProposer
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    original_init = CPUModelRunner.__init__
    if getattr(original_init, "_vllm_fl_telechat4_mtp_proposer", False):
        return

    class TeleChat4MTPProposer(EagleProposer):
        def model_returns_tuple(self) -> bool:
            # DeepSeek MTP returns ``(pre_norm, post_norm)`` so the proposer
            # can sample from the former and recycle the latter.  vLLM 0.24
            # detects that contract from the literal ``DeepSeekMTPModel``
            # architecture name; our plugin-owned TeleChat/XingChen aliases
            # inherit the same implementation and therefore have the same
            # tuple ABI.
            return True

        def load_model(self, target_model: torch.nn.Module) -> None:
            super().load_model(target_model)
            _install_telechat4_mtp_draft_post_load(self.model)

        def dummy_run(self, num_tokens: int, *args, **kwargs) -> None:
            # vLLM 0.24 profiles the target's complete token budget (648 for
            # the packaged PP512/TG128 profile) through the one-layer drafter.
            # In eager CPU mode this does not establish a reusable graph; it
            # only specializes very large Triton fallback shapes and can spend
            # minutes in LLVM.  The drafter proposes one row at a time for
            # MTP-1 (the two-row shape belongs to target verification), so
            # warm exactly that useful draft shape.
            return super().dummy_run(min(num_tokens, 1), *args, **kwargs)

        def propose(self, *args, **kwargs) -> torch.Tensor:
            common_attn_metadata = kwargs.get("common_attn_metadata")
            if common_attn_metadata is None and len(args) > 5:
                common_attn_metadata = args[5]
            sequence_length = None
            if common_attn_metadata is not None:
                try:
                    sequence_length = int(
                        common_attn_metadata.seq_lens_cpu.max().item()
                    )
                except (AttributeError, RuntimeError, ValueError):
                    # The gate is advisory; an unusual metadata object should
                    # retain the existing opt-in kernel behavior.
                    sequence_length = None
            from vllm_fl.spec_decode.telechat4_mtp import (
                telechat4_mtp_sequence_context,
            )

            with telechat4_mtp_sequence_context(sequence_length):
                with _telechat4_mtp_draft_mla_threads():
                    return super().propose(*args, **kwargs)

        def _create_draft_vllm_config(self):
            base = super()._create_draft_vllm_config()
            return replace(
                base,
                model_config=self.draft_model_config,
                quant_config=None,
            )

        def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return _telechat4_mtp_greedy_sample(self.model, hidden_states)

    def init_with_telechat4_mtp(self, vllm_config, device) -> None:
        original_init(self, vllm_config, device)
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or speculative_config.method != "mtp":
            return
        draft_config = speculative_config.draft_model_config
        draft_hf_config = getattr(draft_config, "hf_config", None)
        architectures = getattr(draft_hf_config, "architectures", ()) or ()
        if (
            getattr(draft_hf_config, "model_type", None)
            not in {"telechat4_mtp", "xingchen4_mtp", "xing4_0_mtp"}
            and not {
                "TeleChat4MTPModel",
                "XingChen4MTPModel",
                "Xing4_0MTPModel",
            }.intersection(architectures)
        ):
            return
        self.drafter = TeleChat4MTPProposer(vllm_config, device, self)

    init_with_telechat4_mtp._vllm_fl_telechat4_mtp_proposer = True
    CPUModelRunner.__init__ = init_with_telechat4_mtp


def _install_cpu_spec_decode_compat() -> None:
    """Bridge the vLLM 0.20.2 CPU sampler to the current MTP call ABI.

    The generic rejection sampler passes the optional synthetic-decoding
    tensors added late in the 0.20.2 cycle.  The CPU C++ implementation still
    implements the ordinary greedy/random algorithm and has the older
    signature.  Ignore those tensors for the ordinary path and fail closed if
    synthetic rejection sampling is explicitly requested.
    """
    import vllm.utils.cpu_triton_utils as cpu_tl

    greedy_kernel = cpu_tl.rejection_greedy_sample_kernel
    greedy_impl = greedy_kernel.func
    if getattr(greedy_impl, "_vllm_fl_spec_decode_abi", False):
        return

    random_kernel = cpu_tl.rejection_random_sample_kernel
    random_impl = random_kernel.func

    def greedy_compat(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
        uniform_probs=None,
        synthetic_conditional_rates=None,
        *,
        SYNTHETIC_MODE=False,
    ):
        del uniform_probs, synthetic_conditional_rates
        if SYNTHETIC_MODE:
            raise NotImplementedError(
                "synthetic speculative rejection sampling is unsupported on "
                "the vLLM 0.20.2 ARM CPU path"
            )
        return greedy_impl(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )

    def random_compat(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        synthetic_conditional_rates=None,
        *,
        NO_DRAFT_PROBS=False,
        SYNTHETIC_MODE=False,
    ):
        del synthetic_conditional_rates
        if SYNTHETIC_MODE:
            raise NotImplementedError(
                "synthetic speculative rejection sampling is unsupported on "
                "the vLLM 0.20.2 ARM CPU path"
            )
        return random_impl(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            NO_DRAFT_PROBS=NO_DRAFT_PROBS,
        )

    greedy_compat._vllm_fl_spec_decode_abi = True
    random_compat._vllm_fl_spec_decode_abi = True
    greedy_kernel.func = greedy_compat
    random_kernel.func = random_compat


def install_arm_cpu_vllm_0202_compat() -> bool:
    """Install all Python-only compatibility hooks exactly once."""
    global _INSTALLED
    if _INSTALLED:
        return False
    _require_vllm_0202()
    _install_packed_w4a8()
    _install_cpu_gemm_guard()
    _install_cpu_moe_native_silu()
    _install_cpu_mla_metadata_pin_guard()
    _install_text_only_vision_guard()
    _install_cpu_attention_block_constraint()
    _install_cpu_cleanup_guard()
    _install_cpu_hybrid_kv_zeroing()
    _install_cpu_dflash2_proposer()
    _install_cpu_telechat4_mtp_proposer()
    _install_cpu_spec_decode_compat()
    _INSTALLED = True
    return True


__all__ = [
    "install_arm_cpu_vllm_0202_compat",
    "install_telechat4_w4a16_post_load",
    "install_telechat4_w4a8_post_load",
]
