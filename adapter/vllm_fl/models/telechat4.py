"""Inference-only TeleChat4 model for the ARM CPU FL runtime.

TeleChat4 keeps the DeepSeek-V3 MLA and MoE blocks but replaces the scalar
residual path with four-stream manifold-constrained Hyper-Connections (mHC).
The implementation follows the checkpoint contract and the mHC paper:

* one fused projection produces pre/post/residual mapping logits;
* pre is sigmoid, post is twice sigmoid;
* residual mapping is projected to the Birkhoff polytope with Sinkhorn;
* attention and FFN each own an independent mHC module.
"""

from __future__ import annotations

import os
import time
from itertools import islice
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
    _get_llama_4_scaling,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.sequence import IntermediateTensors
from vllm_fl.runtime_families import (
    cpu_runtime_model_name,
    is_mhc_mla_w4a8_family,
)


class _TeleChat4MLAPairCoordinator(nn.Module):
    """Share one decode activation pack across the MLA A projections."""

    def __init__(
        self,
        first: nn.Module,
        second: nn.Module,
        *,
        active_env: str | None = None,
    ) -> None:
        super().__init__()
        self.first = first
        self.second = second
        self.active_env = active_env
        self._apply_pair = None
        self._cached_input = None
        self._cached_output = None

    def _project(self, value: torch.Tensor) -> torch.Tensor | None:
        if self.active_env is not None and os.environ.get(
            self.active_env, "1"
        ).lower() in {"0", "false", "off"}:
            self.clear_cache()
            return None
        # The pair is deliberately resolved after weight loading.  Keeping
        # this lazy also makes model construction and CPU prefill unchanged.
        if self._apply_pair is None:
            try:
                from flag_gems.runtime.backend._arm.q4.linear import (
                    prepare_vllm_q4_g128_pair,
                )

                self._apply_pair = prepare_vllm_q4_g128_pair(
                    self.first, self.second
                )
            except (ImportError, AttributeError, OSError, RuntimeError, ValueError):
                self._apply_pair = False
        if not self._apply_pair:
            return None
        if value is self._cached_input and self._cached_output is not None:
            return self._cached_output
        output = self._apply_pair(value)
        if output is None:
            return None
        self._cached_input = value
        self._cached_output = output
        return output

    def clear_cache(self) -> None:
        self._cached_input = None
        self._cached_output = None


class _TeleChat4MLAPairView(nn.Module):
    def __init__(
        self,
        coordinator: _TeleChat4MLAPairCoordinator,
        index: int,
        width: int,
    ) -> None:
        super().__init__()
        self.coordinator = coordinator
        self.index = index
        self.width = width

    def forward(self, value: torch.Tensor):
        joined = self.coordinator._project(value)
        if joined is None:
            original = (
                self.coordinator.first
                if self.index == 0
                else self.coordinator.second
            )
            return original(value)
        offset = 0 if self.index == 0 else self.coordinator.first.output_size
        return joined.narrow(-1, offset, self.width), None


def install_telechat4_mla_pair_decode(model: nn.Module) -> int:
    """Install the optional W4A16 MLA A-projection pair after model load."""
    if getattr(model, "_vllm_fl_telechat4_mla_pair", False):
        return 0
    if os.environ.get("FL_CPU_RUNTIME_FAMILY", "qwen").lower() != "telechat4":
        return 0
    if os.environ.get("FL_CPU_TELECHAT4_MLA_PAIR_DECODE", "1").lower() in {
        "0",
        "false",
        "off",
    }:
        return 0

    installed = 0
    for module in model.modules():
        attention = getattr(module, "self_attn", None)
        if attention is None or not hasattr(attention, "q_a_proj"):
            continue
        if not hasattr(attention, "kv_a_proj_with_mqa"):
            continue
        if isinstance(attention.q_a_proj, _TeleChat4MLAPairView):
            continue
        first = attention.q_a_proj
        second = attention.kv_a_proj_with_mqa
        first_width = int(getattr(first, "output_size", 0))
        second_width = int(getattr(second, "output_size", 0))
        if first_width <= 0 or second_width <= 0:
            continue
        try:
            from flag_gems.runtime.backend._arm.q4.linear import (
                prepare_vllm_q4_g128_pair,
            )

            prepared_pair = prepare_vllm_q4_g128_pair(first, second)
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError):
            prepared_pair = None
        # Do not wrap portable W4A16 layers: the native shape marker and
        # operator are intentionally absent when FL_CPU_W4A16_NATIVE=0.
        if prepared_pair is None:
            continue
        coordinator = _TeleChat4MLAPairCoordinator(first, second)
        coordinator._apply_pair = prepared_pair
        attention.q_a_proj = _TeleChat4MLAPairView(
            coordinator, 0, first_width
        )
        attention.kv_a_proj_with_mqa = _TeleChat4MLAPairView(
            coordinator, 1, second_width
        )
        installed += 1
    model._vllm_fl_telechat4_mla_pair = True
    if installed:
        print(
            f"[vllm_fl] TeleChat4 MLA W4A16 decode pair installed "
            f"for {installed} layers",
            flush=True,
        )
    return installed


def install_telechat4_w4a8_mla_pair_decode(model: nn.Module) -> int:
    """Reuse Qwen's checkpoint-W4A8 G128 pair for TeleChat4 MLA decode.

    This path resolves the ``Dynamic4bitLinearKernel`` objects installed by
    FlagGems for compressed-tensors W4A8.  It is deliberately separate from
    :func:`install_telechat4_mla_pair_decode`, whose adapter consumes vLLM's
    WNA16/W4A16 ``weight_packed`` ABI.
    """
    if getattr(model, "_vllm_fl_telechat4_w4a8_mla_pair", False):
        return 0
    if not is_mhc_mla_w4a8_family():
        return 0
    if os.environ.get("FLAGGEMS_Q4_FUSED_GDN_G128", "0").lower() not in {
        "1",
        "true",
        "on",
    }:
        return 0

    installed = 0
    for module in model.modules():
        attention = getattr(module, "self_attn", None)
        if attention is None or not hasattr(attention, "q_a_proj"):
            continue
        if not hasattr(attention, "kv_a_proj_with_mqa"):
            continue
        if isinstance(attention.q_a_proj, _TeleChat4MLAPairView):
            continue
        first = attention.q_a_proj
        second = attention.kv_a_proj_with_mqa
        first_width = int(getattr(first, "output_size", 0))
        second_width = int(getattr(second, "output_size", 0))
        if first_width <= 0 or second_width <= 0:
            continue
        try:
            from flag_gems.runtime.backend._arm.q4.linear import (
                prepare_vllm_q4_g32_pair,
            )

            prepared_pair = prepare_vllm_q4_g32_pair(first, second)
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError):
            prepared_pair = None
        if prepared_pair is None:
            continue
        coordinator = _TeleChat4MLAPairCoordinator(
            first,
            second,
            active_env="FLAGGEMS_Q4_FUSED_GDN_G128_ACTIVE",
        )
        coordinator._apply_pair = prepared_pair
        attention.q_a_proj = _TeleChat4MLAPairView(
            coordinator, 0, first_width
        )
        attention.kv_a_proj_with_mqa = _TeleChat4MLAPairView(
            coordinator, 1, second_width
        )
        installed += 1
    model._vllm_fl_telechat4_w4a8_mla_pair = True
    if installed:
        print(
            f"[vllm_fl] TeleChat4 MLA W4A8 G128 decode pair installed "
            f"for {installed} layers",
            flush=True,
        )
    return installed


def install_telechat4_w4a8_shared_mlp_decode(model: nn.Module) -> int:
    """Fuse the G128 shared-expert MLP without touching routed weights."""
    if getattr(model, "_vllm_fl_telechat4_w4a8_shared_mlp", False):
        return 0
    if not is_mhc_mla_w4a8_family():
        return 0
    if os.environ.get(
        "FL_CPU_TELECHAT4_W4A8_SHARED_MLP", "0"
    ).lower() not in {"1", "true", "on"}:
        return 0

    try:
        from flag_gems.runtime.backend._arm.q4.linear import (
            _vllm_dynamic4bit_kernel,
            unpack_rhs_qsi4c128p_asym,
        )
    except (ImportError, AttributeError):
        return 0

    installed = 0
    sme2_enabled = os.environ.get(
        "FL_CPU_TELECHAT4_W4A8_SHARED_MLP_SME2", "0"
    ).lower() in {"1", "true", "on"}
    if sme2_enabled and not hasattr(
        torch.ops.triton_jit_cpu, "q4_linear_g128_w4a8_kai_sme2_mlp"
    ):
        raise RuntimeError("shared-expert SME2 W4A8 MLP operator is unavailable")
    for module_name, module in model.named_modules():
        if not module_name.endswith(".shared_experts"):
            continue
        if not all(
            hasattr(module, name)
            for name in ("gate_up_proj", "down_proj", "act_fn")
        ):
            continue
        gate_up_kernel = _vllm_dynamic4bit_kernel(module.gate_up_proj)
        if gate_up_kernel is None:
            continue
        apply_mlp = getattr(
            gate_up_kernel, "_flag_gems_apply_g128_mlp_weights", None
        )
        if apply_mlp is None:
            continue

        if sme2_enabled:
            down_kernel = _vllm_dynamic4bit_kernel(module.down_proj)
            gateup_shape = getattr(
                gate_up_kernel, "_flag_gems_libtriton_jit_q4_shape", None
            )
            down_shape = getattr(
                down_kernel, "_flag_gems_libtriton_jit_q4_shape", None
            )
            if down_kernel is None or gateup_shape is None or down_shape is None:
                raise RuntimeError("shared-expert SME2 weights are not G128 prepared")
            gateup_n, input_k, gateup_group = gateup_shape
            output_n, activation_k, down_group = down_shape
            if (
                gateup_group != 128
                or down_group != 128
                or gateup_n != 2 * activation_k
            ):
                raise RuntimeError("unexpected shared-expert SME2 MLP shape")
            gateup_compact = getattr(
                module.gate_up_proj, gate_up_kernel.w_q_name
            ).detach()
            down_compact = getattr(
                module.down_proj, down_kernel.w_q_name
            ).detach()
            gateup_values, gateup_scales = unpack_rhs_qsi4c128p_asym(
                gateup_compact, gateup_n, input_k
            )
            down_values, down_scales = unpack_rhs_qsi4c128p_asym(
                down_compact, output_n, activation_k
            )
            module._vllm_fl_shared_mlp_sme2_gateup = (
                torch.ops.triton_jit_cpu.q4_pack_w4a8_kai_sme2(
                    gateup_values, gateup_scales
                )[0]
            )
            module._vllm_fl_shared_mlp_sme2_down = (
                torch.ops.triton_jit_cpu.q4_pack_w4a8_kai_sme2(
                    down_values, down_scales
                )[0]
            )
            module._vllm_fl_shared_mlp_sme2_shape = (
                gateup_n,
                input_k,
                output_n,
            )

        original_forward = module.forward

        def fused_forward(self, x: torch.Tensor) -> torch.Tensor:
            rows = x.shape[0] if x.ndim == 2 else 0
            m2_enabled = os.environ.get(
                "FL_CPU_TELECHAT4_W4A8_SHARED_MLP_M2", "0"
            ).lower() in {"1", "true", "on"}
            sme2_active = os.environ.get(
                "FL_CPU_TELECHAT4_W4A8_SHARED_MLP_SME2_ACTIVE", "1"
            ).lower() in {"1", "true", "on"}
            sme2_shape = getattr(
                self, "_vllm_fl_shared_mlp_sme2_shape", None
            )
            if rows == 2 and sme2_active and sme2_shape is not None:
                gateup_n, input_k, output_n = sme2_shape
                return torch.ops.triton_jit_cpu.q4_linear_g128_w4a8_kai_sme2_mlp(
                    x,
                    self._vllm_fl_shared_mlp_sme2_gateup,
                    gateup_n,
                    input_k,
                    self._vllm_fl_shared_mlp_sme2_down,
                    output_n,
                )
            if (
                rows == 1 or (rows == 2 and m2_enabled)
            ) and x.is_contiguous():
                fused = self._vllm_fl_apply_g128_mlp(
                    self.gate_up_proj,
                    self.down_proj,
                    x,
                )
                if fused is not None:
                    return fused
            return self._vllm_fl_shared_mlp_original_forward(x)

        module._vllm_fl_apply_g128_mlp = apply_mlp
        module._vllm_fl_shared_mlp_original_forward = original_forward
        module.forward = MethodType(fused_forward, module)
        installed += 1

    model._vllm_fl_telechat4_w4a8_shared_mlp = True
    if installed:
        print(
            f"[vllm_fl] {cpu_runtime_model_name()} W4A8 fused "
            "shared-expert MLP installed "
            f"for {installed} layers",
            flush=True,
        )
    return installed


def _telechat4_w4a8_gate_direct_forward(
    self, x: torch.Tensor
) -> tuple[torch.Tensor, None] | torch.Tensor:
    """Call the already-prepared CPU GEMM handler directly for decode M=1."""
    active = os.environ.get(
        "FL_CPU_TELECHAT4_W4A8_GATE_M1_DIRECT", "0"
    ).lower() in {"1", "true", "on"}
    if (
        active
        and x.ndim == 2
        and x.shape == (1, self.input_size)
        and x.is_contiguous()
    ):
        output = self.cpu_linear(x, self.weight, None)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, None
    return self._vllm_fl_gate_original_forward(x)


def install_telechat4_w4a8_gate_m1_direct(model: nn.Module) -> int:
    """Bypass generic ReplicatedLinear dispatch around the unchanged CPU GEMM."""
    if getattr(model, "_vllm_fl_telechat4_w4a8_gate_m1_direct", False):
        return 0
    if not is_mhc_mla_w4a8_family():
        return 0

    from vllm.model_executor.layers.fused_moe.router.gate_linear import (
        GateLinear,
    )

    installed = 0
    for module_name, module in model.named_modules():
        if (
            not module_name.endswith(".mlp.gate")
            or not isinstance(module, GateLinear)
            or module.input_size != 3584
            or module.output_size != 64
            or module.bias is not None
            or not hasattr(module, "cpu_linear")
        ):
            continue
        module._vllm_fl_gate_original_forward = module.forward
        module.forward = MethodType(
            _telechat4_w4a8_gate_direct_forward, module
        )
        installed += 1

    model._vllm_fl_telechat4_w4a8_gate_m1_direct = True
    if installed:
        print(
            f"[vllm_fl] {cpu_runtime_model_name()} W4A8 direct M1 "
            f"MoE gate installed for {installed} layers",
            flush=True,
        )
    return installed


def install_telechat4_w4a8_rope_decode(model: nn.Module) -> int:
    """Route only the one-token TeleChat4 YaRN RoPE through the ARM op."""
    if getattr(model, "_vllm_fl_telechat4_w4a8_rope", False):
        return 0
    if (
        not is_mhc_mla_w4a8_family()
        or os.environ.get(
            "FL_CPU_TELECHAT4_W4A8_ROPE_NATIVE", "0"
        ).lower()
        not in {"1", "true", "on"}
    ):
        return 0
    if not hasattr(torch.ops.triton_jit_cpu, "telechat4_rope_decode"):
        raise RuntimeError(
            "native TeleChat4 RoPE requires the FlagGems "
            "telechat4_rope_decode operator"
        )

    installed = 0
    for module in model.modules():
        if type(module).__name__ != "DeepseekScalingRotaryEmbedding":
            continue
        if (
            int(getattr(module, "head_size", 0)) != 64
            or int(getattr(module, "rotary_dim", 0)) != 64
            or bool(getattr(module, "is_neox_style", True))
        ):
            continue
        if hasattr(module, "_vllm_fl_reference_forward"):
            continue
        module._vllm_fl_reference_forward = module.forward

        def native_forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor | None = None,
            offsets: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if (
                offsets is None
                and key is not None
                and query.dtype == torch.bfloat16
                and key.dtype == torch.bfloat16
                and query.shape == (1, 32, 64)
                and key.shape == (1, 1, 64)
                and self.cos_sin_cache.dtype == torch.bfloat16
            ):
                return torch.ops.triton_jit_cpu.telechat4_rope_decode(
                    positions,
                    query,
                    key,
                    self.cos_sin_cache,
                )
            return self._vllm_fl_reference_forward(
                positions, query, key, offsets
            )

        module.forward = MethodType(native_forward, module)
        installed += 1
    model._vllm_fl_telechat4_w4a8_rope = True
    if installed:
        print(
            f"[vllm_fl] {cpu_runtime_model_name()} W4A8 native decode "
            "RoPE installed "
            f"for {installed} module(s)",
            flush=True,
        )
    return installed


class TeleChat4MHC(nn.Module):
    """Token-wise manifold-constrained Hyper-Connection mapping."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_streams = int(config.num_residual_streams)
        self.num_residual_logits = self.num_streams * self.num_streams
        self.num_mapping_logits = self.num_residual_logits + 2 * self.num_streams
        self.sinkhorn_iterations = int(config.mhc_sinkhorn_iterations)
        self.clamp_min = float(config.mhc_h_res_clamp_min)
        self.clamp_max = float(config.mhc_h_res_clamp_max)
        # XingChen4's September checkpoint replaced the original eight-tensor
        # TeleChat mHC ABI with hc_fn/hc_base/hc_scale and changed Sinkhorn to
        # row-then-column normalization with hc_eps in each denominator.
        self._xingchen4_v2 = getattr(config, "model_type", "") in {"xingchen4", "xing4_0"}
        self.sinkhorn_eps = float(getattr(config, "hc_eps", 1.0e-6))
        self._native_decode = (
            is_mhc_mla_w4a8_family()
            and os.environ.get(
                "FL_CPU_TELECHAT4_W4A8_MHC_NATIVE", "0"
            ).lower()
            in {"1", "true", "on"}
        )
        self._native_prefill = self._native_decode and (
            os.environ.get(
                "FL_CPU_TELECHAT4_W4A8_MHC_PREFILL_NATIVE", "0"
            ).lower()
            in {"1", "true", "on"}
            or os.environ.get(
                "FL_CPU_TELECHAT4_W4A8_MHC_M2_NATIVE", "0"
            ).lower()
            in {"1", "true", "on"}
        )
        if self._native_decode and not all(
            hasattr(torch.ops.triton_jit_cpu, name)
            for name in ("telechat4_mhc_pre", "telechat4_mhc_post")
        ):
            raise RuntimeError(
                "native TeleChat4 mHC requires the FlagGems "
                "telechat4_mhc_pre/post operators"
            )
        if self._native_prefill and not all(
            hasattr(torch.ops.triton_jit_cpu, name)
            for name in (
                "telechat4_mhc_prefill",
                "telechat4_mhc_post_prefill",
            )
        ):
            raise RuntimeError(
                "native TeleChat4 mHC prefill requires the FlagGems "
                "telechat4_mhc_prefill/post_prefill operators"
            )
        if (
            self._native_decode
            and os.environ.get(
                "FL_CPU_TELECHAT4_W4A8_MHC_PREFILL_POST_NATIVE", "0"
            ).lower()
            in {"1", "true", "on"}
            and not hasattr(
                torch.ops.triton_jit_cpu, "telechat4_mhc_post_prefill"
            )
        ):
            raise RuntimeError(
                "native TeleChat4 mHC prefill post requires the FlagGems "
                "telechat4_mhc_post_prefill operator"
            )
        if (
            self._native_decode
            and os.environ.get(
                "FL_CPU_TELECHAT4_W4A8_MHC_PREFILL_MIX_NATIVE", "0"
            ).lower()
            in {"1", "true", "on"}
            and not hasattr(
                torch.ops.triton_jit_cpu, "telechat4_mhc_mix_prefill"
            )
        ):
            raise RuntimeError(
                "native TeleChat4 mHC prefill mix requires the FlagGems "
                "telechat4_mhc_mix_prefill operator"
            )
        dtype = torch.get_default_dtype()

        if self._xingchen4_v2:
            self.hc_fn = nn.Parameter(
                torch.empty(
                    self.num_mapping_logits,
                    self.num_streams * self.hidden_size,
                    dtype=dtype,
                )
            )
            self.hc_base = nn.Parameter(
                torch.empty(self.num_mapping_logits, dtype=dtype)
            )
            # The released checkpoint stores these three scales in FP32.
            self.hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        else:
            # Checkpoint names and shapes:
            # mapping_weight: [n^2 + 2n, nC], bias: [n^2 + 2n].
            self.mapping_weight = nn.Parameter(
                torch.empty(
                    self.num_mapping_logits,
                    self.num_streams * self.hidden_size,
                    dtype=dtype,
                )
            )
            self.bias = nn.Parameter(
                torch.empty(self.num_mapping_logits, dtype=dtype)
            )
            self.alpha_pre = nn.Parameter(torch.empty(1, dtype=dtype))
            self.alpha_post = nn.Parameter(torch.empty(1, dtype=dtype))
            self.alpha_res = nn.Parameter(torch.empty(1, dtype=dtype))

            # The deployment checkpoint carries split bias tensors in addition
            # to the fused bias. Load them for strict checkpoint accounting;
            # the fused tensor is the single source used by the forward path.
            self.bias_pre = nn.Parameter(
                torch.empty(self.num_streams, dtype=dtype)
            )
            self.bias_post = nn.Parameter(
                torch.empty(self.num_streams, dtype=dtype)
            )
            self.bias_res = nn.Parameter(
                torch.empty(self.num_residual_logits, dtype=dtype)
            )

    def _mapping_logits(self, streams: torch.Tensor) -> tuple[torch.Tensor, ...]:
        flat = streams.reshape(streams.shape[0], -1)
        flat_f32 = flat.float()
        inv_rms = torch.rsqrt(flat_f32.square().mean(dim=-1, keepdim=True) + 1e-6)
        mapping_weight = self.hc_fn if self._xingchen4_v2 else self.mapping_weight
        normalized = (flat_f32 * inv_rms).to(mapping_weight.dtype)
        projected = F.linear(normalized, mapping_weight).float()

        n = self.num_streams
        if self._xingchen4_v2:
            pre = projected[:, :n] * self.hc_scale[0] + self.hc_base[:n].float()
            post = (
                projected[:, n : 2 * n] * self.hc_scale[1]
                + self.hc_base[n : 2 * n].float()
            )
            res = (
                projected[:, 2 * n :] * self.hc_scale[2]
                + self.hc_base[2 * n :].float()
            )
        else:
            pre = projected[:, :n] * self.alpha_pre.float() + self.bias[:n].float()
            post = (
                projected[:, n : 2 * n] * self.alpha_post.float()
                + self.bias[n : 2 * n].float()
            )
            res = (
                projected[:, 2 * n :] * self.alpha_res.float()
                + self.bias[2 * n :].float()
            )
        return pre, post, res.view(-1, n, n)

    def _native_prefill_enabled(self, rows: int) -> bool:
        if not self._native_decode:
            return False
        if os.environ.get(
            "FL_CPU_TELECHAT4_W4A8_MHC_PREFILL_NATIVE", "0"
        ).lower() in {"1", "true", "on"}:
            return True
        return rows == 2 and os.environ.get(
            "FL_CPU_TELECHAT4_W4A8_MHC_M2_NATIVE", "0"
        ).lower() in {"1", "true", "on"}

    def _torch_prefill_optimized_enabled(self) -> bool:
        return self._native_decode and os.environ.get(
            "FL_CPU_TELECHAT4_W4A8_MHC_PREFILL_TORCH_OPT", "0"
        ).lower() in {"1", "true", "on"}

    def _native_prefill_post_enabled(self) -> bool:
        return self._native_decode and os.environ.get(
            "FL_CPU_TELECHAT4_W4A8_MHC_PREFILL_POST_NATIVE", "0"
        ).lower() in {"1", "true", "on"}

    def _native_prefill_mix_enabled(self) -> bool:
        return self._native_decode and os.environ.get(
            "FL_CPU_TELECHAT4_W4A8_MHC_PREFILL_MIX_NATIVE", "0"
        ).lower() in {"1", "true", "on"}

    def _native_prefill_sinkhorn_enabled(self) -> bool:
        return (
            self._native_decode
            and os.environ.get(
                "FL_CPU_TELECHAT4_W4A8_MHC_PREFILL_SINKHORN_NATIVE", "0"
            ).lower()
            in {"1", "true", "on"}
            and hasattr(torch.ops.triton_jit_cpu, "telechat4_mhc_sinkhorn")
        )

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clamp(self.clamp_min, self.clamp_max)
        if self._xingchen4_v2:
            logits = logits - logits.amax(dim=-1, keepdim=True)
        matrix = logits.exp()
        tiny = torch.finfo(matrix.dtype).tiny
        for _ in range(self.sinkhorn_iterations):
            if self._xingchen4_v2:
                matrix = matrix / (
                    matrix.sum(dim=-1, keepdim=True) + self.sinkhorn_eps
                )
                matrix = matrix / (
                    matrix.sum(dim=-2, keepdim=True) + self.sinkhorn_eps
                )
            else:
                matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(tiny)
                matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(tiny)
        return matrix

    def _sinkhorn_inplace(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clamp(self.clamp_min, self.clamp_max)
        if self._xingchen4_v2:
            logits = logits - logits.amax(dim=-1, keepdim=True)
        matrix = logits.exp()
        tiny = torch.finfo(matrix.dtype).tiny
        for _ in range(self.sinkhorn_iterations):
            if self._xingchen4_v2:
                matrix.div_(matrix.sum(dim=-1, keepdim=True) + self.sinkhorn_eps)
                matrix.div_(matrix.sum(dim=-2, keepdim=True) + self.sinkhorn_eps)
            else:
                matrix.div_(matrix.sum(dim=-2, keepdim=True).clamp_min(tiny))
                matrix.div_(matrix.sum(dim=-1, keepdim=True).clamp_min(tiny))
        return matrix

    def pre(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._xingchen4_v2:
            mapping_weight = self.hc_fn
            bias = self.hc_base
            alpha_pre = self.hc_scale[0:1]
            alpha_post = self.hc_scale[1:2]
            alpha_res = self.hc_scale[2:3]
            variant = 2
            sinkhorn_eps = self.sinkhorn_eps
        else:
            mapping_weight = self.mapping_weight
            bias = self.bias
            alpha_pre = self.alpha_pre
            alpha_post = self.alpha_post
            alpha_res = self.alpha_res
            variant = 1
            sinkhorn_eps = 0.0
        if self._native_decode and streams.shape[0] == 1:
            return torch.ops.triton_jit_cpu.telechat4_mhc_pre(
                streams.contiguous(),
                mapping_weight,
                bias,
                alpha_pre,
                alpha_post,
                alpha_res,
                self.sinkhorn_iterations,
                self.clamp_min,
                self.clamp_max,
                variant,
                sinkhorn_eps,
            )
        if self._native_prefill_enabled(streams.shape[0]) and streams.shape[0] > 1:
            return torch.ops.triton_jit_cpu.telechat4_mhc_prefill(
                streams.contiguous(),
                mapping_weight,
                bias,
                alpha_pre,
                alpha_post,
                alpha_res,
                self.sinkhorn_iterations,
                self.clamp_min,
                self.clamp_max,
                variant,
                sinkhorn_eps,
            )
        pre_logits, post_logits, res_logits = self._mapping_logits(streams)
        h_pre = torch.sigmoid(pre_logits)
        h_post = 2.0 * torch.sigmoid(post_logits)
        torch_prefill_optimized = (
            self._torch_prefill_optimized_enabled() and streams.shape[0] > 1
        )
        if (
            streams.shape[0] > 1
            and self._native_prefill_sinkhorn_enabled()
        ):
            h_res_logits = res_logits.clamp(self.clamp_min, self.clamp_max)
            if self._xingchen4_v2:
                h_res_logits = h_res_logits - h_res_logits.amax(
                    dim=-1, keepdim=True
                )
            h_res = h_res_logits.exp()
            h_res = torch.ops.triton_jit_cpu.telechat4_mhc_sinkhorn(
                h_res, self.sinkhorn_iterations, variant, sinkhorn_eps
            )
        else:
            h_res = (
                self._sinkhorn_inplace(res_logits)
                if torch_prefill_optimized
                else self._sinkhorn(res_logits)
            )
        if torch_prefill_optimized:
            combine_bmm = os.environ.get(
                "FL_CPU_TELECHAT4_W4A8_MHC_PREFILL_COMBINED_BMM", "0"
            ).lower() in {"1", "true", "on"}
            if self._native_prefill_mix_enabled():
                combined_coefficients = torch.cat(
                    (h_pre.unsqueeze(1), h_res), dim=1
                )
                combined = torch.ops.triton_jit_cpu.telechat4_mhc_mix_prefill(
                    combined_coefficients, streams.contiguous()
                )
                branch_input = combined[:, 0]
                mixed_streams = combined[:, 1:]
            elif combine_bmm:
                combined_coefficients = torch.cat(
                    (h_pre.unsqueeze(1), h_res), dim=1
                )
                combined = torch.bmm(
                    combined_coefficients, streams.float()
                )
                branch_input = combined[:, 0]
                mixed_streams = combined[:, 1:]
            else:
                streams_f32 = streams.float()
                branch_input = torch.bmm(
                    h_pre.unsqueeze(1), streams_f32
                ).squeeze(1)
                mixed_streams = torch.bmm(h_res, streams_f32)
        else:
            branch_input = torch.einsum(
                "ts,tsc->tc", h_pre, streams.float()
            )
            mixed_streams = torch.einsum(
                "tij,tjc->tic", h_res, streams.float()
            )
        return (
            branch_input.to(streams.dtype),
            mixed_streams.to(streams.dtype),
            h_post,
        )

    def post(
        self,
        branch_output: torch.Tensor,
        streams: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        if self._native_decode and streams.shape[0] == 1:
            return torch.ops.triton_jit_cpu.telechat4_mhc_post(
                branch_output.contiguous(), streams.contiguous(), h_post
            )
        if (
            streams.shape[0] > 1
            and (
                self._native_prefill_enabled(streams.shape[0])
                or self._native_prefill_post_enabled()
            )
        ):
            return torch.ops.triton_jit_cpu.telechat4_mhc_post_prefill(
                branch_output.contiguous(), streams.contiguous(), h_post
            )
        update = h_post.unsqueeze(-1) * branch_output.float().unsqueeze(1)
        streams_f32 = streams.float()
        if self._torch_prefill_optimized_enabled() and streams.shape[0] > 1:
            streams_f32.add_(update)
            return streams_f32.to(streams.dtype)
        return (streams_f32 + update).to(streams.dtype)

    def forward(
        self, streams: torch.Tensor, branch
    ) -> torch.Tensor:
        branch_input, streams, h_post = self.pre(streams)
        return self.post(branch(branch_input), streams, h_post)


class TeleChat4DecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, vllm_config, prefix: str, **kwargs) -> None:
        super().__init__(vllm_config, prefix, **kwargs)
        config = vllm_config.model_config.hf_config
        self.attn_hc = TeleChat4MHC(config)
        self.ffn_hc = TeleChat4MHC(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if residual is not None:
            raise ValueError("TeleChat4 keeps residual state inside mHC streams")

        profile = (
            os.environ.get("FL_CPU_TELECHAT_PROFILE", "0").lower()
            in {"1", "true", "on"}
            and hidden_states.shape[0] == 1
        )
        started = time.perf_counter() if profile else 0.0
        attn_input, hidden_states, h_post = self.attn_hc.pre(hidden_states)
        attn_input = self.input_layernorm(attn_input)
        attn_pre_done = time.perf_counter() if profile else 0.0
        attn_kwargs = {"positions": positions, "hidden_states": attn_input}
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        attn_output = self.self_attn(**attn_kwargs)
        attn_done = time.perf_counter() if profile else 0.0
        hidden_states = self.attn_hc.post(attn_output, hidden_states, h_post)

        ffn_input, hidden_states, h_post = self.ffn_hc.pre(hidden_states)
        ffn_input = self.post_attention_layernorm(ffn_input)
        ffn_pre_done = time.perf_counter() if profile else 0.0
        ffn_output = self.mlp(ffn_input)
        ffn_done = time.perf_counter() if profile else 0.0
        hidden_states = self.ffn_hc.post(ffn_output, hidden_states, h_post)
        if profile:
            prefix = getattr(self, "prefix", "")
            print(
                "[vllm_fl] telechat4_profile "
                f"layer={prefix} "
                f"attn_pre={attn_pre_done - started:.6f} "
                f"attn={attn_done - attn_pre_done:.6f} "
                f"attn_post={ffn_pre_done - attn_done:.6f} "
                f"mlp={ffn_done - ffn_pre_done:.6f} "
                f"mlp_post={time.perf_counter() - ffn_done:.6f}",
                flush=True,
            )
        return hidden_states, None


class TeleChat4Model(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.num_streams = int(config.num_residual_streams)
        self.vocab_size = config.vocab_size

        # DeepseekV2Model.load_weights contains the vLLM 0.24-aware packed
        # projection and routed-expert mapping.  TeleChat4 replaces the model
        # body instead of subclassing DeepseekV2Model, so retain the two bits
        # of loader state that method expects and delegate to it below.
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        self.use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )

        if get_pp_group().world_size != 1:
            raise ValueError("TeleChat4 CPU bring-up currently requires pipeline size 1")

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: TeleChat4DecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size * self.num_streams
        )
        self.aux_hidden_state_layers = tuple[int, ...]()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load split dense projections and per-expert W4A8 tensors on 0.24."""
        return DeepseekV2Model.load_weights(self, weights)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("TeleChat4 CPU bring-up currently requires pipeline size 1")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_input_ids(input_ids)

        hidden_states = inputs_embeds.unsqueeze(1).expand(
            -1, self.num_streams, -1
        ).contiguous()

        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        if llama_4_scaling_config is None:
            llama_4_scaling = None
        else:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config[
                    "original_max_position_embeddings"
                ],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, _ = layer(
                positions, hidden_states, None, llama_4_scaling
            )

        if getattr(self.config, "model_type", "") in {"xingchen4", "xing4_0"}:
            hidden_states = hidden_states.mean(dim=1)
        else:
            hidden_states = hidden_states.sum(dim=1)
        return self.norm(hidden_states)


class TeleChat4ForCausalLM(DeepseekV2ForCausalLM):
    model_cls = TeleChat4Model
