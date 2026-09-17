"""TeleChat4 single-layer MTP support for the ARM CPU runtime.

The released TeleChat4 draft checkpoint is not a small standalone model.  It
contains the normal 40-layer BF16 target weights plus one extra
``model.layers.40`` MTP layer.  vLLM's DeepSeek MTP implementation already
matches that checkpoint layout (``enorm``, ``hnorm``, ``eh_proj`` and a
standard MLA/MoE decoder block), so this module reuses its weight mapping and
only fixes the two configuration details that are different for TeleChat4:

* the draft config must resolve to a registered ``TeleChat4MTPModel``;
* the MTP block must be built with the unquantized draft config.  The target
  model is W4A8, but the MTP tensors in the supplied checkpoint are BF16.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MethodType

import torch
from vllm.config.utils import replace
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP


_BF16_SHARED_MLP_HITS = 0
_BF16_MOE_HITS = 0
_BF16_MOE_M1_CALLS = 0
_BF16_MOE_M2_CALLS = 0
_BF16_MOE_OTHER_CALLS = 0
_W4A8_MOE_HITS = 0
_W8_MOE_HITS = 0
_MTP_SEQUENCE_LENGTH = ContextVar(
    "telechat4_mtp_sequence_length", default=None
)


@contextmanager
def telechat4_mtp_sequence_context(sequence_length: int | None):
    """Expose the current draft sequence length to opt-in draft kernels."""
    token = _MTP_SEQUENCE_LENGTH.set(sequence_length)
    try:
        yield
    finally:
        _MTP_SEQUENCE_LENGTH.reset(token)


def telechat4_mtp_w4a8_moe_active() -> bool:
    """Enable draft W4A8 MoE only within its validated context window."""
    active = os.environ.get("FL_CPU_TELECHAT4_MTP_W4A8_MOE", "0").lower()
    if active not in {"1", "true", "on"}:
        return False
    keep_bf16 = os.environ.get(
        "FL_CPU_TELECHAT4_MTP_W4A8_MOE_KEEP_BF16", "1"
    ).lower() in {"1", "true", "on"}
    # Releasing the source weights is an explicit memory-first deployment
    # mode.  It cannot fall back after packing, so retain its legacy always-on
    # behavior; the launcher defaults to keeping BF16 for the safe gate.
    if not keep_bf16:
        return True
    configured = os.environ.get(
        "FL_CPU_TELECHAT4_MTP_W4A8_MOE_MAX_SEQ_LEN", "256"
    ).strip()
    try:
        max_sequence_length = int(configured)
    except ValueError as exc:
        raise ValueError(
            "FL_CPU_TELECHAT4_MTP_W4A8_MOE_MAX_SEQ_LEN must be a positive integer"
        ) from exc
    if max_sequence_length <= 0:
        raise ValueError(
            "FL_CPU_TELECHAT4_MTP_W4A8_MOE_MAX_SEQ_LEN must be a positive integer"
        )
    sequence_length = _MTP_SEQUENCE_LENGTH.get()
    return sequence_length is None or sequence_length <= max_sequence_length


def telechat4_mtp_w4a8_moe_hits() -> int:
    return _W4A8_MOE_HITS


def telechat4_mtp_bf16_shared_mlp_hits() -> int:
    return _BF16_SHARED_MLP_HITS


def telechat4_mtp_bf16_moe_hits() -> int:
    return _BF16_MOE_HITS


def telechat4_mtp_bf16_moe_row_calls() -> tuple[int, int, int]:
    return (_BF16_MOE_M1_CALLS, _BF16_MOE_M2_CALLS, _BF16_MOE_OTHER_CALLS)


class TeleChat4MTP(DeepSeekMTP):
    """DeepSeek-compatible TeleChat4 MTP head with BF16 draft weights."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        speculative_config = vllm_config.speculative_config
        if speculative_config is None:
            raise ValueError("TeleChat4MTP requires a speculative configuration")

        draft_model_config = speculative_config.draft_model_config
        if draft_model_config is None:
            raise ValueError("TeleChat4MTP is missing draft_model_config")

        # The supplied ModelScope snapshot stores the MTP layer in the last
        # safetensors shard.  Restrict the loader to shards containing the
        # extra layer; otherwise the loader needlessly walks the entire BF16
        # 40-layer target checkpoint before discarding every tensor.  If a
        # future export has no index or uses a different layout, leave the
        # normal loader behavior unchanged.
        self.allow_patterns_overrides = self._find_mtp_weight_shards(
            draft_model_config.model,
            int(getattr(draft_model_config.hf_config, "num_hidden_layers", 0)),
        )

        # The target vllm_config carries the W4A8 quantization config.  Passing
        # it into the MTP block would construct quantized linear/MoE modules
        # even though the draft checkpoint stores the MTP layer in BF16.  Keep
        # all scheduler/cache/speculation fields intact while replacing only
        # the model and quantization fields used by DeepSeekMTP's constructor.
        draft_vllm_config = replace(
            vllm_config,
            model_config=draft_model_config,
            quant_config=None,
        )
        super().__init__(vllm_config=draft_vllm_config, prefix=prefix)
        # FlagGems installs its unquantized-linear policy before vLLM runs
        # process_weights_after_loading().  Prefixes such as
        # ``model.layers.40`` do not identify this separately loaded model as
        # a drafter, so mark the complete subtree while construction is still
        # unambiguous.  ParallelLMHead and embeddings retain their dedicated
        # routing rules; all other BF16 draft weights stay out of target Q4.
        for module in self.modules():
            module._flag_gems_spec_draft = True

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        if os.environ.get("FL_DEBUG_TELECHAT4_MTP", "0") == "1":
            print(
                "[telechat4-mtp-debug] "
                f"quant_config={self.quant_config!r} "
                f"loaded={len(loaded)}",
                flush=True,
            )
            for name, parameter in self.named_parameters():
                if "kv_b_proj" in name or "shared_head.head" in name:
                    quant_method = getattr(parameter, "quant_method", None)
                    print(
                        "[telechat4-mtp-debug] "
                        f"{name}: shape={tuple(parameter.shape)} "
                        f"dtype={parameter.dtype} "
                        f"quant={getattr(getattr(parameter, 'quant_method', None), '__class__', type(None)).__name__} "
                        f"param_attrs={{'quant_method': {getattr(quant_method, '__class__', type(None)).__name__}}}",
                        flush=True,
                    )
            for module_name, module in self.named_modules():
                if module_name.endswith("kv_b_proj"):
                    print(
                        "[telechat4-mtp-debug] module "
                        f"{module_name}: weight={tuple(getattr(module, 'weight').shape)} "
                        f"quant_config={getattr(module, 'quant_config', None)!r} "
                        f"quant_method={getattr(getattr(module, 'quant_method', None), '__class__', type(None)).__name__}",
                        flush=True,
                    )
        return loaded

    @staticmethod
    def _find_mtp_weight_shards(
        model_path: str,
        mtp_layer_idx: int,
    ) -> list[str] | None:
        index_path = Path(model_path) / "model.safetensors.index.json"
        if not index_path.is_file():
            return None
        try:
            weight_map = json.loads(index_path.read_text())[
                "weight_map"
            ]
            prefix = f"model.layers.{mtp_layer_idx}."
            shards = sorted(
                {
                    filename
                    for name, filename in weight_map.items()
                    if name.startswith(prefix)
                }
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return shards or None


class XingChen4MTP(TeleChat4MTP):
    """XingChen4 alias for the shared single-layer mHC/MLA MTP contract."""

    pass


def _quantize_symmetric_g128(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one BF16 expert matrix to checkpoint-compatible W4/G128."""
    if weight.ndim != 2 or weight.shape[1] % 128:
        raise ValueError("TeleChat4 MTP W4A8 requires a 2D G128 weight")
    output_features, input_features = weight.shape
    groups = input_features // 128
    grouped = weight.float().view(output_features, groups, 128)
    absmax = torch.maximum(-grouped.amin(dim=-1), grouped.amax(dim=-1))
    # compressed-tensors symmetric INT4 uses [-8, 7], whose range is 15.
    # Its scale is max_abs / (range / 2), then stored as BF16 for G128 KAI.
    scales = (absmax / 7.5).to(torch.bfloat16)
    scales.masked_fill_(scales == 0, 1.0)
    quantized = torch.round(grouped / scales.float().unsqueeze(-1))
    quantized.clamp_(-8, 7)
    return quantized.to(torch.int8).view_as(weight), scales


def _pack_mtp_experts_w4a8(
    weight: torch.Tensor, *, use_p4: bool = False
) -> torch.Tensor:
    """Quantize and KAI-pack a BF16 ``[E,N,K]`` expert tensor by expert."""
    if weight.ndim != 3 or weight.dtype != torch.bfloat16:
        raise ValueError("TeleChat4 MTP experts must be BF16 [E,N,K]")
    experts, output_features, input_features = weight.shape
    packed = []
    for expert in range(experts):
        quantized, scales = _quantize_symmetric_g128(weight[expert])
        if use_p4:
            packed.append(
                torch.ops.triton_jit_cpu.q4_pack_w4a8_kai_prefill_16x4(
                    quantized.unsqueeze(0), scales.unsqueeze(0)
                )[0]
            )
        else:
            unsigned = quantized.add(8)
            nibbles = (
                (unsigned[:, 1::2] << 4) | unsigned[:, ::2]
            ).to(torch.uint8)
            packed.append(
                torch.ops.aten._dyn_quant_pack_4bit_weight(
                    nibbles,
                    scales,
                    None,
                    128,
                    input_features,
                    output_features,
                )
            )
    return torch.stack(packed)


def _pack_mtp_experts_w8(weight: torch.Tensor) -> torch.Tensor:
    """Per-channel W8-pack a BF16 ``[E,N,K]`` draft expert tensor."""
    if weight.ndim != 3 or weight.dtype != torch.bfloat16:
        raise ValueError("TeleChat4 MTP W8 experts must be BF16 [E,N,K]")
    from flag_gems.runtime.backend._arm.q4.linear import (
        prepare_w8_weight_kai,
    )

    return torch.stack(
        [prepare_w8_weight_kai(expert) for expert in weight.unbind(0)]
    )


def _apply_mtp_w8_moe4(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Run per-channel W8 experts and retain per-row reduction order."""
    if x.ndim != 2 or x.shape[0] not in (1, 2):
        raise ValueError("TeleChat4 MTP W8 MoE supports one or two rows")
    row_outputs = []
    for row in range(x.shape[0]):
        routed_outputs = []
        row_input = x[row : row + 1].contiguous()
        for route in range(4):
            expert = int(topk_ids[row, route])
            gate_up = torch.ops.triton_jit_cpu.w8_linear_kai(
                row_input,
                layer.w13_weight_w8[expert],
                layer.w13_out_features,
                layer.w13_in_features,
            )
            intermediate = torch.nn.functional.silu(
                gate_up[:, : layer.w2_in_features]
            ) * gate_up[:, layer.w2_in_features :]
            routed_outputs.append(
                torch.ops.triton_jit_cpu.w8_linear_kai(
                    intermediate.contiguous(),
                    layer.w2_weight_w8[expert],
                    layer.w2_out_features,
                    layer.w2_in_features,
                )
            )
        routed = torch.stack(routed_outputs, dim=1).to(topk_weights.dtype)
        # Sum only this row's four routes.  Keeping each row independent
        # matches vLLM's CPU reduction and avoids cross-row reassociation.
        row_outputs.append(
            routed.mul_(topk_weights[row : row + 1].unsqueeze(-1)).sum(dim=1)
        )
    return torch.cat(row_outputs, dim=0).to(x.dtype)


def install_telechat4_mtp_w8_moe(model: torch.nn.Module) -> int:
    """Install an opt-in per-channel W8 cache for the BF16 draft MoE."""
    active_env = "FL_CPU_TELECHAT4_MTP_W8_MOE"
    if os.environ.get(active_env, "0").lower() not in {"1", "true", "on"}:
        return 0
    conflicts = (
        "FL_CPU_TELECHAT4_MTP_W4A8_MOE",
        "FL_CPU_TELECHAT4_MTP_BF16_MOE_NATIVE",
    )
    if any(
        os.environ.get(name, "0").lower() in {"1", "true", "on"}
        for name in conflicts
    ):
        raise RuntimeError(
            "TeleChat4 MTP W8, W4A8 and native BF16 MoE are mutually exclusive"
        )
    if not hasattr(torch.ops.triton_jit_cpu, "w8_linear_kai"):
        raise RuntimeError("TeleChat4 MTP W8 requires w8_linear_kai")

    from vllm.model_executor.layers.fused_moe.cpu_fused_moe import (
        select_experts,
    )
    from vllm_fl.patches.cpu_w4a8_moe import _select_experts_for_layer

    installed = 0
    for module_name, layer in model.named_modules():
        if (
            not module_name.endswith(".mtp_block.mlp.experts")
            or getattr(layer, "_vllm_fl_mtp_w8", False)
            or not hasattr(layer, "w13_weight")
            or not hasattr(layer, "w2_weight")
        ):
            continue
        w13 = layer.w13_weight.detach()
        w2 = layer.w2_weight.detach()
        if (
            w13.dtype != torch.bfloat16
            or w2.dtype != torch.bfloat16
            or tuple(w13.shape) != (64, 2048, 3584)
            or tuple(w2.shape) != (64, 3584, 1024)
        ):
            raise RuntimeError(
                "unexpected TeleChat4 MTP W8 expert layout: "
                f"w13={tuple(w13.shape)}/{w13.dtype}, "
                f"w2={tuple(w2.shape)}/{w2.dtype}"
            )
        layer.register_parameter(
            "w13_weight_w8",
            torch.nn.Parameter(_pack_mtp_experts_w8(w13), requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_w8",
            torch.nn.Parameter(_pack_mtp_experts_w8(w2), requires_grad=False),
        )
        layer.w13_in_features = int(w13.shape[2])
        layer.w13_out_features = int(w13.shape[1])
        layer.w2_in_features = int(w2.shape[2])
        layer.w2_out_features = int(w2.shape[1])
        original_apply = layer.runner._apply_quant_method

        def apply_w8(
            self,
            layer,
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids=None,
            *,
            _original_apply=original_apply,
        ):
            x = hidden_states
            active = os.environ.get(active_env, "0").lower() in {
                "1",
                "true",
                "on",
            }
            if (
                not active
                or x.ndim != 2
                or x.shape[0] not in (1, 2)
                or x.shape[1] != 3584
                or layer.apply_router_weight_on_input
            ):
                return _original_apply(
                    layer=layer,
                    hidden_states=x,
                    router_logits=router_logits,
                    shared_experts_input=shared_experts_input,
                    input_ids=input_ids,
                )
            from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
                SharedExpertsOrder,
            )

            self._maybe_apply_shared_experts(
                shared_experts_input, SharedExpertsOrder.NO_OVERLAP
            )
            topk_weights, topk_ids = _select_experts_for_layer(
                select_experts, layer, x, router_logits
            )
            global _W8_MOE_HITS
            _W8_MOE_HITS += 1
            fused_output = _apply_mtp_w8_moe4(
                layer,
                x.contiguous(),
                topk_ids.contiguous(),
                topk_weights.contiguous(),
            )
            self._maybe_apply_shared_experts(
                shared_experts_input,
                SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
            )
            shared_output = (
                self._shared_experts.output
                if self._shared_experts is not None
                else None
            )
            return shared_output, fused_output

        layer.runner._apply_quant_method = MethodType(
            apply_w8, layer.runner
        )
        layer._vllm_fl_mtp_w8 = True
        installed += 1
    if installed == 0:
        raise RuntimeError(
            "TeleChat4 MTP W8 was requested but no compatible draft MoE "
            "layer was found"
        )
    return installed


def telechat4_mtp_w8_moe_hits() -> int:
    return _W8_MOE_HITS


def install_telechat4_mtp_w4a8_moe(model: torch.nn.Module) -> int:
    """Pack only the BF16 MTP routed experts into the ARM W4A8 KAI ABI."""
    if os.environ.get("FL_CPU_TELECHAT4_MTP_W4A8_MOE", "0").lower() not in {
        "1",
        "true",
        "on",
    }:
        return 0
    if os.environ.get(
        "FL_CPU_TELECHAT4_MTP_BF16_MOE_NATIVE", "0"
    ).lower() in {"1", "true", "on"}:
        raise RuntimeError(
            "TeleChat4 MTP W4A8 and native BF16 MoE are mutually exclusive"
        )
    use_p4 = os.environ.get(
        "FL_CPU_TELECHAT4_MTP_W4A8_MOE_P4",
        os.environ.get("FL_CPU_TELECHAT4_W4A8_MOE_KAI_P4", "0"),
    ).lower() in {"1", "true", "on"}
    required_ops = ["q4_linear_g128_w4a8_kai_moe4_packed_decode"]
    if use_p4:
        required_ops.extend(
            [
                "q4_pack_w4a8_kai_prefill_16x4",
                "q4_linear_g128_w4a8_kai_moe4_packed_decode_p4",
                "q4_linear_g128_w4a8_kai_moe4_m2_p4",
                "q4_linear_g128_w4a8_kai_moe4_prefill_16x4",
            ]
        )
    missing_ops = [
        name for name in required_ops
        if not hasattr(torch.ops.triton_jit_cpu, name)
    ]
    if missing_ops:
        raise RuntimeError(
            "TeleChat4 MTP W4A8 requires " + ", ".join(missing_ops)
        )
    keep_bf16 = os.environ.get(
        "FL_CPU_TELECHAT4_MTP_W4A8_MOE_KEEP_BF16", "1"
    ).lower() in {"1", "true", "on"}
    if not keep_bf16 and not hasattr(
        torch.ops.triton_jit_cpu,
        "q4_linear_g128_w4a8_kai_moe4_prefill",
    ):
        raise RuntimeError(
            "releasing MTP BF16 experts requires the KAI W4A8 prefill op"
        )

    from vllm.model_executor.layers.fused_moe.cpu_fused_moe import (
        select_experts,
    )
    from vllm_fl.patches.cpu_w4a8_moe import _select_experts_for_layer

    installed = 0
    expert_candidates = []
    for module_name, layer in model.named_modules():
        if hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight"):
            expert_candidates.append(
                (
                    module_name,
                    tuple(layer.w13_weight.shape),
                    str(layer.w13_weight.dtype),
                    tuple(layer.w2_weight.shape),
                    str(layer.w2_weight.dtype),
                )
            )
        is_legacy_experts = module_name.endswith(".mtp_block.mlp.experts")
        is_v024_routed_experts = module_name.endswith(
            ".mtp_block.mlp.experts.routed_experts"
        )
        if (
            not (is_legacy_experts or is_v024_routed_experts)
            or getattr(layer, "_vllm_fl_mtp_w4a8", False)
            or not hasattr(layer, "w13_weight")
            or not hasattr(layer, "w2_weight")
        ):
            continue
        w13 = layer.w13_weight.detach()
        w2 = layer.w2_weight.detach()
        if (
            w13.ndim != 3
            or w2.ndim != 3
            or w13.dtype != torch.bfloat16
            or w2.dtype != torch.bfloat16
            or w13.shape[0] != w2.shape[0]
            or w13.shape[1] != 2 * w2.shape[2]
            or w13.shape[2] != w2.shape[1]
            or w13.shape[2] % 128
            or w2.shape[2] % 128
        ):
            raise RuntimeError(
                "unexpected TeleChat4 MTP routed-expert layout: "
                f"w13={tuple(w13.shape)}/{w13.dtype}, "
                f"w2={tuple(w2.shape)}/{w2.dtype}"
            )
        if not keep_bf16 and layer.apply_router_weight_on_input:
            raise RuntimeError(
                "cannot release MTP BF16 experts with input-side router weights"
            )

        w13_packed = _pack_mtp_experts_w4a8(w13, use_p4=use_p4)
        w2_packed = _pack_mtp_experts_w4a8(w2, use_p4=use_p4)
        layer.register_parameter(
            "w13_weight_packed",
            torch.nn.Parameter(w13_packed, requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_packed",
            torch.nn.Parameter(w2_packed, requires_grad=False),
        )
        layer.w13_in_features = int(w13.shape[2])
        layer.w13_out_features = int(w13.shape[1])
        layer.w2_in_features = int(w2.shape[2])
        layer.w2_out_features = int(w2.shape[1])

        # vLLM 0.24 split the old FusedMoE object into an MoERunner parent
        # and a RoutedExperts child that owns the weights.  Older versions
        # expose that runner through ``layer.runner``.
        if is_v024_routed_experts:
            runner_name = module_name.rsplit(".routed_experts", 1)[0]
            runner = model.get_submodule(runner_name)
        else:
            runner = layer.runner
        original_apply = runner._apply_quant_method

        def apply_w4a8(
            self,
            *args,
            _original_apply=original_apply,
            _expert_layer=layer,
            _is_v024=is_v024_routed_experts,
            **kwargs,
        ):
            if _is_v024:
                layer = _expert_layer
                hidden_states = kwargs.get(
                    "hidden_states", args[0] if len(args) > 0 else None
                )
                router_logits = kwargs.get(
                    "router_logits", args[1] if len(args) > 1 else None
                )
                shared_experts_input = kwargs.get(
                    "shared_experts_input", args[2] if len(args) > 2 else None
                )
                input_ids = kwargs.get(
                    "input_ids", args[3] if len(args) > 3 else None
                )
            else:
                layer = kwargs.get("layer", args[0] if len(args) > 0 else None)
                hidden_states = kwargs.get(
                    "hidden_states", args[1] if len(args) > 1 else None
                )
                router_logits = kwargs.get(
                    "router_logits", args[2] if len(args) > 2 else None
                )
                shared_experts_input = kwargs.get(
                    "shared_experts_input", args[3] if len(args) > 3 else None
                )
                input_ids = kwargs.get(
                    "input_ids", args[4] if len(args) > 4 else None
                )
            x = hidden_states
            rows = x.shape[0]
            active = telechat4_mtp_w4a8_moe_active()
            if (
                not active
                or layer.apply_router_weight_on_input
            ):
                if not keep_bf16:
                    raise RuntimeError(
                        "MTP BF16 expert fallback is unavailable after "
                        "production W4A8 packing"
                    )
                fallback_kwargs = dict(
                    hidden_states=x,
                    router_logits=router_logits,
                    shared_experts_input=shared_experts_input,
                    input_ids=input_ids,
                )
                if not _is_v024:
                    fallback_kwargs["layer"] = layer
                return _original_apply(**fallback_kwargs)
            from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
                SharedExpertsOrder,
            )

            self._maybe_apply_shared_experts(
                shared_experts_input, SharedExpertsOrder.NO_OVERLAP
            )
            global _W4A8_MOE_HITS
            _W4A8_MOE_HITS += 1
            topk_weights, topk_ids = _select_experts_for_layer(
                select_experts, layer, x, router_logits
            )
            if rows == 2 and use_p4:
                output = torch.ops.triton_jit_cpu.q4_linear_g128_w4a8_kai_moe4_m2_p4(
                    x.contiguous(),
                    layer.w13_weight_packed,
                    layer.w13_out_features,
                    layer.w13_in_features,
                    layer.w2_weight_packed,
                    layer.w2_out_features,
                    topk_ids.contiguous(),
                    topk_weights.contiguous(),
                    False,
                )
            elif rows == 2 and hasattr(
                torch.ops.triton_jit_cpu,
                "q4_linear_g128_w4a8_kai_moe4_m2",
            ):
                output = (
                    torch.ops.triton_jit_cpu.q4_linear_g128_w4a8_kai_moe4_m2(
                        x.contiguous(),
                        layer.w13_weight_packed,
                        layer.w13_out_features,
                        layer.w13_in_features,
                        layer.w2_weight_packed,
                        layer.w2_out_features,
                        topk_ids.contiguous(),
                        topk_weights.contiguous(),
                        False,
                    )
                )
            elif rows == 1:
                decode_op = (
                    torch.ops.triton_jit_cpu.q4_linear_g128_w4a8_kai_moe4_packed_decode_p4
                    if use_p4
                    else torch.ops.triton_jit_cpu.q4_linear_g128_w4a8_kai_moe4_packed_decode
                )
                output = decode_op(
                    x.contiguous(),
                    layer.w13_weight_packed,
                    layer.w13_out_features,
                    layer.w13_in_features,
                    layer.w2_weight_packed,
                    layer.w2_out_features,
                    topk_ids.contiguous(),
                    topk_weights.contiguous(),
                    False,
                )
            else:
                prefill_op = (
                    torch.ops.triton_jit_cpu.q4_linear_g128_w4a8_kai_moe4_prefill_16x4
                    if use_p4
                    else torch.ops.triton_jit_cpu.q4_linear_g128_w4a8_kai_moe4_prefill
                )
                output = prefill_op(
                    x.contiguous(),
                    layer.w13_weight_packed,
                    layer.w13_out_features,
                    layer.w13_in_features,
                    layer.w2_weight_packed,
                    layer.w2_out_features,
                    topk_ids.contiguous(),
                    topk_weights.contiguous(),
                    False,
                )
            self._maybe_apply_shared_experts(
                shared_experts_input,
                SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
            )
            shared_output = (
                self._shared_experts.output
                if self._shared_experts is not None
                else None
            )
            return shared_output, output

        runner._apply_quant_method = MethodType(apply_w4a8, runner)
        if not keep_bf16:
            layer.w13_weight = torch.nn.Parameter(
                torch.empty(0, dtype=w13.dtype, device=w13.device),
                requires_grad=False,
            )
            layer.w2_weight = torch.nn.Parameter(
                torch.empty(0, dtype=w2.dtype, device=w2.device),
                requires_grad=False,
            )
        layer._vllm_fl_mtp_w4a8 = True
        layer._vllm_fl_mtp_w4a8_p4 = use_p4
        layer._vllm_fl_mtp_w4a8_keep_bf16 = keep_bf16
        installed += 1
    if installed == 0:
        raise RuntimeError(
            "TeleChat4 MTP W4A8 was requested but no compatible routed "
            f"expert layer was found; candidates={expert_candidates}"
        )
    return installed


def install_telechat4_mtp_bf16_moe(model: torch.nn.Module) -> int:
    """Use the exact-weight Arm BFDOT M=1 kernel for the BF16 MTP MoE."""
    active_env = "FL_CPU_TELECHAT4_MTP_BF16_MOE_NATIVE"
    if os.environ.get(active_env, "0").lower() not in {"1", "true", "on"}:
        return 0
    if os.environ.get(
        "FL_CPU_TELECHAT4_MTP_W4A8_MOE", "0"
    ).lower() in {"1", "true", "on"}:
        raise RuntimeError(
            "TeleChat4 MTP native BF16 and W4A8 MoE are mutually exclusive"
        )
    if not hasattr(torch.ops.triton_jit_cpu, "telechat4_mtp_bf16_moe4"):
        raise RuntimeError("TeleChat4 MTP native BF16 MoE operator is unavailable")

    from vllm.model_executor.layers.fused_moe.cpu_fused_moe import (
        select_experts,
    )
    from vllm_fl.patches.cpu_w4a8_moe import _select_experts_for_layer

    installed = 0
    candidates = []
    for module_name, layer in model.named_modules():
        if not module_name.endswith(".mtp_block.mlp.experts"):
            continue
        quant_method = getattr(layer, "quant_method", None)
        cpu_moe = getattr(quant_method, "cpu_fused_moe", None)
        candidates.append(
            (
                module_name,
                type(quant_method).__name__,
                type(cpu_moe).__name__,
                getattr(cpu_moe, "isa", None),
            )
        )
        if (
            getattr(layer, "_vllm_fl_mtp_bf16_native", False)
            # The native kernel consumes the row-major weights retained by
            # CPUFusedMOE when vLLM's optional prepack op is unavailable.
            or getattr(cpu_moe, "isa", None) != "none"
            or not hasattr(layer, "w13_weight")
            or not hasattr(layer, "w2_weight")
        ):
            continue
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        if (
            w13.dtype != torch.bfloat16
            or w2.dtype != torch.bfloat16
            or tuple(w13.shape) != (64, 2048, 3584)
            or tuple(w2.shape) != (64, 3584, 1024)
        ):
            raise RuntimeError(
                "unexpected row-major TeleChat4 MTP BF16 expert layout: "
                f"w13={tuple(w13.shape)}/{w13.dtype}, "
                f"w2={tuple(w2.shape)}/{w2.dtype}"
            )

        original_apply = layer.runner._apply_quant_method

        def apply_native_bf16(
            self,
            layer,
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids=None,
            *,
            _original_apply=original_apply,
        ):
            x = hidden_states
            global _BF16_MOE_M1_CALLS
            global _BF16_MOE_M2_CALLS
            global _BF16_MOE_OTHER_CALLS
            if x.ndim == 2 and x.shape[0] == 1:
                _BF16_MOE_M1_CALLS += 1
            elif x.ndim == 2 and x.shape[0] == 2:
                _BF16_MOE_M2_CALLS += 1
            else:
                _BF16_MOE_OTHER_CALLS += 1
            active = os.environ.get(active_env, "0").lower() in {
                "1",
                "true",
                "on",
            }
            if (
                not active
                or x.ndim != 2
                or x.shape[0] not in (1, 2)
                or layer.apply_router_weight_on_input
            ):
                return _original_apply(
                    layer=layer,
                    hidden_states=x,
                    router_logits=router_logits,
                    shared_experts_input=shared_experts_input,
                    input_ids=input_ids,
                )
            from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
                SharedExpertsOrder,
            )

            self._maybe_apply_shared_experts(
                shared_experts_input, SharedExpertsOrder.NO_OVERLAP
            )
            topk_weights, topk_ids = _select_experts_for_layer(
                select_experts, layer, x, router_logits
            )
            global _BF16_MOE_HITS
            _BF16_MOE_HITS += 1
            fused_output = torch.ops.triton_jit_cpu.telechat4_mtp_bf16_moe4(
                x.contiguous(),
                layer.w13_weight,
                layer.w2_weight,
                topk_ids.contiguous(),
                topk_weights.contiguous(),
            )
            self._maybe_apply_shared_experts(
                shared_experts_input,
                SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
            )
            shared_output = (
                self._shared_experts.output
                if self._shared_experts is not None
                else None
            )
            return shared_output, fused_output

        layer.runner._apply_quant_method = MethodType(
            apply_native_bf16, layer.runner
        )
        layer._vllm_fl_mtp_bf16_native = True
        installed += 1
    if installed == 0:
        raise RuntimeError(
            "TeleChat4 MTP native BF16 MoE requested but no compatible "
            f"CPUFusedMOE(none) layer was found; candidates={candidates}"
        )
    return installed


def install_telechat4_mtp_bf16_shared_mlp(model: torch.nn.Module) -> int:
    """Fuse the BF16 shared expert in the separately loaded MTP layer.

    This hook is draft-only.  The target's checkpoint-W4A8 shared expert is
    owned by ``install_telechat4_w4a8_shared_mlp_decode`` and never reaches
    this operator.  Keep the feature opt-in while its draft-token acceptance
    is being validated against the stock BF16 linear path.
    """
    active_env = "FL_CPU_TELECHAT4_MTP_BF16_SHARED_MLP_NATIVE"
    if os.environ.get(active_env, "0").lower() not in {"1", "true", "on"}:
        return 0
    if not hasattr(torch.ops.triton_jit_cpu, "telechat4_mtp_bf16_shared_mlp"):
        raise RuntimeError(
            "TeleChat4 native BF16 shared MLP operator is unavailable"
        )
    installed = 0
    candidates = []
    for module_name, module in model.named_modules():
        if not module_name.endswith(".mtp_block.mlp.shared_experts"):
            continue
        candidates.append(module_name)
        if getattr(module, "_vllm_fl_mtp_bf16_shared_mlp", False):
            continue
        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        gate_up_weight = getattr(gate_up, "weight", None)
        down_weight = getattr(down, "weight", None)
        if gate_up_weight is None or down_weight is None:
            continue
        if (
            gate_up_weight.dtype != torch.bfloat16
            or down_weight.dtype != torch.bfloat16
            or tuple(gate_up_weight.shape) != (2048, 3584)
            or tuple(down_weight.shape) != (3584, 1024)
            or not gate_up_weight.is_contiguous()
            or not down_weight.is_contiguous()
        ):
            continue

        original_forward = module.forward

        def forward_native(
            self,
            x,
            *,
            _original_forward=original_forward,
            _active_env=active_env,
        ):
            active = os.environ.get(_active_env, "0").lower() in {
                "1",
                "true",
                "on",
            }
            if (
                active
                and x.device.type == "cpu"
                and x.dtype == torch.bfloat16
                and x.dim() in (1, 2)
                and ((x.dim() == 1 and x.shape[0] == 3584) or
                     (x.dim() == 2 and x.shape[0] in (1, 2) and
                      x.shape[1] == 3584))
                and x.is_contiguous()
            ):
                global _BF16_SHARED_MLP_HITS
                _BF16_SHARED_MLP_HITS += 1
                rows = 1 if x.dim() == 1 else x.shape[0]
                result = torch.ops.triton_jit_cpu.telechat4_mtp_bf16_shared_mlp(
                    x.view(rows, 3584),
                    self.gate_up_proj.weight,
                    self.down_proj.weight,
                )
                return result.view_as(x)
            return _original_forward(x)

        module._vllm_fl_mtp_bf16_shared_mlp_original_forward = original_forward
        module.forward = MethodType(forward_native, module)
        module._vllm_fl_mtp_bf16_shared_mlp = True
        installed += 1
    if installed == 0:
        raise RuntimeError(
            "TeleChat4 native BF16 shared MLP requested but no compatible "
            f"MTP shared expert was found; candidates={candidates}"
        )
    return installed


def install_telechat4_mtp_draft_lmhead_q4(model: torch.nn.Module) -> bool:
    """Scope the optional Q4 copy of the shared head to draft logits only."""
    if os.environ.get("FLAGGEMS_MTP_DRAFT_LMHEAD_Q4", "0").lower() not in {
        "1",
        "true",
        "on",
    }:
        return False
    if getattr(model, "_vllm_fl_mtp_draft_lmhead_q4", False):
        return False

    from flag_gems.runtime.backend._arm.q4 import (
        mtp_draft_lmhead_q4_context,
    )
    from flag_gems.runtime.backend._arm.q4.linear import (
        mtp_draft_lmhead_q4_w8_refine_hits,
    )

    original_compute_logits = model.compute_logits

    def compute_logits_with_draft_q4(
        self,
        *args,
        _original_compute_logits=original_compute_logits,
        **kwargs,
    ):
        refine_before = mtp_draft_lmhead_q4_w8_refine_hits()
        with mtp_draft_lmhead_q4_context():
            logits = _original_compute_logits(*args, **kwargs)
        self._vllm_fl_mtp_last_draft_head_w8_refined = (
            mtp_draft_lmhead_q4_w8_refine_hits() > refine_before
        )
        return logits

    model.compute_logits = MethodType(compute_logits_with_draft_q4, model)
    model._vllm_fl_mtp_last_draft_head_w8_refined = False
    model._vllm_fl_mtp_draft_lmhead_q4 = True
    return True


def install_telechat4_mtp_config_override() -> bool:
    """Teach vLLM's speculative config loader about TeleChat4 MTP.

    ``SpeculativeConfig`` calls ``hf_config_override`` while constructing the
    draft ``ModelConfig``.  The upstream override recognizes DeepSeek model
    types but not TeleChat4, despite the checkpoint using the same MTP weight
    layout.  This process-local wrapper keeps the target architecture intact
    and changes only the draft config's model type/architecture.
    """

    from vllm.config.speculative import SpeculativeConfig

    current = SpeculativeConfig.hf_config_override
    if getattr(current, "_vllm_fl_telechat4_mtp", False):
        return False

    def override(hf_config):
        config = current(hf_config)
        target_model_type = getattr(config, "model_type", None)
        if (
            target_model_type in {"telechat4", "xingchen4", "xing4_0"}
            and int(getattr(config, "num_nextn_predict_layers", 0) or 0) > 0
        ):
            n_predict = int(config.num_nextn_predict_layers)
            # Use the upstream MTP model type so vLLM's existing speculative
            # config validation recognizes this as an MTP head.  The
            # architecture remains plugin-owned and selects TeleChat4MTP.
            config.model_type = "deepseek_mtp"
            config.update(
                {
                    "n_predict": n_predict,
                    "architectures": [
                        "Xing4_0MTPModel" if target_model_type == "xing4_0" else
                        "XingChen4MTPModel"
                        if target_model_type == "xingchen4"
                        else "TeleChat4MTPModel"
                    ],
                }
            )
        return config

    override._vllm_fl_telechat4_mtp = True
    SpeculativeConfig.hf_config_override = staticmethod(override)
    return True


def install_telechat4_mtp_safetensors_loader() -> bool:
    """Keep exact MTP shard overrides on the safetensors iterator.

    vLLM's ``DefaultModelLoader`` normally infers the iterator from the
    filename glob it selected.  An exact override such as
    ``model-00041-of-00041.safetensors`` is intentionally narrower than the
    usual ``*.safetensors`` pattern, but older vLLM releases leave
    ``use_safetensors`` false in that case and then dispatch to
    ``torch.load``.  The TeleChat4 snapshot is safetensors-only, so make this
    inference explicit in a process-local, idempotent shim.
    """

    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    current = DefaultModelLoader._prepare_weights
    if getattr(current, "_vllm_fl_telechat4_mtp", False):
        return False

    def prepare_with_exact_safetensors(
        self,
        model_name_or_path,
        subfolder,
        revision,
        fall_back_to_pt,
        allow_patterns_overrides,
    ):
        folder, files, use_safetensors = current(
            self,
            model_name_or_path,
            subfolder,
            revision,
            fall_back_to_pt,
            allow_patterns_overrides,
        )
        if (
            not use_safetensors
            and allow_patterns_overrides
            and files
            and all(str(path).endswith(".safetensors") for path in files)
        ):
            use_safetensors = True
        return folder, files, use_safetensors

    prepare_with_exact_safetensors._vllm_fl_telechat4_mtp = True
    DefaultModelLoader._prepare_weights = prepare_with_exact_safetensors
    return True


__all__ = [
    "install_telechat4_mtp_bf16_moe",
    "install_telechat4_mtp_bf16_shared_mlp",
    "telechat4_mtp_bf16_moe_hits",
    "telechat4_mtp_bf16_moe_row_calls",
    "telechat4_mtp_bf16_shared_mlp_hits",
    "TeleChat4MTP",
    "XingChen4MTP",
    "install_telechat4_mtp_draft_lmhead_q4",
    "install_telechat4_mtp_config_override",
    "install_telechat4_mtp_safetensors_loader",
    "install_telechat4_mtp_w4a8_moe",
    "install_telechat4_mtp_w8_moe",
    "telechat4_mtp_sequence_context",
    "telechat4_mtp_w4a8_moe_active",
    "telechat4_mtp_w4a8_moe_hits",
    "telechat4_mtp_w8_moe_hits",
]
