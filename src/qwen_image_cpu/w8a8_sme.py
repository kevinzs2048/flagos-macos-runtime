"""Qwen-Image checkpoint policy and loading; math belongs to FlagGems."""

import json
import time
from pathlib import Path

import torch

from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import (
    DynamicW8SMELinear,
    quantize_weight,
)


def selected_layer(name, policy):
    if not name.startswith("transformer_blocks."):
        return False
    block = int(name.split(".")[1])
    if policy == "same112":
        return 2 <= block < 30 and name.endswith(
            (".attn.to_q", ".attn.to_k", ".img_mlp.proj", ".img_mlp.gate_layer")
        )
    if policy == "middle196":
        return 2 <= block < 30
    if policy == "all224":
        return True
    raise ValueError("Unknown W8 policy")


@torch.no_grad()
def enable_transformer_w8a8(model, policy="same112"):
    started = time.perf_counter()
    records = []
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, torch.nn.Linear) or not selected_layer(name, policy):
            continue
        codes, scales = quantize_weight(layer.weight)
        parent, _, key = name.rpartition(".")
        setattr(
            model.get_submodule(parent),
            key,
            DynamicW8SMELinear(codes, scales, layer.bias),
        )
        records.append({"name": name, "n": layer.out_features, "k": layer.in_features})
    if len(records) != {"same112": 112, "middle196": 196, "all224": 224}[policy]:
        raise ValueError("Unexpected Transformer W8 Linear count")
    return {
        "policy": policy,
        "linear_count": len(records),
        "seconds": time.perf_counter() - started,
        "weight": "symmetric INT8 [-127,127], per-output-channel, FP32 scale",
        "activation": "dynamic asymmetric INT8 per-token; KAI rounding; FP32 scale",
        "backend": "KleidiAI SME2 INT8 MOPA, INT32 accumulation, FP32 scaling, BF16 output",
        "source": "original BF16; no W4 requantization",
        "disk_export": False,
        "layers": records,
    }


def load_transformer_w8(directory, model_class):
    from accelerate import init_empty_weights
    from safetensors import safe_open

    directory = Path(directory)
    if (
        not (directory / "EXPORT_COMPLETE").is_file()
        or (directory / "EXPORT_FAILED").exists()
    ):
        raise ValueError("Incomplete W8 export")
    root = directory / "transformer"
    quant = json.loads((root / "quantization_config.json").read_text())
    required = {
        "format_version": 1,
        "scheme": "w8a8_dynamic_per_channel",
        "weight_dtype": "int8",
        "weight_mapping": "symmetric",
        "weight_qmin": -127,
        "weight_qmax": 127,
        "weight_zero_point": 0,
        "weight_granularity": "per_output_channel",
        "weight_scale_dtype": "float32",
        "weight_rounding": "nearest_even",
        "activation_dtype": "int8",
        "activation_mapping": "asymmetric",
        "activation_granularity": "per_token",
        "activation_scale_dtype": "float32",
        "activation_code_rounding": "nearest_ties_away_from_zero",
        "activation_zero_point_rounding": "nearest_even",
        "activation_quantizer": "KleidiAI 1.29.0 kai_lhs_quant_pack_qai8dxp_f32",
        "output_dtype": "bfloat16",
        "accumulator_dtype": "int32",
        "dequant_dtype": "float32",
    }
    if any(quant.get(key) != value for key, value in required.items()):
        raise ValueError("Unsupported W8 quantization metadata")
    with init_empty_weights():
        model = model_class.from_config(json.loads((root / "config.json").read_text()))
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    for name, layer in quant["quantized_layers"].items():
        original = model.get_submodule(name)
        n, k = layer["n"], layer["k"]
        if not isinstance(original, torch.nn.Linear) or (
            original.out_features,
            original.in_features,
        ) != (n, k):
            raise ValueError("W8 model shape mismatch: " + name)
        if (
            n <= 0
            or k <= 0
            or k % 8
            or k > 32768
            or layer["weight_codes"] != name + ".weight_codes"
            or layer["weight_scale"] != name + ".weight_scale"
        ):
            raise ValueError("Invalid W8 layer metadata")
        replacement = DynamicW8SMELinear(
            torch.empty(n, k, dtype=torch.int8, device="meta"),
            torch.empty(n, dtype=torch.float32, device="meta"),
            original.bias,
        )
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent), attribute, replacement)
    index = json.loads(
        (root / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    expected = model.state_dict()
    seen = set()
    if set(index) != set(expected):
        raise ValueError("W8 checkpoint state keys differ from model")
    for filename in sorted(set(index.values())):
        if Path(filename).name != filename:
            raise ValueError("Invalid shard path")
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        for key, tensor in tensors.items():
            if key in seen or key not in expected or index.get(key) != filename:
                raise ValueError("Duplicate/misindexed W8 tensor")
            if (
                tensor.shape != expected[key].shape
                or tensor.dtype != expected[key].dtype
            ):
                raise ValueError("W8 tensor shape/dtype mismatch: " + key)
            if key.endswith(".weight_scale") and (
                not tensor.isfinite().all() or not (tensor > 0).all()
            ):
                raise ValueError("Invalid W8 scale")
            if key.endswith(".weight_codes") and (tensor == -128).any():
                raise ValueError("W8 code outside declared [-127,127]")
        model.load_state_dict(tensors, strict=False, assign=True)
        seen.update(tensors)
        del tensors
    if seen != set(expected) or any(
        t.device.type == "meta" for t in model.state_dict().values()
    ):
        raise ValueError("Unloaded W8 tensors")
    report = {
        "policy": quant["policy"],
        "linear_count": len(quant["quantized_layers"]),
        "checkpoint": str(directory.resolve()),
        "disk_export": True,
        "format": required,
        "backend": "KleidiAI SME2 INT8 MOPA",
    }
    return model.eval().requires_grad_(False), report
