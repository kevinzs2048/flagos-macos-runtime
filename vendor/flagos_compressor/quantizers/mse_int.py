from __future__ import annotations

import torch


def mse_int_quantize(
    weight: torch.Tensor,
    *,
    num_bits: int,
    group_size: int,
    n_candidates: int = 200,
    chunk_size: int = 4096,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D float weight to signed integers and group scales."""
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight tensor, got {weight.dim()}D")
    if num_bits not in (4, 8):
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")
    out_features, in_features = weight.shape
    if group_size <= 0 or in_features % group_size:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )
    if n_candidates <= 0 or chunk_size <= 0:
        raise ValueError("n_candidates and chunk_size must be positive")

    q_min = -(1 << (num_bits - 1))
    q_max = (1 << (num_bits - 1)) - 1
    num_groups = in_features // group_size
    groups = weight.to(torch.float32).reshape(-1, group_size)
    if not torch.isfinite(groups).all().item():
        raise ValueError("Cannot quantize weights containing NaN or infinity")
    total_groups = groups.shape[0]
    multipliers = torch.linspace(
        0.5, 1.5, n_candidates, device=groups.device, dtype=torch.float32
    )
    best_scales = torch.empty(total_groups, device=groups.device, dtype=torch.float32)
    quantized = torch.empty_like(groups, dtype=torch.int8)

    # ``raw`` below has shape [groups, candidates, group_size]. Cap that
    # temporary at 32M float elements so channelwise quantization (where one
    # group spans the whole input row) cannot scale peak memory with a large
    # user-provided chunk size.
    max_candidate_elements = 32 * 1024 * 1024
    memory_bounded_chunk = max(
        1,
        max_candidate_elements // (n_candidates * group_size),
    )
    effective_chunk_size = min(chunk_size, memory_bounded_chunk)

    for start in range(0, total_groups, effective_chunk_size):
        end = min(start + effective_chunk_size, total_groups)
        chunk = groups[start:end]
        # Keep the existing symmetric convention: the negative endpoint is
        # exactly representable and the positive endpoint saturates at q_max.
        base_scale = (chunk.abs().amax(dim=1) / float(-q_min)).clamp(min=1e-10)
        candidates = base_scale.unsqueeze(1) * multipliers.unsqueeze(0)
        raw = torch.round(chunk.unsqueeze(1) / candidates.unsqueeze(2)).clamp(
            q_min, q_max
        )
        mse = ((chunk.unsqueeze(1) - raw * candidates.unsqueeze(2)) ** 2).mean(
            dim=2
        )
        best_idx = mse.argmin(dim=1)
        rows = end - start
        selected = candidates[
            torch.arange(rows, device=groups.device),
            best_idx,
        ]
        best_scales[start:end] = selected
        quantized[start:end] = (
            torch.round(chunk / selected.unsqueeze(1))
            .clamp(q_min, q_max)
            .to(torch.int8)
        )

    return (
        quantized.reshape(out_features, in_features),
        best_scales.reshape(out_features, num_groups).to(scale_dtype),
    )
