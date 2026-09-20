#!/usr/bin/env python3
"""Compare the production fused MLP with mixed SME2/NEON on captured model data."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(18)
    torch.set_num_interop_threads(1)
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops
    from flag_gems.runtime.backend._arm.quantized_linear.sme2.pointwise import (
        silu_table,
    )

    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    x = data["input"].reshape(-1, data["input"].shape[-1])
    m, k = x.shape
    n = data["gate_codes"].shape[0]
    w8, fused = load_ops("w8a8_sme"), load_ops("w8_swiglu")
    bf16, down = load_ops("bf16_sme"), load_ops("bf16_register_epilogue")
    table = silu_table()
    clusters = torch.tensor([0] * 6 + [1] * 6 + [2] * 6, dtype=torch.long)
    weights = [(data[p + "_codes"], data[p + "_scale"]) for p in ("gate", "up")]
    full = [w8.pack(*w) for w in weights]
    down_rhs = bf16.pack_direct(data["down_weight"])
    splits = {}
    for fraction in (0.375, 0.5):
        split = int(n * fraction) // 64 * 64
        splits[fraction] = (
            split,
            [w8.pack(q[:split], s[:split]) for q, s in weights],
            [w8.pack_neon(q[split:], s[split:]) for q, s in weights],
        )

    def run(policy):
        start = time.perf_counter()
        lhs = w8.quant_pack_bf16(x)
        if policy is not None:
            nl = w8.repack_neon_lhs(lhs, m, k)
        packed_at = time.perf_counter()
        if policy is None:
            output = fused.run(lhs, *full, table, m, n, k, 12, 64, 256)
        else:
            workers, fraction, nmt, nnt = policy
            split, sme, neon = splits[fraction]
            output = fused.run_hybrid(
                lhs,
                nl,
                *sme,
                *neon,
                table,
                m,
                split,
                n - split,
                k,
                workers,
                clusters,
                64,
                256,
                nmt,
                nnt
            )
        fused_at = time.perf_counter()
        result = down.matmul(
            output, down_rhs, m, data["down_weight"].shape[0], n, 12, 32, 256
        )
        done = time.perf_counter()
        return (
            output,
            result,
            {
                "pack": packed_at - start,
                "fused_w8": fused_at - packed_at,
                "bf16_down": done - fused_at,
                "total": done - start,
            },
        )

    gold_packed, gold, _ = run(None)
    for _ in range(32):
        run(None)
    results = []
    for workers in (12, 18):
        for fraction in (0.375, 0.5):
            for tile in ((32, 64), (128, 128)):
                policy = (workers, fraction, *tile)
                for _ in range(2):
                    packed, y, _ = run(policy)
                    assert torch.equal(packed, gold_packed)
                    assert torch.equal(y, gold)
                rows = []
                for group in range(args.groups):
                    order = (
                        [False, True, True, False]
                        if group % 2 == 0
                        else [True, False, False, True]
                    )
                    for mixed in order:
                        packed, y, times = run(policy if mixed else None)
                        assert torch.equal(y, gold)
                        rows.append(dict(group=group, mixed=mixed, **times))
                means = {
                    mode: {
                        field: statistics.mean(
                            r[field] for r in rows if r["mixed"] == mixed
                        )
                        for field in ("pack", "fused_w8", "bf16_down", "total")
                    }
                    for mode, mixed in (("pure", False), ("mixed", True))
                }
                record = {
                    "policy": policy,
                    "means": means,
                    "rows": rows,
                    "packed_bytes_exact": True,
                    "bf16_output_exact": True,
                    "latency_change_percent": 100
                    * (means["mixed"]["total"] / means["pure"]["total"] - 1),
                }
                results.append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "capture": str(args.capture),
                            "shape_mnk": [m, n, k],
                            "scope": "Complete captured MLP: A8 + fused gate/up/SwiGLU + BF16 down; production 12-worker baseline",
                            "results": results,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    json.dumps({k: v for k, v in record.items() if k != "rows"}),
                    flush=True,
                )


if __name__ == "__main__":
    main()
