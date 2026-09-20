#!/usr/bin/env python3
"""Explore a single-team NEON-W8 / SME2-BF16 row pipeline on real model data."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(18)
    torch.set_num_interop_threads(1)
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops
    from flag_gems.runtime.backend._arm.quantized_linear.sme2.pointwise import (
        silu_table,
    )

    data = torch.load(a.capture, map_location="cpu", weights_only=True)
    x = data["input"].reshape(-1, data["input"].shape[-1])
    m, k = x.shape
    n = data["gate_codes"].shape[0]
    d = data["down_weight"].shape[0]
    w8, fused, bf16, down = [
        load_ops(s)
        for s in ["w8a8_sme", "w8_swiglu", "bf16_sme", "bf16_register_epilogue"]
    ]
    weights = [(data[s + "_codes"], data[s + "_scale"]) for s in ["gate", "up"]]
    full = [w8.pack(*w) for w in weights]
    neon = [w8.pack_neon(*w) for w in weights]
    rhs = bf16.pack_direct(data["down_weight"])
    table = silu_table()
    clusters = torch.tensor([0] * 6 + [1] * 6 + [2] * 6)

    def run(policy):
        lhs = w8.quant_pack_bf16(x)
        if policy is None:
            packed = fused.run(lhs, *full, table, m, n, k, 12, 64, 256)
            return down.matmul(packed, rhs, m, d, n, 12, 32, 256)
        workers, mt, nt, dt = policy
        nl = w8.repack_neon_lhs(lhs, m, k)
        return fused.run_wavefront(
            nl, *neon, rhs, table, m, n, k, d, workers, clusters, mt, nt, dt
        )

    gold = run(None)
    for _ in range(16):
        run(None)
    results = []
    for workers in [12, 18]:
        for mt in [32, 64, 128]:
            policy = (workers, mt, 128, 256)
            for _ in range(2):
                assert torch.equal(run(policy), gold)
            rows = []
            for g in range(a.groups):
                for mixed in (
                    [False, True, True, False]
                    if g % 2 == 0
                    else [True, False, False, True]
                ):
                    t = time.perf_counter()
                    y = run(policy if mixed else None)
                    seconds = time.perf_counter() - t
                    assert torch.equal(y, gold)
                    rows.append(dict(group=g, mixed=mixed, seconds=seconds))
            means = {
                str(mode): statistics.mean(
                    r["seconds"] for r in rows if r["mixed"] == mode
                )
                for mode in [False, True]
            }
            result = dict(
                policy=policy,
                means=means,
                rows=rows,
                bf16_output_exact=True,
                latency_change_percent=100 * (means["True"] / means["False"] - 1),
            )
            results.append(result)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(
                json.dumps(
                    dict(
                        capture=str(a.capture),
                        shape=[m, n, k, d],
                        scope="Full MLP including A8 and BF16 output; 12-worker production baseline; wavefront NEON W8 overlaps SME2 BF16",
                        results=results,
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(
                json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True
            )


if __name__ == "__main__":
    main()
