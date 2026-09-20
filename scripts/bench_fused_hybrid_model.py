#!/usr/bin/env python3
"""Teacher-forced full Transformer comparison of the opt-in fused hybrid MLP."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--fraction", type=float, default=0.375)
    parser.add_argument("--wavefront", action="store_true")
    parser.add_argument(
        "--mlp-stride",
        type=int,
        default=1,
        help="Enable one W8 MLP per N blocks to limit mixed-compute duty",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_interop_threads(1)
    from qwen_image_cpu.pipeline import generate, load_pipeline
    from qwen_image_cpu.w8_swiglu import (
        configure_hybrid,
        select_hybrid,
        configure_wavefront,
        select_wavefront,
    )

    pipe, provider, metadata = load_pipeline(args.model)
    model = pipe.transformer
    original_forward = model.forward
    captured = {}
    count = 0

    def capture(*a, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            captured["kwargs"] = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
        out = original_forward(*a, **kwargs)
        if count == 2:
            captured["prediction"] = out[0].clone()
        return out

    class Captured(Exception):
        pass

    def progress(step, total):
        if step == 2:
            raise Captured

    model.forward = capture
    try:
        generate(
            pipe,
            provider,
            prompt="A capybara reading a book by candlelight, watercolor painting",
            width=1024,
            height=1024,
            steps=40,
            seed=42,
            progress=progress,
        )
    except Captured:
        pass
    finally:
        model.forward = original_forward
    assert captured["kwargs"]["kv_cache_mode"] == "cached"
    kwargs = dict(
        workers=args.workers,
        cpu_clusters=torch.tensor([0] * 6 + [1] * 6 + [2] * 6),
        enabled=False,
    )
    if args.wavefront:
        if args.mlp_stride < 1:
            raise ValueError("MLP stride must be positive")
        targets = [
            b.img_mlp
            for b in model.transformer_blocks
            if hasattr(b.img_mlp, "_w8_swiglu_enabled")
        ][:: args.mlp_stride]
        configured = sum(configure_wavefront(layer, **kwargs) for layer in targets)
        select_policy = select_wavefront
        counter = "_wavefront_swiglu_calls"
    else:
        configured = configure_hybrid(model, fraction=args.fraction, **kwargs)
        select_policy = select_hybrid
        counter = "_hybrid_swiglu_calls"
    assert configured == (
        (28 + args.mlp_stride - 1) // args.mlp_stride if args.wavefront else 28
    )
    layers = [p for p in model.modules() if hasattr(p, counter)]
    for _ in range(4):
        with model.cache_context("cond"):
            assert torch.equal(model(**captured["kwargs"])[0], captured["prediction"])
    totals = defaultdict(float)
    originals = []

    def instrument(namespace, name, label):
        original = getattr(namespace, name)

        def wrapper(*a, **kw):
            category = label(a) if callable(label) else label
            start = time.perf_counter()
            result = original(*a, **kw)
            totals[category] += time.perf_counter() - start
            return result

        originals.append((namespace, name, original))
        setattr(namespace, name, wrapper)

    instrument(torch.ops.qwen21_w8_swiglu, "run", "fused_w8_mlp")
    instrument(torch.ops.qwen21_w8_swiglu, "run_hybrid", "fused_w8_mlp")
    if args.wavefront:
        instrument(torch.ops.qwen21_w8_swiglu, "run_wavefront", "wavefront_full_mlp")
    instrument(torch.ops.qwen21_w8a8_sme, "linear_pair_bf16", "w8_qk")
    instrument(
        torch.ops.qwen21_bf16_register_epilogue,
        "matmul",
        lambda a: "bf16_down" if a[4] == 12288 else "bf16_other",
    )
    rows = []
    try:
        for group in range(args.groups):
            for mixed in (
                [False, True, True, False]
                if group % 2 == 0
                else [True, False, False, True]
            ):
                select_policy(model, mixed)
                totals.clear()
                provider.reset()
                before = sum(getattr(p, counter) for p in layers)
                start = time.perf_counter()
                with model.cache_context("cond"):
                    prediction = model(**captured["kwargs"])[0]
                elapsed = time.perf_counter() - start
                assert torch.equal(prediction, captured["prediction"])
                hybrid_calls = sum(getattr(p, counter) for p in layers) - before
                assert hybrid_calls == (configured if mixed else 0)
                assert provider.snapshot()["calls"] == {"native_w8_gemm": 112}
                row = dict(
                    group=group,
                    mixed=mixed,
                    seconds=elapsed,
                    operators_seconds=dict(totals),
                    hybrid_mlp_calls=hybrid_calls,
                    prediction_exact=True,
                )
                rows.append(row)
                args.output.write_text(
                    json.dumps(
                        {
                            "metadata": metadata,
                            "scope": "1024x1024 full Transformer; fixed step2 of actual40-step schedule; prefix cache reused read-only; 4 warmup forwards",
                            "policy": {
                                "mode": "wavefront" if args.wavefront else "split_n",
                                "workers": args.workers,
                                "active_mlps": configured,
                                "mlp_stride": args.mlp_stride,
                                "fraction": None if args.wavefront else args.fraction,
                                "tile": [32, 128, 256] if args.wavefront else [32, 64],
                            },
                            "rows": rows,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(json.dumps(row), flush=True)
    finally:
        for namespace, name, original in originals:
            setattr(namespace, name, original)
    means = {
        str(mixed): statistics.mean(r["seconds"] for r in rows if r["mixed"] == mixed)
        for mixed in (False, True)
    }
    print("MEANS", means, flush=True)


if __name__ == "__main__":
    main()
