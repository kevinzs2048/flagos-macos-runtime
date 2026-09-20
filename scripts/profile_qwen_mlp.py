#!/usr/bin/env python3
"""Profile a prefix of the actual 40-step schedule and capture one real MLP input.

Intentionally stops before image decoding. No inference defaults are modified.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_interop_threads(1)
    from qwen_image_cpu.pipeline import generate, load_pipeline

    pipe, provider, metadata = load_pipeline(args.model)
    totals, calls = defaultdict(float), defaultdict(int)
    originals, steps = [], []
    capture = {}
    step = 0

    def instrument(namespace, name, label):
        original = getattr(namespace, name)

        def wrapped(*values, **kwargs):
            category = label(values) if callable(label) else label
            started = time.perf_counter()
            result = original(*values, **kwargs)
            totals[category] += time.perf_counter() - started
            calls[category] += 1
            return result

        originals.append((namespace, name, original))
        setattr(namespace, name, wrapped)

    instrument(torch.ops.qwen21_w8_swiglu, "run", "w8_fused_gate_up_swiglu_pack")
    instrument(torch.ops.qwen21_w8a8_sme, "linear_pair_bf16", "w8_paired_qk_with_a8")
    instrument(torch.ops.qwen21_w8a8_sme, "quant_pack_bf16", "standalone_a8_pack")
    instrument(
        torch.ops.qwen21_bf16_register_epilogue,
        "matmul",
        lambda values: "bf16_down_gemm" if values[4] == 12288 else "bf16_other_gemm",
    )
    layer = pipe.transformer.transformer_blocks[2].img_mlp

    def capture_input(module, values):
        if step == 1 and "input" not in capture:
            capture["input"] = values[0].detach().clone()

    handle = layer.register_forward_pre_hook(capture_input)
    previous = time.perf_counter()

    class PrefixComplete(Exception):
        pass

    def progress(done, total):
        nonlocal previous, step
        now = time.perf_counter()
        steps.append(
            {
                "step": done,
                "interval_seconds": now - previous,
                "operators_seconds": dict(totals),
                "calls": dict(calls),
            }
        )
        print(json.dumps(steps[-1]), flush=True)
        totals.clear()
        calls.clear()
        previous = now
        step = done
        if done == 3:
            raise PrefixComplete

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
    except PrefixComplete:
        pass
    finally:
        handle.remove()
        for namespace, name, original in originals:
            setattr(namespace, name, original)
    assert len(steps) == 3 and "input" in capture
    capture.update(
        gate_codes=layer.gate_layer.weight_codes,
        gate_scale=layer.gate_layer.weight_scale,
        up_codes=layer.proj.weight_codes,
        up_scale=layer.proj.weight_scale,
        down_weight=layer.out.weight,
    )
    torch.save(capture, args.output / "block2-step2-mlp.pt")
    metadata.update(
        scope="First three steps of a 40-step request; intentionally no image; "
        "first interval includes text encoding; instrumented timings",
        steps=steps,
        dispatch=provider.snapshot(),
    )
    (args.output / "profile.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
