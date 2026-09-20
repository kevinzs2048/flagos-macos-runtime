#!/usr/bin/env python3
"""Same-process native/mixed 40-step images, matched to a saved request."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", default="mixed,native")
    parser.add_argument("--wavefront", action="store_true")
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument(
        "--wavefront-tile",
        type=int,
        nargs=3,
        default=[32, 128, 256],
        metavar=("M", "N", "DOWN_N"),
    )
    parser.add_argument("--drain-tail", action="store_true")
    args = parser.parse_args()
    order = args.order.split(",")
    if sorted(order) != ["mixed", "native"]:
        raise ValueError("Run one native and one mixed request")
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.baseline.read_text())
    config = baseline["configuration"]
    assert (
        config["steps"] == 40 and config["true_cfg_scale"] == 1 and config["kv_cache"]
    )
    torch.set_num_interop_threads(1)
    from qwen_image_cpu.pipeline import load_pipeline, generate
    from qwen_image_cpu.w8_swiglu import (
        configure_hybrid,
        select_hybrid,
        configure_wavefront,
        select_wavefront,
    )

    pipe, provider, metadata = load_pipeline(config["model"])
    policy = dict(
        workers=args.workers,
        cpu_clusters=torch.tensor([0] * 6 + [1] * 6 + [2] * 6),
        enabled=False,
    )
    if args.wavefront:
        policy.update(tile=tuple(args.wavefront_tile), drain_tail=args.drain_tail)
        configured = configure_wavefront(pipe.transformer, **policy)
        select_policy = select_wavefront
        counter = "_wavefront_swiglu_calls"
    else:
        configured = configure_hybrid(pipe.transformer, fraction=0.375, **policy)
        select_policy = select_hybrid
        counter = "_hybrid_swiglu_calls"
    assert configured == 28
    layers = [p for p in pipe.transformer.modules() if hasattr(p, counter)]
    kwargs = {k: config[k] for k in ("prompt", "width", "height", "seed")}
    report = dict(
        configuration=config,
        metadata=metadata,
        order=order,
        mixed_policy="wavefront" if args.wavefront else "split_n",
        mixed_workers=args.workers,
        wavefront_tile=args.wavefront_tile if args.wavefront else None,
        drain_tail=args.drain_tail if args.wavefront else False,
        scope="Same process, same weights/prompt/seed, 2-step warmup before each 40-step image; one sample per mode",
        expected_pixel_sha256=baseline["requests"][0]["pixel_sha256"],
        requests=[],
        status="running",
    )
    try:
        for mode in order:
            select_policy(pipe.transformer, mode == "mixed")
            print("WARMUP", mode, flush=True)
            generate(pipe, provider, steps=2, **kwargs)
            before = sum(getattr(p, counter) for p in layers)
            print("IMAGE", mode, flush=True)
            image, record = generate(pipe, provider, steps=40, **kwargs)
            image.save(args.output / (mode + ".png"))
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            assert digest == report["expected_pixel_sha256"]
            hits = sum(getattr(p, counter) for p in layers) - before
            assert hits == (1120 if mode == "mixed" else 0)
            record.update(
                mode=mode,
                pixel_sha256=digest,
                pixel_exact=True,
                hybrid_mlp_calls=hits,
                image=str(args.output / (mode + ".png")),
            )
            report["requests"].append(record)
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print("RESULT", json.dumps(record), flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
