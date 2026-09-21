#!/usr/bin/env python3
"""Condition on native/mixed MLP work, then time identical independent GEMMs."""

import argparse
from collections import Counter
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
    parser.add_argument("--condition-seconds", type=float, default=3)
    parser.add_argument(
        "--condition-iterations",
        type=int,
        default=0,
        help="Use matched MLP call counts instead of matched requested duration",
    )
    parser.add_argument("--rest-ms", type=float, nargs="+", default=[0, 20])
    parser.add_argument("--trace-probe", action="store_true")
    parser.add_argument("--compare-backends", action="store_true")
    args = parser.parse_args()
    if args.trace_probe and args.compare_backends:
        parser.error("Tracing and backend comparison are separate experiments")
    if (
        args.groups < 1
        or args.condition_seconds <= 0
        or min(args.rest_ms) < 0
        or args.condition_iterations < 0
    ):
        parser.error("positive groups/duration and nonnegative rest required")
    torch.set_num_threads(18)
    torch.set_num_interop_threads(1)
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops
    from flag_gems.runtime.backend._arm.quantized_linear.sme2.pointwise import (
        silu_table,
    )

    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    x = data["input"].reshape(-1, data["input"].shape[-1])
    m, k = x.shape
    n, d = data["gate_codes"].shape[0], data["down_weight"].shape[0]
    w8, fused, bf16, gemm = [
        load_ops(s)
        for s in ("w8a8_sme", "w8_swiglu", "bf16_sme", "bf16_register_epilogue")
    ]
    weights = [(data[s + "_codes"], data[s + "_scale"]) for s in ("gate", "up")]
    full, neon = [w8.pack(*w) for w in weights], [w8.pack_neon(*w) for w in weights]
    down_rhs, table = bf16.pack_direct(data["down_weight"]), silu_table()
    clusters = torch.tensor([0] * 6 + [1] * 6 + [2] * 6)
    # This is a fixed proxy for a following square BF16 projection. It does not
    # consume either MLP output. Both paths use the same packed tensor objects.
    probe_lhs = bf16.pack_lhs_neon(x)
    probe_rhs = bf16.pack_direct(data["down_weight"][:, :k].contiguous())
    tracer = (
        load_ops("bf16_variants") if args.trace_probe or args.compare_backends else None
    )

    def mlp(mixed):
        lhs = w8.quant_pack_bf16(x)
        if mixed:
            return fused.run_wavefront(
                w8.repack_neon_lhs(lhs, m, k),
                *neon,
                down_rhs,
                table,
                m,
                n,
                k,
                d,
                18,
                clusters,
                32,
                64,
                64,
                False,
                15,
                5,
            )
        packed = fused.run(lhs, *full, table, m, n, k, 12, 64, 256)
        return gemm.matmul(packed, down_rhs, m, d, n, 12, 32, 256)

    def probe(backend="register"):
        if args.trace_probe:
            return tracer.matmul_trace(probe_lhs, probe_rhs, m, d, k, 18, 64, 256, 2, 0)
        if backend == "standard":
            return (
                tracer.matmul(
                    probe_lhs, probe_rhs, m, d, k, 18, 64, 256, 2, 0
                ).bfloat16(),
                None,
            )
        return gemm.matmul(probe_lhs, probe_rhs, m, d, k, 18, 64, 256), None

    expected_mlp = mlp(False)
    expected_probe = gemm.matmul(probe_lhs, probe_rhs, m, d, k, 18, 64, 256)
    assert torch.equal(mlp(True), expected_mlp)
    for _ in range(16):
        mlp(False)
        probe()
        if args.compare_backends:
            assert torch.equal(probe("standard")[0], expected_probe)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        scope="Diagnostic conditioning, not end-to-end performance; fixed prepacked independent BF16 probe inputs/weights; output allocation remains inside GEMM",
        condition_scope=(
            "Same MLP iteration count; elapsed conditioning duration may differ"
            if args.condition_iterations
            else "Same requested wall duration, possibly different MLP iteration counts; actual duration includes last completed iteration"
        ),
        probe_shape=[m, d, k],
        probe_weight="First K columns of captured BF16 down weight, not an actual Q/V projection checkpoint",
        condition_seconds=args.condition_seconds,
        condition_iterations=args.condition_iterations,
        trace_probe=args.trace_probe,
        compare_backends=args.compare_backends,
        standard_probe_scope="Untraced KAI FP32-output GEMM plus BF16 conversion, all included in probe latency",
        trace_scope=(
            "Diagnostic FP32-output KAI kernel with same packing/tile order; differs from production register-BF16 epilogue; raw mach ticks, not nanoseconds"
            if args.trace_probe
            else None
        ),
        mixed_policy=dict(workers=18, neon_workers=15, idle_us=5, tile=[32, 64, 64]),
        rows=rows,
    )
    for rest_ms in args.rest_ms:
        for group in range(args.groups):
            if args.compare_backends:
                modes = [
                    (False, "register"),
                    (True, "register"),
                    (True, "standard"),
                    (False, "standard"),
                ]
                offset = group % len(modes)
                modes = modes[offset:] + modes[:offset]
                order = modes + modes[::-1]
            else:
                order = [
                    (mixed, "register")
                    for mixed in (
                        [False, True, True, False]
                        if group % 2 == 0
                        else [True, False, False, True]
                    )
                ]
            for mixed, backend in order:
                started = time.perf_counter()
                iterations = 0
                while (
                    iterations < args.condition_iterations
                    if args.condition_iterations
                    else time.perf_counter() - started < args.condition_seconds
                ):
                    output = mlp(mixed)
                    iterations += 1
                conditioned = time.perf_counter() - started
                if rest_ms:
                    time.sleep(rest_ms / 1000)
                timings, outputs, traces = [], [], []
                for _ in range(2):
                    started = time.perf_counter()
                    result, trace = probe(backend)
                    timings.append(time.perf_counter() - started)
                    outputs.append(result)
                    traces.append(trace)
                assert torch.equal(output, expected_mlp)
                assert all(
                    torch.equal(result.bfloat16(), expected_probe) for result in outputs
                )
                row = dict(
                    group=group,
                    mixed=mixed,
                    probe_backend=backend,
                    rest_ms=rest_ms,
                    condition_iterations=iterations,
                    condition_seconds=conditioned,
                    probe_seconds=timings,
                    output_exact=True,
                    torch_threads=torch.get_num_threads(),
                )
                if args.trace_probe:
                    summaries = []
                    for trace in traces:
                        records = trace.tolist()
                        assert len(records) == ((m + 63) // 64) * ((d + 255) // 256)
                        spans = [r[7] - r[6] for r in records]
                        per_cpu = {}
                        for cpu in sorted(set(r[4] for r in records)):
                            stable = [
                                r[7] - r[6]
                                for r in records
                                if r[4] == cpu and r[5] == cpu
                            ]
                            per_cpu[str(cpu)] = dict(
                                tiles=len(stable),
                                median_ticks=(
                                    statistics.median(stable) if stable else None
                                ),
                            )
                        summaries.append(
                            dict(
                                tiles=len(records),
                                migrations=sum(r[4] != r[5] for r in records),
                                cpu_tiles=dict(Counter(r[4] for r in records)),
                                per_cpu=per_cpu,
                                median_kernel_ticks=statistics.median(spans),
                                task_span_ticks=max(r[7] for r in records)
                                - min(r[6] for r in records),
                                sum_kernel_ticks=sum(spans),
                            )
                        )
                    row["trace"] = summaries
                rows.append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
