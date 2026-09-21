#!/usr/bin/env python3
"""Summarize samples contained within timed model forwards; no causal attribution."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.report.read_text())["rows"]
    samples = [json.loads(line) for line in args.telemetry.read_text().splitlines()]
    groups = defaultdict(list)
    for previous, current in zip(samples, samples[1:]):
        completed = current["completed_rows"]
        dt = current["read_at_epoch"] - previous["read_at_epoch"]
        if (
            completed != previous["completed_rows"]
            or not 1 <= completed < len(rows)
            or not 0 < dt <= 2.5
        ):
            continue
        row = rows[completed]
        groups[
            (
                row["group"],
                row["mixed"],
                row.get("neon_workers") or 0,
                row.get("idle_us") or 0,
            )
        ].append((dt, current["metrics"]))
    result = []
    for (group, mixed, neon_workers, idle_us), contained in sorted(groups.items()):
        seconds = sum(dt for dt, _ in contained)
        metrics = {
            key: sum(dt * m[key] for dt, m in contained) / seconds
            for key in (
                "cpu_power",
                "ecpu_freq_mhz",
                "pcpu_freq_mhz",
                "cpu_active_ratio",
            )
        }
        metrics["cpu_temp_avg"] = (
            sum(dt * m["temp"]["cpu_temp_avg"] for dt, m in contained) / seconds
        )
        result.append(
            dict(
                group=group,
                mixed=mixed,
                neon_workers=neon_workers if mixed else None,
                idle_us=idle_us if mixed else None,
                samples=len(contained),
                sampled_seconds=seconds,
                means=metrics,
            )
        )
    args.output.write_text(
        json.dumps(
            dict(
                scope="Time-weighted raw macmon fields; adjacent samples inside one timed forward only; not proof of cause or SME clock frequency",
                rows=result,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
