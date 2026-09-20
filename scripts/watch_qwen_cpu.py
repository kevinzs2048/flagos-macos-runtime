#!/usr/bin/env python3
"""Read-only CPU telemetry tagged by completed model benchmark rows.

Only adjacent samples with the same completed_rows >= 1 belong to a fully
contained timed forward. The first phase also contains setup/warmup.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--macmon", required=True)
p.add_argument("--report", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--rows", type=int, default=16)
a = p.parse_args()
proc = subprocess.Popen(
    [a.macmon, "pipe", "-i", "1000"], stdout=subprocess.PIPE, text=True
)
try:
    with a.output.open("w") as stream:
        for line in proc.stdout:
            try:
                completed = len(json.loads(a.report.read_text())["rows"])
            except (OSError, ValueError, KeyError):
                completed = 0
            sample = {
                "read_at_epoch": time.time(),
                "completed_rows": completed,
                "metrics": json.loads(line),
            }
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            if completed >= a.rows:
                break
finally:
    proc.terminate()
    proc.wait(timeout=10)
