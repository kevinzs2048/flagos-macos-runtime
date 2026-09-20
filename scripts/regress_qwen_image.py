#!/usr/bin/env python3
"""Replay saved 512/1024 image requests and verify migration pixel parity."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image


def revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    summary = {
        "runtime_revision": revision(root),
        "source_lock": json.loads((root / "qwen-image.sources.lock.json").read_text()),
        "baseline": str(args.baseline.resolve()),
        "status": "running",
        "requests": [],
    }
    try:
        for size in (512, 1024):
            name = f"{size}-40"
            baseline = json.loads((args.baseline / (name + ".json")).read_text())
            config = baseline["configuration"]
            assert (config["width"], config["height"], config["steps"]) == (
                size,
                size,
                40,
            )
            assert config["true_cfg_scale"] == 1 and config["kv_cache"] is True
            assert Path(config["model"]).resolve() == args.model.resolve()
            image = args.output / (name + ".png")
            command = [
                "bash",
                str(root / "bin/qwen-image"),
                "--model",
                str(args.model.resolve()),
                "--prompt",
                config["prompt"],
                "--width",
                str(size),
                "--height",
                str(size),
                "--steps",
                "40",
                "--seed",
                str(config["seed"]),
                "--warmup-steps",
                "2",
                "--output",
                str(image.resolve()),
            ]
            print("Running " + name, flush=True)
            with (args.output / (name + ".log")).open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            result = json.loads(image.with_suffix(".json").read_text())
            request = result["requests"][0]
            previous = baseline["requests"][0]
            for key in (
                "prompt",
                "width",
                "height",
                "steps",
                "seed",
                "true_cfg_scale",
                "kv_cache",
                "threads",
                "mlp_workers",
                "backend",
            ):
                assert result["configuration"][key] == config[key], key
            with Image.open(image) as im:
                assert im.size == (size, size)
                pixel_hash = hashlib.sha256(im.tobytes()).hexdigest()
            assert pixel_hash == request["pixel_sha256"] == previous["pixel_sha256"]
            assert (
                hashlib.sha256(image.read_bytes()).hexdigest() == request["png_sha256"]
            )
            dispatch = request["dispatch"]
            assert dispatch["distinct_w8_modules"] == 112
            assert dispatch["calls"] == {"native_w8_gemm": 4480}
            assert set(dispatch["modules"].values()) == {40}
            record = {
                "resolution": size,
                "steps": 40,
                "resident_seconds": request["resident_seconds"],
                "baseline_seconds": previous["resident_seconds"],
                "latency_change_percent": 100
                * (request["resident_seconds"] / previous["resident_seconds"] - 1),
                "pixel_sha256": pixel_hash,
                "pixel_parity": True,
                "distinct_w8_modules": 112,
                "native_w8_gemm_calls": 4480,
                "result": str(image.with_suffix(".json").resolve()),
            }
            summary["requests"].append(record)
            (args.output / "report.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)
        summary["status"] = "complete"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = repr(exc)
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
