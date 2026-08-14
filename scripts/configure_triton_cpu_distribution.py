#!/usr/bin/env python3
"""Make copied Triton distribution metadata describe the CPU-only Runtime."""

from __future__ import annotations

import configparser
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: configure_triton_cpu_distribution.py SITE_PACKAGES")

    site = pathlib.Path(sys.argv[1]).resolve()
    metadata_dirs = sorted(site.glob("triton-*.dist-info"))
    if len(metadata_dirs) != 1:
        raise SystemExit(
            f"expected one Triton dist-info directory under {site}, "
            f"found {len(metadata_dirs)}"
        )

    entry_points = metadata_dirs[0] / "entry_points.txt"
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(entry_points)
    if "triton.backends" not in parser:
        raise SystemExit(f"missing triton.backends entry points: {entry_points}")
    if not (site / "triton/backends/cpu/compiler.py").is_file():
        raise SystemExit("the packaged Triton CPU compiler backend is missing")

    parser["triton.backends"] = {"cpu": "triton.backends.cpu"}
    with entry_points.open("w", encoding="utf-8") as stream:
        parser.write(stream, space_around_delimiters=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
