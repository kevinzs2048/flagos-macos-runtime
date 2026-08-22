#!/usr/bin/env python3
"""Make an embedded CPython sysconfig independent of its build prefix."""

from __future__ import annotations

import argparse
import pprint
import re
import runpy
from pathlib import Path


TOKEN = "@FLAGOS_RUNTIME_PYTHON@"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_root", type=Path)
    parser.add_argument("build_prefix")
    args = parser.parse_args()

    config = args.python_root / "lib/python3.11/_sysconfigdata__darwin_darwin.py"
    original = config.read_text(encoding="utf-8")
    if TOKEN not in original:
        namespace = runpy.run_path(str(config))
        values = namespace["build_time_vars"]
        build_dir = values.get("abs_builddir", "")
        source_dir = values.get("abs_srcdir", "")

        relocated: dict[str, object] = {}
        for key, value in values.items():
            if isinstance(value, str):
                value = value.replace(args.build_prefix, TOKEN)
                if build_dir:
                    value = value.replace(build_dir, "python-3.11.9")
                if source_dir:
                    value = value.replace(source_dir, "python-3.11.9")
            relocated[key] = value

        payload = pprint.pformat(relocated, width=100, sort_dicts=True)
        config.write_text(
            "# Relocated CPython configuration for the FlagOS Runtime.\n"
            "from pathlib import Path as _Path\n"
            f"build_time_vars = {payload}\n"
            "_runtime_prefix = str(_Path(__file__).resolve().parents[2])\n"
            "for _key, _value in tuple(build_time_vars.items()):\n"
            "    if isinstance(_value, str):\n"
            f"        build_time_vars[_key] = _value.replace({TOKEN!r}, _runtime_prefix)\n"
            "del _key, _value, _runtime_prefix, _Path\n",
            encoding="utf-8",
        )

    site_packages = args.python_root / "lib/python3.11/site-packages"
    for package in ("numpy", "scipy"):
        package_config = site_packages / package / "__config__.py"
        if not package_config.is_file():
            continue
        text = package_config.read_text(encoding="utf-8")
        # Binary wheels may retain an ephemeral PEP 517 build-environment
        # prefix in their informational show_config() payload.  It is not a
        # runtime search path, so replace the complete quoted value rather
        # than leaking a host-specific /private/var/folders location.
        text = re.sub(
            r'r?"[^"\n]*/private/var/folders/[^"\n]*"',
            '"build-environment"',
            text,
        )
        package_config.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
