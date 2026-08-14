"""Build-time Mach-O dependency audit for the self-contained Runtime."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_PREFIXES = (
    "/Users/",
    "/home/",
    "/opt/homebrew",
    "/opt/llvm-openmp",
    "/usr/local",
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )


def is_macho(path: Path) -> bool:
    result = _run("/usr/bin/file", "-b", str(path))
    return result.returncode == 0 and "Mach-O" in result.stdout


def macho_files(root: Path) -> Iterable[Path]:
    for directory, names, files in os.walk(root):
        names[:] = [
            name
            for name in names
            if name != "__pycache__" and not name.endswith(".dSYM")
        ]
        for name in files:
            path = Path(directory) / name
            candidate = path.suffix in {".so", ".dylib", ".bundle"} or os.access(
                path, os.X_OK
            )
            if candidate and not path.is_symlink() and is_macho(path):
                yield path


def dependencies(path: Path) -> list[str]:
    result = _run("/usr/bin/otool", "-L", str(path))
    if result.returncode:
        return []
    return [
        value
        for line in result.stdout.splitlines()[1:]
        if (value := line.strip().split(" (compatibility", 1)[0])
    ]


def rpaths(path: Path) -> list[str]:
    result = _run("/usr/bin/otool", "-l", str(path))
    if result.returncode:
        return []
    paths: list[str] = []
    armed = False
    for line in result.stdout.splitlines():
        if line.strip() == "cmd LC_RPATH":
            armed = True
        elif armed:
            match = re.match(r"\s*path (.*?) \(offset ", line)
            if match:
                paths.append(match.group(1))
                armed = False
    return paths


def audit_tree(root: Path) -> dict[str, Any]:
    records = []
    violations = []
    openmp = set()
    for path in macho_files(root):
        deps = dependencies(path)
        paths = rpaths(path)
        relative = str(path.relative_to(root))
        bad = []
        for item in deps + paths:
            forbidden_named = any(
                item.startswith(prefix) for prefix in FORBIDDEN_PREFIXES
            )
            forbidden_absolute = item.startswith("/") and not item.startswith(
                ("/usr/lib/", "/System/")
            )
            if forbidden_named or forbidden_absolute:
                bad.append(item)
        for dependency in deps:
            if dependency.endswith("libomp.dylib"):
                openmp.add(dependency)
        if bad:
            violations.append({"path": relative, "references": bad})
        records.append({"path": relative, "dependencies": deps, "rpaths": paths})
    openmp_files = sorted(root.rglob("libomp*.dylib"))
    openmp_resolved = sorted({str(path.resolve()) for path in openmp_files})
    return {
        "macho_count": len(records),
        "forbidden_references": violations,
        "openmp_install_names": sorted(openmp),
        "openmp_resolved_files": openmp_resolved,
        "single_openmp_runtime": len(openmp_resolved) == 1 and len(openmp) <= 1,
    }
