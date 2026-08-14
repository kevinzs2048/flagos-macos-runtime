#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*args: str, required: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )
    if required and result.returncode:
        raise RuntimeError(f"{' '.join(args)}: {result.stderr.strip()}")
    return result


def is_macho(path: Path) -> bool:
    return "Mach-O" in run("/usr/bin/file", "-b", str(path), required=False).stdout


def dependencies(path: Path) -> list[str]:
    lines = run("/usr/bin/otool", "-L", str(path)).stdout.splitlines()[1:]
    return [line.strip().split(" (compatibility", 1)[0] for line in lines]


def rpaths(path: Path) -> list[str]:
    lines = run("/usr/bin/otool", "-l", str(path)).stdout.splitlines()
    result = []
    armed = False
    for line in lines:
        if line.strip() == "cmd LC_RPATH":
            armed = True
        elif armed:
            match = re.match(r"\s*path (.*?) \(offset ", line)
            if match:
                result.append(match.group(1))
                armed = False
    return result


def install_id(path: Path) -> str | None:
    result = run("/usr/bin/otool", "-D", str(path), required=False)
    lines = [line.strip() for line in result.stdout.splitlines()[1:] if line.strip()]
    return lines[0] if lines else None


def add_rpath(path: Path, value: str) -> None:
    if value in rpaths(path):
        return
    result = run(
        "/usr/bin/install_name_tool", "-add_rpath", value, str(path), required=False
    )
    if result.returncode and "larger updated load commands do not fit" not in result.stderr:
        raise RuntimeError(result.stderr.strip())


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    files = []
    by_name: dict[str, list[Path]] = {}
    for directory, names, filenames in os.walk(root):
        names[:] = [
            name
            for name in names
            if name != "__pycache__" and not name.endswith(".dSYM")
        ]
        for name in filenames:
            path = Path(directory) / name
            by_name.setdefault(path.name, []).append(path)
            candidate = path.suffix in {".so", ".dylib", ".bundle"} or os.access(
                path, os.X_OK
            )
            if candidate and not path.is_symlink() and is_macho(path):
                files.append(path)

    changes = 0
    for path in files:
        description = run("/usr/bin/file", "-b", str(path), required=False).stdout
        if "universal binary" in description and "arm64" in description:
            handle, temporary = tempfile.mkstemp(
                prefix=path.name + ".", suffix=".arm64", dir=str(path.parent)
            )
            os.close(handle)
            try:
                run(
                    "/usr/bin/lipo",
                    str(path),
                    "-thin",
                    "arm64",
                    "-output",
                    temporary,
                )
                os.chmod(temporary, path.stat().st_mode)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        # install_name_tool invalidates ad-hoc signatures. Remove them before
        # rewriting; unsigned local alpha is intentional.
        run("/usr/bin/codesign", "--remove-signature", str(path), required=False)
        identity = install_id(path)
        if identity and identity.startswith("/") and not identity.startswith(
            ("/usr/lib/", "/System/")
        ):
            run(
                "/usr/bin/install_name_tool",
                "-id",
                f"@rpath/{path.name}",
                str(path),
            )
            changes += 1
        for old in dependencies(path):
            replacement = None
            if old.endswith("libpython3.11.dylib") and not old.startswith("@"):
                replacement = "@rpath/libpython3.11.dylib"
            elif old.endswith("libomp.dylib") and not old.startswith("@"):
                replacement = "@rpath/libomp.dylib"
            elif old.startswith("/") and not old.startswith(("/usr/lib/", "/System/")):
                candidates = by_name.get(Path(old).name, [])
                if candidates:
                    replacement = f"@rpath/{Path(old).name}"
            if replacement and replacement != old:
                run(
                    "/usr/bin/install_name_tool",
                    "-change",
                    old,
                    replacement,
                    str(path),
                )
                changes += 1
        for old_rpath in rpaths(path):
            if old_rpath.startswith("/"):
                run(
                    "/usr/bin/install_name_tool",
                    "-delete_rpath",
                    old_rpath,
                    str(path),
                )
                changes += 1
        if path.name == "libpython3.11.dylib":
            run(
                "/usr/bin/install_name_tool",
                "-id",
                "@rpath/libpython3.11.dylib",
                str(path),
            )
        add_rpath(path, "@loader_path")

    python = root / "python" / "bin" / "python3.11"
    if python.is_file():
        # Python extension modules load companion libraries through @rpath.
        # The build places relative links for those libraries in Runtime/lib;
        # do not rely on DYLD_* variables, which macOS may sanitize.
        for relative in (
            "@executable_path/../../lib",
            "@executable_path/../lib",
        ):
            add_rpath(python, relative)
    jit_runtime = root / "lib" / "libtriton_jit.dylib"
    if jit_runtime.is_file():
        add_rpath(jit_runtime, "@loader_path/../python/lib")
        add_rpath(
            jit_runtime,
            "@loader_path/../python/lib/python3.11/site-packages/torch/lib",
        )
    flag_gems_ops = (
        root
        / "python/lib/python3.11/site-packages/flag_gems/csrc/arm"
        / "libflag_gems_arm_ops.dylib"
    )
    if flag_gems_ops.is_file():
        add_rpath(flag_gems_ops, "@loader_path/../../../../../../../lib")
    # Apple Silicon enforces a valid code directory even for local unsigned
    # executables. This is an ad-hoc signature, not Developer ID signing.
    for path in files:
        run(
            "/usr/bin/codesign",
            "--force",
            "--sign",
            "-",
            "--timestamp=none",
            str(path),
        )
    print(f"relocated {len(files)} Mach-O files with {changes} dependency/rpath changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
