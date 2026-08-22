#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0-alpha.1"
RUNTIME_ASSET = (
    ROOT
    / "artifacts"
    / f"flagos-runtime-{VERSION}-darwin-arm64-m5pro.tar.gz"
)
WHEELHOUSE_ASSET = (
    ROOT
    / "artifacts"
    / f"flagos-wheelhouse-{VERSION}-cp311-darwin-arm64.tar.gz"
)
RUNTIME_PART_SIZE = 50 * 1024 * 1024
RUNTIME_PARTS_MANIFEST = Path(str(RUNTIME_ASSET) + ".parts")
RUNTIME_ROOT = (
    ROOT
    / "build"
    / f"runtime-{VERSION}"
    / f"flagos-runtime-{VERSION}-darwin-arm64-m5pro"
)
ACCEPTANCE = ROOT / "RELEASE_ACCEPTANCE.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_runtime() -> list[Path]:
    for stale in RUNTIME_ASSET.parent.glob(RUNTIME_ASSET.name + ".part-*"):
        stale.unlink()

    parts: list[Path] = []
    with RUNTIME_ASSET.open("rb") as source:
        index = 0
        while chunk := source.read(RUNTIME_PART_SIZE):
            part = Path(f"{RUNTIME_ASSET}.part-{index:03d}")
            part.write_bytes(chunk)
            parts.append(part)
            index += 1

    manifest_lines = [f"{sha256(part)}  {part.name}" for part in parts]
    RUNTIME_PARTS_MANIFEST.write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    return parts


def text_file_count(root: Path) -> int:
    """Use the same text-file definition as the release relocation audit."""
    count = 0
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size > 8 << 20
        ):
            continue
        if b"\0" not in path.read_bytes()[:8192]:
            count += 1
    return count


def replace_once(text: str, pattern: str, replacement: str) -> str:
    updated, replacements = re.subn(pattern, replacement, text, count=1)
    if replacements != 1:
        raise RuntimeError(f"release acceptance pattern did not match: {pattern}")
    return updated


def update_acceptance(runtime_parts: list[Path]) -> None:
    runtime_manifest = json.loads(
        (RUNTIME_ROOT / "share/flagos/runtime-files.sha256.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_files = int(runtime_manifest["file_count"])
    text_files = text_file_count(RUNTIME_ROOT)
    runtime_mib = RUNTIME_ASSET.stat().st_size / (1024 * 1024)
    wheelhouse_mib = WHEELHOUSE_ASSET.stat().st_size / (1024 * 1024)

    text = ACCEPTANCE.read_text(encoding="utf-8")
    text = replace_once(
        text,
        r"\| Runtime logical archive \(\d+ checksummed Release parts\) \|"
        r" [0-9.]+ MiB \| `[0-9a-f]{64}` \|",
        f"| Runtime logical archive ({len(runtime_parts)} checksummed Release parts) "
        f"| {runtime_mib:.1f} MiB | `{sha256(RUNTIME_ASSET)}` |",
    )
    text = replace_once(
        text,
        rf"\| `flagos-wheelhouse-{re.escape(VERSION)}-cp311-darwin-arm64\.tar\.gz`"
        r" \| [0-9.]+ MiB \| `[0-9a-f]{64}` \|",
        f"| `flagos-wheelhouse-{VERSION}-cp311-darwin-arm64.tar.gz` "
        f"| {wheelhouse_mib:.1f} MiB | `{sha256(WHEELHOUSE_ASSET)}` |",
    )
    text = replace_once(
        text,
        r"Archive verification checked [0-9,]+ Runtime files, [0-9,]+ text files",
        "Archive verification checked "
        f"{runtime_files:,} Runtime files, {text_files:,} text files",
    )
    ACCEPTANCE.write_text(text, encoding="utf-8")


def main() -> int:
    required = [
        RUNTIME_ASSET,
        WHEELHOUSE_ASSET,
        ROOT / "install.sh",
        ROOT / "runtime-manifest.json",
        ROOT / "sources.lock.json",
        RUNTIME_ROOT / "share/flagos/runtime-files.sha256.json",
        ACCEPTANCE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing release input(s): " + ", ".join(missing))

    installer_digest = sha256(ROOT / "install.sh")
    (ROOT / "install.sh.sha256").write_text(
        f"{installer_digest}  install.sh\n", encoding="utf-8"
    )

    runtime_parts = split_runtime()
    update_acceptance(runtime_parts)
    checksummed = [
        (RUNTIME_PARTS_MANIFEST, RUNTIME_PARTS_MANIFEST.name),
        *((part, part.name) for part in runtime_parts),
        (WHEELHOUSE_ASSET, WHEELHOUSE_ASSET.name),
        (ROOT / "install.sh", "install.sh"),
        (ROOT / "runtime-manifest.json", "runtime-manifest.json"),
        (ROOT / "sources.lock.json", "sources.lock.json"),
    ]
    lines = [f"{sha256(path)}  {release_name}" for path, release_name in checksummed]
    (ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"updated release metadata for {VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
