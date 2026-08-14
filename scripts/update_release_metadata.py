#!/usr/bin/env python3
from __future__ import annotations

import hashlib
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


def main() -> int:
    required = [
        RUNTIME_ASSET,
        WHEELHOUSE_ASSET,
        ROOT / "install.sh",
        ROOT / "runtime-manifest.json",
        ROOT / "sources.lock.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing release input(s): " + ", ".join(missing))

    installer_digest = sha256(ROOT / "install.sh")
    (ROOT / "install.sh.sha256").write_text(
        f"{installer_digest}  install.sh\n", encoding="utf-8"
    )

    runtime_parts = split_runtime()
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
