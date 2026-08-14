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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    checksummed = [
        (RUNTIME_ASSET, RUNTIME_ASSET.name),
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
