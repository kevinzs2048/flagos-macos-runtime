#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    files = {}
    for directory, names, filenames in os.walk(root):
        names[:] = sorted(name for name in names if name != "__pycache__")
        for filename in sorted(filenames):
            path = Path(directory) / filename
            # The bootstrap Runtime can already contain an older integrity
            # manifest at the output path. Never hash that stale file and
            # then overwrite it: a manifest cannot contain its own stable
            # digest.
            if path.resolve() == output:
                continue
            if path.is_symlink():
                digest = hashlib.sha256(os.readlink(path).encode()).hexdigest()
            else:
                hasher = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
            files[str(path.relative_to(root))] = {
                "sha256": digest,
                "size": path.lstat().st_size,
            }
    payload = {
        "schema_version": 1,
        "root": root.name,
        "file_count": len(files),
        "files": files,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
