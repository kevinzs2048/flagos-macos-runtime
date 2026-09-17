"""Verify all non-cache Runtime files against the packaged integrity manifest."""
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / 'share/flagos/runtime-files.sha256.json').read_text())
failures = []
for name, meta in manifest['files'].items():
    path = root / name
    if not path.exists() and not path.is_symlink():
        failures.append(name + ': missing')
        continue
    if path.is_symlink():
        actual = hashlib.sha256(os.readlink(path).encode()).hexdigest()
    else:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                h.update(chunk)
        actual = h.hexdigest()
    if actual != meta['sha256']:
        failures.append(name + ': checksum')
if failures:
    raise RuntimeError('\n'.join(failures))
print(f'RUNTIME_INTEGRITY_PASS {len(manifest["files"])} files')
