"""Export a runtime without dropping dependency packages named 'cache'."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = a.runtime.resolve()
output = a.output.resolve()
if output.exists():
    raise FileExistsError(output)
if output.is_relative_to(root):
    raise ValueError('Archive output must be outside the runtime directory')
output.parent.mkdir(parents=True, exist_ok=True)
# bsdtar exclusion patterns can also match nested dependency directories.
# Select the root members explicitly, omitting only the runtime's own cache.
members = sorted(path.name for path in root.iterdir() if path.name != 'cache')
subprocess.run(['tar', '--no-xattrs', '--no-mac-metadata', '-czf', str(output),
                '-C', str(root), *members],
               env=dict(os.environ, COPYFILE_DISABLE='1'), check=True)
digest = hashlib.sha256()
with output.open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
        digest.update(block)
sha = digest.hexdigest()
output.with_name(output.name + '.sha256').write_text(f'{sha}  {output.name}\n')
print(json.dumps(dict(archive=str(output), bytes=output.stat().st_size, sha256=sha)))
