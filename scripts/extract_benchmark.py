"""Extract the unmodified Runtime benchmark's JSON evidence from its log."""
import json
from pathlib import Path
import sys

prefix = 'TELECHAT4_W4A8_STAGE_BENCH_JSON='
matches = [line.split(prefix, 1)[1] for line in Path(sys.argv[1]).read_text().splitlines()
           if prefix in line]
if len(matches) != 1:
    raise RuntimeError(f'Expected one finished benchmark, found {len(matches)}')
data = json.loads(matches[0])
assert all(data['token_exact'].values()), 'Repeated greedy outputs diverged'
output = Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(data['rates'], indent=2))
