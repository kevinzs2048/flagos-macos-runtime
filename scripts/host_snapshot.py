"""Read-only Mac conditions accompanying each performance evidence file."""
import datetime
import json
from pathlib import Path
import subprocess
import sys


def read(cmd):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False).stdout.strip()


result = {'time': datetime.datetime.now().astimezone().isoformat(),
          'macos': read(['/usr/bin/sw_vers']),
          'hardware': read(['/usr/sbin/sysctl', 'hw.model', 'machdep.cpu.brand_string', 'hw.memsize']),
          'thermal_status': read(['/usr/bin/pmset', '-g', 'therm']),
          'process_load_top_20': read(['/bin/ps', '-Ao', 'pid,pcpu,pmem,comm', '-r']).splitlines()[:21],
          'memory_pressure': read(['/usr/bin/memory_pressure', '-Q'])}
output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
