"""Same stage benchmark, adding cooldown OUTSIDE timing before each round.

No weights, kernels, sampling, forward hooks, rate formula, or scenario order
are changed. Cooldown is thermal idle time, not a filesystem/JIT cache flush.
Absolute chip temperature is unavailable without additional sensor privileges.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=['off', 'on'], required=True)
parser.add_argument('--cooldown-seconds', type=int, default=180)
parser.add_argument('--evidence-directory', type=Path, required=True)
args = parser.parse_args()
if args.cooldown_seconds < 180:
    raise ValueError('This retained cooldown protocol requires at least 180 seconds')
args.evidence_directory.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[1]
path = root / 'launcher/bench_xingchen4_w4a8_stages.py'
code = path.read_text()
sha = hashlib.sha256(code.encode()).hexdigest()
if sha != 'dcddfbddc3e3c4e2dcd60c08ce8f83ee83a86940df49d601dbbfadf905ed7cee':
    raise RuntimeError('Base benchmark changed; inspect its protocol before measuring')
draft = os.environ.get('TELECHAT4_MTP_MODEL', '').strip()
if bool(draft) != (args.mode == 'on'):
    raise RuntimeError('Launcher-selected speculative mode does not match requested mode')
os.environ['TELECHAT4_STAGE_BENCH_ROUNDS'] = '3'
if args.mode == 'on':
    if os.environ.get('TELECHAT4_MTP_TOKENS', '1') != '1':
        raise ValueError('Retained speculative protocol is exactly one token')

events = []


def snapshot(event, round_index):
    def read(command):
        return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False).stdout.strip()
    item = {'time': datetime.datetime.now().astimezone().isoformat(),
            'event': event, 'round': round_index + 1, 'mode': args.mode,
            'thermal_status': read(['/usr/bin/pmset', '-g', 'therm']),
            'top_processes': read(['/bin/ps', '-Ao', 'pid,pcpu,pmem,comm', '-r']).splitlines()[:21]}
    events.append(item)
    (args.evidence_directory / ('cooldown_' + args.mode + '.json')).write_text(
        json.dumps(events, indent=2) + '\n')


def cold_before_round(round_index):
    snapshot('cooldown_begin', round_index)
    print('COOLDOWN_BEGIN', args.mode, 'round', round_index + 1,
          'seconds', args.cooldown_seconds, flush=True)
    deadline = time.monotonic() + args.cooldown_seconds
    while time.monotonic() < deadline:
        time.sleep(min(30, max(0, deadline - time.monotonic())))
        print('COOLDOWN_WAIT', args.mode, 'round', round_index + 1,
              'remaining_s', max(0, round(deadline - time.monotonic())), flush=True)
    snapshot('cooldown_end_before_timed_round', round_index)
    print('COOLDOWN_END', args.mode, 'round', round_index + 1, flush=True)


anchor = '    for _ in range(rounds):\n        for name in ("pp512", "tg128", "total"):\n'
replacement = ('    for round_index in range(rounds):\n'
               '        cold_before_round(round_index)\n'
               '        for name in ("pp512", "tg128", "total"):\n')
if code.count(anchor) != 1:
    raise RuntimeError('Measured-round anchor must occur exactly once')
code = code.replace(anchor, replacement)
sample_anchor = '            elapsed_ms, tokens, calls, target_ms, draft_ms = run(name)\n'
if code.count(sample_anchor) != 1:
    raise RuntimeError('Measured-sample anchor must occur exactly once')
code = code.replace(sample_anchor, sample_anchor +
    '            print("COOL_SAMPLE", json.dumps({"round": round_index + 1, "scenario": name, '
    '"request_ms": elapsed_ms, "target_forwards": calls}), flush=True)\n')
protocol = {'mode': args.mode, 'rounds': 3, 'cooldown_seconds_before_each_round': args.cooldown_seconds,
            'mode_order': ['off', 'on'], 'scenario_order': ['pp512', 'tg128_baseline', 'tg128', 'total'],
            'base_benchmark_sha256': sha, 'only_changes': 'untimed per-round cooldown and untimed sample printing',
            'runtime': os.environ['FLAGOS_RUNTIME_ROOT'], 'target': os.environ['MODEL_DIR'],
            'draft_source': draft or None, 'draft_speculative_tokens': 1 if draft else 0,
            'warmup': 'original PP512 and TG128 shapes, four outputs each, before first cooldown',
            'absolute_chip_temperature_verified': False,
            'no_claim_of_filesystem_cache_cold_or_background_free_host': True}
(args.evidence_directory / ('protocol_' + args.mode + '.json')).write_text(
    json.dumps(protocol, indent=2) + '\n')
print('COOLED_BENCH_PROTOCOL', json.dumps(protocol), flush=True)
exec(compile(code, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path),
                                     'cold_before_round': cold_before_round})
snapshot('benchmark_finished', 2)
