#!/usr/bin/env python3
"""Run a Cix P1 CPU benchmark with an explicit, temporary frequency floor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, record):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(record, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def write_floor(path, value):
    path.write_text(value)
    deadline = time.monotonic() + 5
    # Frequency QoS policy work may update sysfs readback asynchronously.
    while path.read_text().strip() != value:
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Frequency floor readback timed out: {path}')
        time.sleep(.01)


def restore(record):
    expected = {f'/sys/devices/system/cpu/cpufreq/policy{cpu}' for cpu in (0, 6, 8, 10)}
    if set(record['original']) != expected:
        raise ValueError('Journal does not describe the four validated Cix P1 policies')
    for values in record['original'].values():
        if not values['scaling_min_freq'].isdigit():
            raise ValueError('Invalid frequency in journal')
    errors = {}
    for name, values in record['original'].items():
        try:
            p = Path(name) / 'scaling_min_freq'
            write_floor(p, values['scaling_min_freq'])
        except (OSError, RuntimeError) as error:
            errors[name] = str(error)
    record['restore_errors'] = errors
    record['restored'] = not errors
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True,
                        help='New JSON journal; saved before changing any policy')
    parser.add_argument('--restore', action='store_true',
                        help='Restore a previous journal after an interrupted run')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.platform != 'linux' or os.uname().machine != 'aarch64':
        parser.error('This policy is for the validated Linux AArch64 Cix P1')
    if os.geteuid() != 0:
        parser.error('Root is required to write the host CPU frequency floor')
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with Path('/run/lock/flagos-cix-p1-frequency.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.restore:
            record = json.loads(args.state.read_text())
            ok = restore(record)
            save(args.state, record)
            return 0 if ok else 1
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            parser.error('A command is required')
        if args.state.exists():
            parser.error('Refusing to replace an existing frequency journal')
        original = {}
        for cpu, expected in ((0, {0, 1}), (6, {6, 7}), (8, {8, 9}), (10, {10, 11})):
            p = Path(f'/sys/devices/system/cpu/cpufreq/policy{cpu}')
            if set(map(int, (p / 'related_cpus').read_text().split())) != expected:
                parser.error(f'Unexpected Cix P1 CPU topology: {p}')
            original[str(p)] = {n: (p / n).read_text().strip() for n in
                               ('scaling_min_freq', 'scaling_max_freq', 'scaling_governor')}
        record = dict(original=original, restored=False, complete=False)
        save(args.state, record)
        child = None
        def interrupted(signum, frame):
            raise SystemExit(128 + signum)
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, interrupted)
        try:
            for name, values in original.items():
                write_floor(Path(name) / 'scaling_min_freq', values['scaling_max_freq'])
            record['applied'] = {name: (Path(name) / 'scaling_min_freq').read_text().strip()
                                 for name in original}
            save(args.state, record)
            child = subprocess.Popen(command, start_new_session=True)
            record['returncode'] = child.wait()
            record['complete'] = True
            return record['returncode']
        finally:
            for signum in (signal.SIGINT, signal.SIGTERM):
                signal.signal(signum, signal.SIG_IGN)
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            ok = restore(record)
            save(args.state, record)
            if not ok:
                raise RuntimeError(f'Frequency restoration failed; see {args.state}')


if __name__ == '__main__':
    sys.exit(main())
