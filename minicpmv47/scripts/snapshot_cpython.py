#!/usr/bin/env python3
"""Snapshot the qualified Linux CPython interpreter, stdlib and JIT headers.

Run this script with the interpreter used for host qualification. Model weights
and site-packages are deliberately excluded. The Docker base supplies libc.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import sysconfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or platform.machine() != 'aarch64' or sys.version_info[:2] != (3, 11):
        raise RuntimeError('Run with the qualified Linux arm64 CPython 3.11')
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(root)
    (root / 'bin').mkdir(parents=True)
    (root / 'lib').mkdir()
    executable = Path(sys.executable).resolve()
    shutil.copy2(executable, root / 'bin/python3.11')
    stdlib = Path(sysconfig.get_path('stdlib'))
    shutil.copytree(stdlib, root / 'lib/python3.11', symlinks=False,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.a', 'site-packages', 'dist-packages'))
    include = Path(sysconfig.get_path('include'))
    shutil.copytree(include, root / 'include/python3.11', symlinks=False)
    # Debian's dispatch header refers to a multiarch include outside its prefix.
    # This package is specifically Linux aarch64, so install that exact header.
    arch_header = Path('/usr/include/aarch64-linux-gnu/python3.11/pyconfig.h')
    if arch_header.exists():
        shutil.copy2(arch_header, root / 'include/python3.11/pyconfig.h')
    libpython = Path(sysconfig.get_config_var('LIBDIR')) / sysconfig.get_config_var('LDLIBRARY')
    if not libpython.is_file():
        raise FileNotFoundError(libpython)
    shutil.copy2(libpython.resolve(), root / 'lib/libpython3.11.so.1.0')
    for name in ('libpython3.11.so', 'libpython3.11.so.1'):
        (root / 'lib' / name).symlink_to('libpython3.11.so.1.0')
    # Keep configure provenance but make JIT library/config locations relocatable.
    for config in (root / 'lib/python3.11').glob('_sysconfigdata*.py'):
        if not config.is_file():
            continue
        contents = config.read_text()
        contents += '''\nimport os as _flagos_os
_flagos_prefix = _flagos_os.path.dirname(_flagos_os.path.dirname(_flagos_os.path.dirname(__file__)))
build_time_vars.update(LIBDIR=_flagos_os.path.join(_flagos_prefix, 'lib'),
                      LIBPL=_flagos_os.path.join(_flagos_prefix, 'lib/python3.11/config-3.11-aarch64-linux-gnu'),
                      BINDIR=_flagos_os.path.join(_flagos_prefix, 'bin'),
                      INCLUDEPY=_flagos_os.path.join(_flagos_prefix, 'include/python3.11'))
'''
        config.write_text(contents)
    license_path = Path('/usr/share/doc/python3.11/copyright')
    if license_path.is_file():
        shutil.copy2(license_path, root / 'COPYRIGHT.python3.11')
    manifest = dict(executable=str(executable), version=sys.version,
                    configure_args=sysconfig.get_config_var('CONFIG_ARGS'),
                    files={})
    for path in root.rglob('*'):
        if path.is_file() and not path.is_symlink():
            manifest['files'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'snapshot.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(snapshot=str(root), executable_sha256=manifest['files']['bin/python3.11'])))


if __name__ == '__main__':
    main()
