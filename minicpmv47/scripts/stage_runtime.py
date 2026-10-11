#!/usr/bin/env python3
"""Package immutable, already validated runtime inputs without model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile


def copy_tree(source, destination):
    shutil.copytree(source, destination, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--triton', type=Path, required=True)
    parser.add_argument('--installed', type=Path, required=True)
    parser.add_argument('--wheels', type=Path, required=True)
    parser.add_argument('--vllm', type=Path)
    parser.add_argument('--cpython', type=Path, help='Qualified Linux CPython snapshot')
    parser.add_argument('--cpu-tle-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    recipe = Path(__file__).resolve().parents[1]
    system = 'macos' if sys.platform == 'darwin' else 'cix-p1'
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite a staged runtime')
    if system == 'cix-p1' and (args.vllm is None or args.cpython is None):
        raise RuntimeError('Linux requires materialized vLLM and a qualified CPython snapshot')
    copy_tree(args.runtime, args.output)
    python_root = args.output / ('python' if system == 'macos' else 'venv')
    site = python_root / 'lib/python3.11/site-packages'
    for name in ('flag_gems', 'vllm_fl', 'triton'):
        path = site / name
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    for p in site.glob('__editable__*'):
        if p.is_file():
            p.unlink()
    for pattern in ('flag_gems*.dist-info', 'vllm_plugin_fl*.dist-info', 'triton*.dist-info'):
        for p in site.glob(pattern):
            shutil.rmtree(p)
    copy_tree(args.triton.resolve(), site / 'triton')
    # The prepared compiler differs here only by API documentation. Ship the
    # frontend from the locked source commit rather than an older copied doc.
    shutil.copy2(args.cpu_tle_source, site / 'triton/language/extra/cpu/tle_ops.py')
    wheels = sorted(args.wheels.glob('*.whl'))
    if len(wheels) != 2:
        raise RuntimeError('Expected the validated FlagGems and plugin wheels')
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            if not all((site / p).resolve().is_relative_to(site) for p in archive.namelist()):
                raise RuntimeError('Unsafe wheel member')
            archive.extractall(site)
    copy_tree(args.installed / 'flag_gems/runtime/backend/_arm/lib',
              site / 'flag_gems/runtime/backend/_arm/lib')
    if args.vllm:
        p = site / 'vllm'
        if p.exists():
            shutil.rmtree(p)
        copy_tree(args.vllm, p)
    if system == 'cix-p1':
        if args.vllm is None:
            raise RuntimeError('Linux editable vLLM must be materialized explicitly')
        if args.cpython is None:
            raise RuntimeError('Linux requires the host-qualified CPython snapshot')
        copy_tree(args.cpython, args.output / 'cpython')
        python_version = json.loads((args.cpython / 'snapshot.json').read_text())['version'].split()[0]
        for name in ('python', 'python3', 'python3.11'):
            p = python_root / 'bin' / name
            p.unlink(missing_ok=True)
            p.symlink_to('../../cpython/bin/python3.11')
        (python_root / 'pyvenv.cfg').write_text(
            'home = /opt/flagos-runtime/cpython/bin\ninclude-system-site-packages = false\n'
            f'version = {python_version}\n')
        fonts = args.output / 'share/fonts'
        fonts.mkdir(parents=True, exist_ok=True)
        shutil.copy2('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', fonts)
        shutil.copy2('/usr/share/doc/fonts-dejavu-core/copyright', fonts / 'LICENSE.dejavu')
    else:
        # Torch's library can itself be a symlink into the runtime root. Keep
        # one real file before redirecting every other OpenMP reference.
        canonical = args.output / 'lib/libomp.dylib'
        contents = canonical.resolve().read_bytes()
        canonical.unlink()
        canonical.write_bytes(contents)
        canonical.chmod(0o755)
        for p in args.output.rglob('libomp*.dylib'):
            if p == canonical:
                continue
            p.unlink()
            p.symlink_to(os.path.relpath(canonical, p.parent))
    metadata = site / 'triton-3.7.2+g36133df.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: triton\nVersion: 3.7.2+g36133df\n')
    (metadata / 'top_level.txt').write_text('triton\n')
    bindir = args.output / 'bin'
    if bindir.exists():
        shutil.rmtree(bindir)
    bindir.mkdir()
    for name in ('flagos-python', 'flagos-vllm'):
        shutil.copy2(recipe / 'scripts' / name, bindir / name)
        (bindir / name).chmod(0o755)
    share = args.output / 'share/flagos-minicpmv47'
    share.mkdir(parents=True)
    copy_tree(recipe / 'profiles' / system, share / 'profiles')
    copy_tree(recipe / 'validation', share / 'validation')
    shutil.copy2(recipe / 'sources.lock.json', share)
    lock = json.loads((recipe / 'sources.lock.json').read_text())
    manifest = dict(schema=1, platform=system, version=lock['version'],
                    sources=lock['sources'],
                    build_policy='Snapshot of verified Python/Torch/vLLM dependencies plus locked component sources and matched native/compiler artifacts; not a fresh from-source rebuild of all dependencies',
                    models_included=False, compiled_artifacts={})
    if system == 'cix-p1':
        manifest['cpython_snapshot'] = json.loads((args.output / 'cpython/snapshot.json').read_text())
    for p in [*(site / 'triton/_C').glob('*'), *(site / 'flag_gems/runtime/backend/_arm/lib').glob('*'),
              site / 'vllm/_C.abi3.so', site / 'torch/lib'/('libtorch_cpu.dylib' if system == 'macos' else 'libtorch_cpu.so')]:
        if p.is_file():
            manifest['compiled_artifacts'][str(p.relative_to(args.output))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (share / 'staging-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(staged=str(args.output), platform=system)))


if __name__ == '__main__':
    main()
