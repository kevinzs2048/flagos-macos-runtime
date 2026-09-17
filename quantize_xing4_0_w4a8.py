#!/usr/bin/env python3
"""Stream a Xing4_0 BF16 checkpoint into the validated Mac W4A8 G128 ABI.

Only the quantization selection/configuration is reused from the earlier model;
every quantized value is recomputed from the supplied source weights.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from flagos_compressor.quantizers.mse_int4 import mse_int4_quantize


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reference-metadata', type=Path,
                    default=Path(__file__).resolve().parent / 'recipe')
    ap.add_argument('--threads', type=int, default=14)
    args = ap.parse_args()
    source, output, reference = args.source.resolve(), args.output.resolve(), args.reference_metadata.resolve()
    if output.exists():
        raise RuntimeError(f'Refusing to overwrite {output}')
    config = json.loads((source / 'config.json').read_text())
    expected = {'model_type': 'xing4_0', 'hidden_size': 3584, 'num_hidden_layers': 40,
                'n_routed_experts': 64, 'num_experts_per_tok': 4}
    if any(config.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Unexpected source model contract')
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    manifest = copy.deepcopy(json.loads((reference / 'quantization_manifest.json').read_text()))
    selected = {name for name, meta in manifest['tensors'].items()
                if meta['format'] == 'compressed-tensors-int-quantized-int4'}
    if len(selected) != 7616 or not selected.issubset(index['weight_map']):
        raise RuntimeError('Quantization selection does not match the new checkpoint')
    if manifest['algorithm'] != {'name': 'mse_grid_search', 'num_bits': 4,
                                 'activation_num_bits': 8, 'strategy': 'group',
                                 'group_size': 128, 'n_candidates': 200, 'chunk_size': 4096}:
        raise RuntimeError('Unexpected reference quantization recipe')
    shards = sorted(set(index['weight_map'].values()))
    if len(shards) != 41 or len(index['weight_map']) != 8307:
        raise RuntimeError('Unexpected source inventory')
    if any(not (source / name).is_file() for name in shards):
        raise RuntimeError('Missing source shards')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    started = time.time()
    mapping, hashes, errors = {}, {}, []
    total_bytes = converted = kept = excluded = 0
    for shard_number, shard in enumerate(shards, 1):
        shard_start = time.perf_counter()
        tensors = {}
        with safe_open(source / shard, framework='pt', device='cpu') as f:
            for name in sorted(f.keys()):
                if name.startswith('model.layers.40.'):
                    excluded += 1
                    continue
                weight = f.get_tensor(name)
                if name in selected:
                    meta = manifest['tensors'][name]
                    if list(weight.shape) != meta['logical_shape'] or weight.dtype != torch.bfloat16:
                        raise RuntimeError(f'Unexpected shape/dtype: {name}')
                    quant, scales = mse_int4_quantize(weight, group_size=128, n_candidates=200, chunk_size=4096)
                    tensors[name], tensors[meta['scale']] = quant.contiguous(), scales.contiguous()
                    if len(errors) < 10:
                        rows = min(4, weight.shape[0])
                        restored = quant[:rows].float() * scales[:rows].float().repeat_interleave(128, dim=1)
                        errors.append({'tensor': name, 'sample_mse': (restored - weight[:rows].float()).square().mean().item()})
                    converted += 1
                else:
                    tensors[name] = weight.contiguous()
                    kept += 1
        if tensors:
            save_file(tensors, output / shard, metadata={'format': 'pt'})
            for name, tensor in tensors.items():
                mapping[name] = shard
                total_bytes += tensor.numel() * tensor.element_size()
        del tensors
        hashes[shard] = digest(source / shard)
        elapsed = time.time() - started
        print(json.dumps({'shard': shard_number, 'shards': len(shards), 'converted': converted,
                          'kept': kept, 'mtp_excluded': excluded, 'elapsed_s': round(elapsed, 2),
                          'shard_s': round(time.perf_counter() - shard_start, 2)}), flush=True)
    if (converted, kept, excluded, len(mapping)) != (7616, 479, 212, 15711):
        raise RuntimeError('Final inventory mismatch; output must not be published')
    for name in ['config.json', 'configuration.json', 'configuration_xing4_0.py',
                 'modeling_xing4_0.py', 'tokenization_xing4_0.py', 'tokenizer.model',
                 'tokenizer_config.json', 'generation_config.json', 'chat_template.jinja']:
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    config['torch_dtype'] = 'bfloat16'
    config['quantization_config'] = json.loads((reference / 'config.json').read_text())['quantization_config']
    write_json(output / 'config.json', config)
    write_json(output / 'model.safetensors.index.json', {'metadata': {'total_size': total_bytes}, 'weight_map': mapping})
    manifest['source'] = {'path': str(source), 'config_sha256': digest(source / 'config.json'),
                          'index_sha256': digest(source / 'model.safetensors.index.json'),
                          'shard_sha256': hashes, 'mtp_excluded_tensors': excluded}
    write_json(output / 'quantization_manifest.json', manifest)
    write_json(output / 'quantization_report.json', {'backend': 'cpu', 'threads': args.threads,
               'started_at': started, 'finished_at': time.time(), 'elapsed_s': time.time() - started,
               'converted': converted, 'kept': kept, 'mtp_excluded': excluded,
               'output_tensors': len(mapping), 'indexed_bytes': total_bytes,
               'sample_errors': errors, 'status': 'complete'})
    print('QUANTIZATION_COMPLETE', output, total_bytes, flush=True)


if __name__ == '__main__':
    main()
