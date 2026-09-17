"""Validate every tensor in the signed-INT4 / BF16-scale Mac checkpoint."""
import json
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open

model, source, evidence = map(Path, sys.argv[1:4])
previous = Path(sys.argv[4]) if len(sys.argv) > 4 else None
started = time.perf_counter()
torch.set_num_threads(8)
index = json.loads((model / 'model.safetensors.index.json').read_text())
source_index = json.loads((source / 'model.safetensors.index.json').read_text())
manifest = json.loads((model / 'quantization_manifest.json').read_text())
config = json.loads((model / 'config.json').read_text())
quant = config['quantization_config']
assert config['model_type'] == 'xing4_0'
assert config['architectures'] == ['Xing4_0ForCausalLM']
assert quant['format'] == 'int-quantized'
group = next(iter(quant['config_groups'].values()))
assert group['weights']['num_bits'] == 4
assert group['weights']['group_size'] == 128
assert group['weights']['symmetric'] is True
assert group['input_activations'] == {'num_bits': 8, 'type': 'int', 'strategy': 'token',
                                     'symmetric': True, 'dynamic': True}
selected = {k: v for k, v in manifest['tensors'].items()
            if v['format'] == 'compressed-tensors-int-quantized-int4'}
original_names = {k for k in source_index['weight_map'] if not k.startswith('model.layers.40.')}
assert set(index['weight_map']) == original_names | {v['scale'] for v in selected.values()}
seen, total_bytes, count, kept = set(), 0, 0, 0
for shard in sorted(set(index['weight_map'].values())):
    prior_context = safe_open(previous / shard, framework='pt') if previous else nullcontext()
    with safe_open(model / shard, framework='pt') as f, safe_open(source / shard, framework='pt') as src, prior_context as prior:
        for name in f.keys():
            assert name not in seen, name
            seen.add(name)
            assert index['weight_map'][name] == shard
            tensor = f.get_tensor(name)
            if prior is not None:
                assert torch.equal(tensor, prior.get_tensor(name)), 'Previous tensor differs: ' + name
            total_bytes += tensor.numel() * tensor.element_size()
            if name in selected:
                meta = selected[name]
                assert tensor.dtype == torch.int8 and list(tensor.shape) == meta['logical_shape'], name
                assert tensor.min().item() >= -8 and tensor.max().item() <= 7, name
                scales = f.get_tensor(meta['scale'])
                assert scales.dtype == torch.bfloat16 and list(scales.shape) == meta['scale_shape'], name
                assert torch.isfinite(scales).all().item() and (scales > 0).all().item(), name
                count += 1
            elif name in original_names:
                original = src.get_tensor(name)
                assert tensor.dtype == original.dtype and torch.equal(tensor, original), name
                kept += 1
    print('validated', shard, count, kept, flush=True)
assert (len(seen), count, kept, total_bytes) == (15711, 7616, 479, 30927393984)
assert total_bytes == index['metadata']['total_size']
result = {'ok': True, 'architecture': config['architectures'][0], 'model_type': config['model_type'],
          'shards': len(set(index['weight_map'].values())), 'quantized_weights': count,
          'kept_exact_tensors': kept, 'mtp_excluded': 212, 'output_tensors': len(seen),
          'indexed_bytes': total_bytes, 'elapsed_s': time.perf_counter() - started}
if previous:
    result['previous_checkpoint_all_tensors_exact'] = True
    result['previous_checkpoint'] = str(previous)
evidence.parent.mkdir(parents=True, exist_ok=True)
evidence.write_text(json.dumps(result, indent=2) + '\n')
print('VALIDATION_PASS', json.dumps(result), flush=True)
