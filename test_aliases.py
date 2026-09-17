"""Small registration/tokenizer/mHC ABI tests; no full model load."""
import json
import copy
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from vllm.platforms import current_platform
from vllm_fl import register_model
from vllm.transformers_utils.config import get_config
from vllm.model_executor.models import ModelRegistry
from vllm_fl.models.telechat4 import TeleChat4MHC
from vllm_fl.configs.xingchen4 import XingChen4Config, Xing4_0Config
from vllm.config.speculative import SpeculativeConfig

register_model()
source = Path(sys.argv[1])
config = get_config(str(source), trust_remote_code=True)
assert isinstance(config, Xing4_0Config), type(config)
assert config.model_type == 'xing4_0'
assert config.num_residual_streams == 4
assert config.mhc_sinkhorn_iterations == 20
original_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
new = TeleChat4MHC(config)
assert new._xingchen4_v2
assert set(new.state_dict()) == {'hc_fn', 'hc_base', 'hc_scale'}
raw = json.loads((source / 'config.json').read_text())
raw.pop('model_type')
old = TeleChat4MHC(XingChen4Config(**raw))
torch.set_default_dtype(original_dtype)
index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
state = {}
for key in new.state_dict():
    name = 'model.layers.0.attn_hc.' + key
    with safe_open(source / index[name], framework='pt') as shard:
        state[key] = shard.get_tensor(name)
new.load_state_dict(state)
old.load_state_dict(state)
assert old._xingchen4_v2
torch.manual_seed(20260916)
for rows in [1, 2, 17]:
    streams = torch.randn(rows, 4, config.hidden_size).to(torch.bfloat16)
    branch = torch.randn(rows, config.hidden_size).to(torch.bfloat16)
    new_pre, old_pre = new.pre(streams.clone()), old.pre(streams.clone())
    assert all(torch.equal(a, b) for a, b in zip(new_pre, old_pre)), rows
    assert torch.equal(new.post(branch, new_pre[1].clone(), new_pre[2]),
                       old.post(branch, old_pre[1].clone(), old_pre[2])), rows
for arch in ['Xing4_0ForCausalLM', 'Xing4_0MTPModel']:
    cls = ModelRegistry.models[arch].load_model_cls()
    print('registered', arch, cls.__name__)
draft_config = SpeculativeConfig.hf_config_override(copy.deepcopy(config))
assert draft_config.architectures == ['Xing4_0MTPModel']
assert draft_config.model_type == 'deepseek_mtp'
assert draft_config.n_predict == 1
assert draft_config.num_residual_streams == 4
assert config.model_type == 'xing4_0'
tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=True, use_fast=False)
assert type(tokenizer).__name__ == 'Xing4_0Tokenizer'
prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': '请问6乘7等于多少？'}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
assert '<_bot></think>' in prompt
assert tokenizer.encode(prompt)
print('tokenizer', type(tokenizer).__name__, 'tokens', len(tokenizer.encode(prompt)))
print('ALIAS_ABI_TESTS_PASS')
if len(sys.argv) > 2:
    evidence = Path(sys.argv[2])
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({'ok': True, 'model_type': config.model_type,
        'mHC_rows_token_exact': [1, 2, 17], 'mHC_state_keys': list(state),
        'registered_architectures': ['Xing4_0ForCausalLM', 'Xing4_0MTPModel'],
        'tokenizer': type(tokenizer).__name__}, indent=2) + '\n')
