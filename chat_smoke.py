"""Real-tokenizer Chinese/English/arithmetic smoke and repeated greedy check."""
import json
import os
from pathlib import Path
import sys

from vllm import LLM, SamplingParams

kwargs = dict(model=os.environ['MODEL_DIR'], dtype='bfloat16', trust_remote_code=True,
              distributed_executor_backend='uni', max_model_len=1024, max_num_seqs=1,
              max_num_batched_tokens=1024, enforce_eager=True,
              enable_prefix_caching=False, disable_log_stats=True)
mtp = os.environ.get('XINGCHEN4_MTP_MODEL', '')
if mtp:
    kwargs['speculative_config'] = {'method': 'mtp', 'model': mtp, 'num_speculative_tokens': 1}
llm = LLM(**kwargs)
tokenizer = llm.get_tokenizer()
cases = [('arithmetic', '请直接回答：6乘7等于多少？只输出数字。', '42'),
         ('chinese', '请用一句中文介绍你自己。', None),
         ('english', 'Write one short sentence about the sky in English.', None)]
records = []
for name, text, expected in cases:
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    rounds = []
    for _ in range(2):
        output = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=96), use_tqdm=False)[0].outputs[0]
        rounds.append({'text': output.text, 'token_ids': list(output.token_ids)})
    assert rounds[0]['token_ids'] == rounds[1]['token_ids'], name
    assert rounds[0]['text'].strip(), name
    if expected:
        assert expected in rounds[0]['text'], rounds[0]['text']
    record = {'case': name, 'prompt': text, 'response': rounds[0]['text'],
              'tokens': len(rounds[0]['token_ids']), 'repeat_token_exact': True}
    records.append(record)
    print('CHAT_CASE', json.dumps(record, ensure_ascii=False), flush=True)
result = {'ok': True, 'model': os.environ['MODEL_DIR'], 'mtp_model': mtp,
          'tokenizer': type(tokenizer).__name__, 'cases': records}
evidence = Path(sys.argv[1])
evidence.parent.mkdir(parents=True, exist_ok=True)
evidence.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print('CHAT_SMOKE_PASS', flush=True)
