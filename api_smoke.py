"""Test the Mac OpenAI-compatible server using only the quantized target."""
import json
from pathlib import Path
import sys
import time
import urllib.request

base, evidence = sys.argv[1], Path(sys.argv[2])
model_name = sys.argv[3] if len(sys.argv) > 3 else 'xing4-0-w4a8'


def call(path, data=None):
    body = json.dumps(data, ensure_ascii=False).encode() if data is not None else None
    request = urllib.request.Request(base + path, data=body,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        assert response.status == 200
        return json.load(response)


models = call('/v1/models')
assert any(m['id'] == model_name for m in models['data'])
cases = [('arithmetic', '请直接回答：6乘7等于多少？只输出数字。'),
         ('chinese', '请用一句中文介绍你自己。'),
         ('english', 'Write one short sentence about the sky in English.')]
records = []
for name, prompt in cases:
    replies = []
    elapsed = []
    for _ in range(2):
        start = time.perf_counter()
        output = call('/v1/chat/completions', {'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
            'max_tokens': 96, 'chat_template_kwargs': {'enable_thinking': False}})
        elapsed.append(time.perf_counter() - start)
        replies.append(output['choices'][0]['message']['content'])
    assert replies[0] == replies[1] and replies[0].strip(), (name, replies)
    if name == 'arithmetic':
        assert '42' in replies[0], replies[0]
    record = {'case': name, 'response': replies[0], 'repeat_text_exact': True,
              'request_elapsed_s': elapsed}
    records.append(record)
    print('API_CASE', json.dumps(record, ensure_ascii=False), flush=True)
evidence.parent.mkdir(parents=True, exist_ok=True)
evidence.write_text(json.dumps({'ok': True, 'mtp': False, 'model': model_name,
                                'cases': records}, ensure_ascii=False, indent=2) + '\n')
print('MAC_TARGET_ONLY_API_PASS', flush=True)
