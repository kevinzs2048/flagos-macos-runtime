"""Launch the packaged CLI and verify one OpenAI-compatible chat request."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--profile', choices=('1b', '35b'), required=True)
p.add_argument('--model', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--port', type=int, default=18080)
p.add_argument('--timeout', type=int, default=600)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
report = dict(complete=False, passed=False, profile=a.profile, model=a.model)


def save():
    tmp = a.output.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(a.output)


command = [str(a.runtime / 'bin/flagos-vllm'), '--profile', a.profile,
           'serve', a.model, '--served-model-name', 'minicpm',
           '--trust-remote-code', '--dtype', 'bfloat16', '--max-model-len', '4096',
           '--max-num-batched-tokens', '512', '--max-num-seqs', '1',
           '--worker-cls', 'vllm_fl.worker.cpu_phase_worker.CPUPhaseWorker',
           '--distributed-executor-backend', 'uni', '--enforce-eager',
           '--kv-cache-memory-bytes', '268435456',
           '--mamba-ssm-cache-dtype', 'float32', '--generation-config', 'vllm',
           '--limit-mm-per-prompt', '{"image":0,"video":0}',
           '--host', '127.0.0.1', '--port', str(a.port)]
report['command'] = command
save()
process = None
try:
    with a.output.with_suffix('.server.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        url = f'http://127.0.0.1:{a.port}'
        deadline = time.monotonic() + a.timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'Server exited with code {process.returncode}')
            try:
                with urllib.request.urlopen(url + '/health', timeout=2) as response:
                    assert response.status == 200
                break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Server readiness timeout')
                time.sleep(1)
        body = dict(model='minicpm', messages=[dict(role='user', content='只输出结果：2+3等于多少？')],
                    temperature=0, max_tokens=64, chat_template_kwargs=dict(enable_thinking=False))
        request = urllib.request.Request(url + '/v1/chat/completions',
                                         data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        report['response'] = result
        assert '5' in result['choices'][0]['message']['content'], result
        report.update(complete=True, passed=True)
        save()
except BaseException as error:
    report['error'] = repr(error)
    save()
    raise
finally:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
