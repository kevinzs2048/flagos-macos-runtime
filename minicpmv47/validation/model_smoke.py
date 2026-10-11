"""Full-model phase worker validation of the packaged MiniCPM runtime.
Each generation and its tokens/timing are saved atomically before proceeding.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--worker', choices=('old', 'phase'), required=True)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--vision', action='store_true')
p.add_argument('--standalone', action='store_true')
p.add_argument('--gems-source', type=Path)
p.add_argument('--package-site', type=Path)
p.add_argument('--plugin-source', type=Path)
p.add_argument('--heap-policy', choices=('retained', 'trim'), default='retained',
               help='Keep warm allocator pages, or release unused pages before every benchmark request')
a = p.parse_args()
if a.repeats < 1:
    p.error('--repeats must be positive')
if a.output.exists():
    raise RuntimeError(f'Refusing to replace existing report {a.output}')
report = dict(complete=False, scope='Packaged runtime full-model validation',
              model=a.model, worker=a.worker, pid=os.getpid(), cases=[], runs=[],
              heap_policy=a.heap_policy,
              python=dict(executable=sys.executable, version=sys.version, compiler=platform.python_compiler()),
              env={k:v for k,v in os.environ.items() if k.startswith(('FL_', 'FLAGGEMS_', 'MINICPM_', 'OMP_', 'VLLM_CPU_', 'GOMP_', 'KMP_', 'MIMALLOC_', 'TCMALLOC_'))})
report['cpu_governors'] = {str(path):path.read_text().strip()
    for path in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor')}
report['cpu_frequency_policy'] = {
    str(path): {name: (path/name).read_text().strip()
                for name in ('related_cpus', 'scaling_min_freq', 'scaling_max_freq', 'scaling_governor')
                if (path/name).exists()}
    for path in Path('/sys/devices/system/cpu/cpufreq').glob('policy*')}
if a.standalone:
    report['scope'] = 'Latest FlagGems and vllm-plugin-FL submission sources with unchanged validated FlagTree CPU compiler; deployed Python/Torch/vLLM supply dependencies only'

def save():
    tmp = a.output.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(a.output)

save()
try:
    from vllm import LLM, SamplingParams
    from vllm_fl.models.minicpmv4_7.registration import worker_options
    worker_args = worker_options() if a.worker == 'old' else dict(worker_cls='vllm_fl.worker.cpu_phase_worker.CPUPhaseWorker')
    memory_args = dict(kv_cache_memory_bytes=int(os.environ['MINICPM_KVCACHE_BYTES'])) if os.getenv('MINICPM_KVCACHE_BYTES') else {}
    start = time.perf_counter()
    llm = LLM(model=a.model, **worker_args, **memory_args, trust_remote_code=True,
              dtype='bfloat16', max_model_len=4096, max_num_batched_tokens=512,
              max_num_seqs=1, mamba_ssm_cache_dtype='float32',
              distributed_executor_backend='uni', enforce_eager=True,
              enable_prefix_caching=False, generation_config='vllm',
              disable_log_stats=False, limit_mm_per_prompt={'image': int(a.vision), 'video': 0})
    worker = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker
    report.update(load_seconds=time.perf_counter()-start,
                  worker_class=type(worker).__module__+'.'+type(worker).__name__)
    import flag_gems, triton, vllm_fl
    import vllm_fl.worker.cpu_thread_policy as policy
    modules = [flag_gems, triton, vllm_fl, policy]
    if a.worker == 'phase':
        import vllm_fl.worker.cpu_phase_worker as phase_worker
        modules.append(phase_worker)
    report['loaded_modules'] = {m.__name__:m.__file__ for m in modules}
    if a.package_site:
        import sys
        unexpected = {name:module.__file__ for name,module in sys.modules.items()
            if name.startswith(('flag_gems','vllm_fl','vllm.','triton.'))
            and getattr(module,'__file__',None)
            and not Path(module.__file__).is_relative_to(a.package_site)}
        assert not unexpected,unexpected
        report['unexpected_package_modules']=unexpected
        report['scope']='Packaged runtime model validation; all component imports under package site; no host source/adapter mount'
        report['package_site']=str(a.package_site)
    paths = [Path(m.__file__) for m in modules]
    paths += [Path(os.environ[k]) for k in ('FLAGGEMS_ARM_NATIVE_LIBRARY', 'FLAGGEMS_ARM_SILU_LIBRARY', 'FLAGGEMS_ARM_SILU_BITCODE') if k in os.environ]
    report['sha256'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    from vllm_fl.quantization.streamed_experts import StreamedExperts
    StreamedExperts._release_unused_pages()
    save()
    tok = llm.get_tokenizer()
    for question, expected in [('只输出结果：2+3等于多少？', '5'), ('中国的首都是哪里？请只回答城市名。', '北京')]:
        prompt = tok.apply_chat_template([dict(role='user', content=question)], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=64), use_tqdm=False)[0]
        row = dict(question=question, text=out.outputs[0].text, token_ids=list(out.outputs[0].token_ids))
        row['passed'] = expected in row['text']
        report['cases'].append(row)
        save()
        assert row['passed'], row
    if a.vision:
        from PIL import Image, ImageDraw, ImageFont
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
        image = Image.new('RGB', (896,448), 'white')
        font_path = next(path for path in (Path(os.getenv('FLAGOS_FONT_PATH','/nonexistent')), Path('/System/Library/Fonts/Supplemental/Arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')) if path.is_file())
        ImageDraw.Draw(image).text((70,155), 'FLAGOS 2026', font=ImageFont.truetype(str(font_path),90), fill='black')
        for name, image, question, expected in [('red', Image.new('RGB',(448,448),'red'), '图片的主要颜色是什么？只回答颜色。', ('红','red')), ('ocr', image, '请逐字读取图片中的英文和数字。', ('FLAGOS','2026'))]:
            prompt = processor.apply_chat_template([dict(role='user', content=[dict(type='image'), dict(type='text', text=question)])], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            out = llm.generate([dict(prompt=prompt, multi_modal_data=dict(image=image))], SamplingParams(temperature=0,max_tokens=64), use_tqdm=False)[0]
            text = out.outputs[0].text
            passed = any(word.lower() in text.lower() for word in expected) if name == 'red' else all(word.lower() in text.lower() for word in expected)
            report['cases'].append(dict(question=name,text=text,token_ids=list(out.outputs[0].token_ids),passed=passed))
            save()
            assert passed, report['cases'][-1]
    ids = tok.encode('请解释机器学习的训练过程，并举出具体例子。'*100, add_special_tokens=False)[:512]
    assert len(ids) == 512
    reference = None
    for repeat in range(a.repeats + 1):
        if a.heap_policy == 'trim':
            StreamedExperts._release_unused_pages()
        start = time.perf_counter()
        out = llm.generate([dict(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=128,min_tokens=128,ignore_eos=True),use_tqdm=False)[0]
        wall = time.perf_counter()-start
        metrics = out.metrics
        tokens = list(out.outputs[0].token_ids)
        assert metrics and not metrics.is_corrupted and len(tokens) == 128
        if reference is None:
            reference = tokens
        row = dict(repeat=repeat,warmup=repeat==0,token_ids=tokens,wall_seconds=wall,
                   tokens_equal=tokens==reference,
                   prefill_seconds=metrics.first_token_ts-metrics.scheduled_ts,
                   decode_seconds=metrics.last_token_ts-metrics.first_token_ts,
                   execution_seconds=metrics.last_token_ts-metrics.scheduled_ts)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        row.update(minor_faults=usage.ru_minflt,
                   peak_rss_bytes=usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
        row.update(pp512=512/row['prefill_seconds'],tg128=127/row['decode_seconds'],
                   total=640/row['execution_seconds'],total_definition='640/(first-token prefill time + 127 decode intervals)',
                   phase_calls=dict(worker.thread_calls),observed_threads=dict(worker.observed_threads))
        report['runs'].append(row)
        save()
        print('BENCH',json.dumps({k:v for k,v in row.items() if k!='token_ids'}),flush=True)
        assert row['tokens_equal'], 'Tokens changed between repetitions'
    rows = [row for row in report['runs'] if not row['warmup']]
    report['medians'] = {key:statistics.median(row[key] for row in rows) for key in ('pp512','tg128','total','execution_seconds','wall_seconds')}
    report['ranges'] = {key:dict(min=min(row[key] for row in rows), max=max(row[key] for row in rows))
                        for key in ('pp512','tg128','total')}
    report.update(complete=True,passed=True)
    save()
except BaseException as error:
    report.update(error=repr(error),passed=False)
    save()
    raise
