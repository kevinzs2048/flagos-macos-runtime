"""Prove installed wheels and default CMake library discovery, no adapter PYTHONPATH."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('--prefix',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a = p.parse_args()
assert not a.output.exists()
for key in ('FLAGGEMS_ARM_NATIVE_LIBRARY','FLAGGEMS_ARM_SILU_LIBRARY','FLAGGEMS_ARM_SILU_BITCODE'):
    assert key not in os.environ, key
report = dict(complete=False,prefix=str(a.prefix),scope='Self-contained packaged runtime; default DSO/bitcode discovery; native, CPU TLE and ordinary Triton checks')

def save():
    tmp = a.output.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(report,f,indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(a.output)

save()
try:
    import flag_gems,vllm_fl
    from flag_gems.runtime.backend._arm.fused.native_ops import arm_native_ops
    from vllm_fl.models.minicpmv4_7.registration import register_model
    from vllm_fl.models.minicpmv4_7.configuration import MiniCPMV4_7Config
    from vllm_fl.quantization import arm_cpu_native
    from vllm_fl.worker import cpu_phase_worker
    modules = (flag_gems,vllm_fl,arm_cpu_native,cpu_phase_worker)
    report['loaded_modules'] = {m.__name__:m.__file__ for m in modules}
    assert all(Path(m.__file__).is_relative_to(a.prefix) for m in modules)
    torch.manual_seed(284)
    torch.set_num_threads(4)
    x = torch.randn(3,64)
    w = torch.randint(-8,8,(128,64),dtype=torch.int8)
    scales = torch.rand(128,2,dtype=torch.bfloat16)*.01
    native,raw = arm_native_ops('w4a8_g32'),arm_native_ops('w4a8_g32',raw=True)
    packed = native.pack(w,scales,0)
    assert torch.equal(native.linear(x,packed,128,0,4,-1),raw.linear(x,packed,128,0,4,-1))
    w8 = torch.randint(-127,128,(128,64),dtype=torch.int8)
    scales8 = torch.rand(128)*.01
    native,raw = arm_native_ops('w8a8'),arm_native_ops('w8a8',raw=True)
    packed = native.pack(w8,scales8,False)
    assert torch.equal(native.linear(x,packed,128,False,32,128),raw.linear(x,packed,128,False,32,128))
    norm_x = torch.randn(3,128,dtype=torch.bfloat16)
    gate = torch.randn_like(norm_x)
    weight = torch.ones(128,dtype=torch.bfloat16)
    native,raw = arm_native_ops('gdn'),arm_native_ops('gdn',raw=True)
    assert torch.equal(native.gdn_rmsnorm_gated(norm_x,weight,gate,1e-6,False),raw.gdn_rmsnorm_gated(norm_x,weight,gate,1e-6,False))
    from flag_gems.runtime.backend._arm.fused.gemma_norm import prepare_gemma_triton
    prepare_gemma_triton()
    for hidden in (128,1024,2048):
        value=torch.randn(16,hidden,dtype=torch.bfloat16)
        weight=torch.randn(hidden,dtype=torch.bfloat16)*.1
        reference=native.gemma_rmsnorm(value,weight,1e-6)
        assert torch.equal(reference,raw.gemma_rmsnorm(value,weight,1e-6))
        assert torch.equal(reference,native.gemma_rmsnorm.triton(value,weight,1e-6))
        residual=torch.randn_like(value)
        results=[]
        for op in (native.gemma_fused_add_rmsnorm.default,native.gemma_fused_add_rmsnorm.raw,native.gemma_fused_add_rmsnorm.triton):
            v=value.clone();r=residual.clone();op(v,r,weight,1e-6);results.append((v,r))
        assert all(torch.equal(results[0][0],r[0]) and torch.equal(results[0][1],r[1]) for r in results)
    register_model()
    dense = MiniCPMV4_7Config(architectures=['MiniCPMV4_7ForConditionalGeneration'],text_config=dict(model_type='qwen3_5_text'))
    assert dense.architectures == ['MiniCPMV4_7DenseForConditionalGeneration']
    import sys
    unexpected = {name:m.__file__ for name,m in sys.modules.items()
        if name.startswith(('flag_gems','vllm_fl')) and getattr(m,'__file__',None)
        and not Path(m.__file__).is_relative_to(a.prefix)}
    assert not unexpected,unexpected
    libraries = list((a.prefix/'flag_gems/runtime/backend/_arm/lib').iterdir())
    report['installed_library_sha256'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in libraries if path.is_file()}
    report.update(complete=True,passed=True,checks=['G32 native/Raw exact','W8 native/Raw exact','GDN norm native/Raw exact','dense/MoE registration','package import isolation','default library discovery','Gemma native/Raw/ordinary Triton plain and fused exact D128/1024/2048'])
    save()
except BaseException as error:
    report.update(error=repr(error),passed=False)
    save()
    raise
