# SME2 + NEON: 4096-row W8A8 Linear review

## Result

On September 20, 2026, the current FlagGems cooperative kernel reduced standalone
W8A8 Linear latency by about 34–35% in confirmation runs. Both FP32 intermediate
and final BF16 outputs were bitwise equal to the pure SME2 route. No model weights,
quantization policy, production default or kernel implementation changed.

| M / N / K | Pure SME2 | SME2 + NEON | Latency reduction | Speedup | Winning groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 / 12288 / 4096 | 75.028 ms | 49.272 ms | 34.33% | 1.523× | 128/128 |
| 4096 / 4096 / 4096 | 23.290 ms | 15.070 ms | 35.29% | 1.545× | 128/128 |

Each confirmation used 128 ABBA/BAAB groups (256 samples per route). Inputs and
weights were synthetic BF16 with seed 42, followed by the existing per-channel W8
quantizer. Timing includes A8 packing, layout conversion, GEMM, bias and BF16 output;
it excludes compilation and weight quantization/prepacking. This is not a bare
microkernel throughput measurement.

For Qwen-Image-2.1, the reference pipeline uses a 16× spatial reduction and plain
spatial flattening: a 1024×1024 image has 4096 image tokens. Text/prefix processing
can change actual first-step projection shapes. N=4096 and N=12288 match the model
projection widths. Synthetic shape matching does not replace real-activation or
fused-block benchmarking.

## Scheduling sweep

| N | Total cooperative workers | SME share of N | Pure SME2 ms | Mixed ms | Latency change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12288 | 12 | 0.500 | 71.054 | 53.816 | -24.26% |
| 12288 | 12 | 0.750 | 70.049 | 79.043 | +12.84% |
| 12288 | 18 | 0.250 | 69.842 | 50.242 | -28.06% |
| 12288 | 18 | 0.375 | 71.006 | 45.308 | -36.19% |
| 12288 | 18 | 0.500 | 69.163 | 44.132 | -36.19% |
| 12288 | 18 | 0.625 | 69.684 | 64.421 | -7.55% |
| 12288 | 18 | 0.750 | 68.722 | 80.226 | +16.74% |
| 4096 | 12 | 0.500 | 23.401 | 18.462 | -21.11% |
| 4096 | 12 | 0.750 | 23.418 | 25.984 | +10.96% |
| 4096 | 18 | 0.250 | 23.471 | 17.545 | -25.25% |
| 4096 | 18 | 0.375 | 23.450 | 15.654 | -33.25% |
| 4096 | 18 | 0.500 | 23.460 | 15.225 | -35.10% |
| 4096 | 18 | 0.625 | 23.412 | 21.150 | -9.66% |
| 4096 | 18 | 0.750 | 23.463 | 27.279 | +16.27% |

Each sweep candidate used 16 interleaved groups. Compare candidates to their own
paired baseline; do not combine baselines across runs. The half-N, 18-worker policy
was selected for the independent longer confirmation. Ratios above one half lost
much of the benefit; three quarters regressed.

The baseline always uses 18 threads. A 12-worker candidate is therefore not a
same-worker-count comparison. Cooperative mode uses a shared dynamic team; the
arguments 3 and 15 sum to 18, not fixed physical assignments. A cluster map and
cluster claims do not prove the number of independent hardware SME engines.

## Why this is not yet an image-generation speedup

The current production path includes paired/fused operations and 12-worker MLP
scheduling that this standalone Linear benchmark does not exercise. The standalone
gain must survive comparison with those optimized modules, not only separate
18-thread Linear calls.

Historical sustained model measurements on the earlier configuration found:
W8 1.392189 → 1.048332 s, other BF16 groups 2.357867 → 2.707993 s,
and overall forward 4.171409 → 4.240134 s (+1.65%). Those are historical data,
not measurements of the current September 20 checkpoint. They show why W8 savings
alone cannot establish whole-model savings. Lower sampled CPU frequency accompanied
the regression; its cause was not established.

## Next optimization priorities

1. Profile the current 1024/40 production route and measure the exact fraction of
   time in eligible W8, retained BF16, attention, packing and synchronization.
2. Preserve shared A8 packing and fused Q/K/MLP paths when implementing mixed
   computation. Avoid replacing fused paths with separate Linear calls.
3. Compare complete W8→BF16 sequences and real blocks under sustained alternating
   load. Tune the NEON work budget, team size, tile sizes and queue overhead together.
4. Only activate per-shape policies that win against the current production module
   and do not slow neighboring BF16 operations. Then rerun teacher-forced precision
   checks and the complete 40-step request.

Illustration, not a performance prediction: if eligible W8 takes 30% of a request,
a 35% W8 latency reduction with all other work unchanged reduces total latency by
10.5% (1.117× speedup). The eligible fraction for the current route remains to be
measured. There is no demonstrated 2× whole-image speedup from this experiment.

## Evidence and reproduction

[Full samples, source hashes and settings](../../benchmarks/qwen-image-sme-neon-4096.json).
Local logs and individual results: `~/qwen-image-2.1-repos/validation/sme-neon-4096-review/`.

The existing standalone harness was reused, but its operator provider was replaced
before execution, so native code came from the current FlagGems repository:

```python
import runpy, sme_neon
from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import ops
sme_neon.ops = ops
runpy.run_path('/Users/kevin/sme2-neon-kit/benchmark.py', run_name='__main__')
```

Use the prepared `env-upstream.sh` environment and put `~/sme2-neon-kit` on
`PYTHONPATH`. Confirmation arguments: `--m 4096 --n 4096 --k 4096 --threads 18
--sme-workers 3 --neon-workers 15 --fraction 0.5 --groups 128 --mode cooperative
--cpu-clusters ~/sme2-neon-kit/examples/m5-pro-local-cpu-clusters.json --output result.json`.
Repeat with `--n 12288`. The supplied CPU map is specific to this validation Mac.
