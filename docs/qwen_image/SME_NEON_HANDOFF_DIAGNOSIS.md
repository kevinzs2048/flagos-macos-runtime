# Why an isolated MLP win did not become a model win

2026-09-21. M5 Pro CPU, Qwen-Image-2.1-0920 same112 W8A8.

**Status: the performance interaction is reproduced with fixed independent
probe inputs. Its exact hardware/runtime cause is not yet established.**
The earlier worker/tile sweeps measured candidates; they did not close that
root-cause investigation. No inference default changes in this diagnostic.

## Full-model time accounting

Final group of `wavefront-park-0921-model.json`, comparing native to the mixed
15-NEON-worker / requested-5-µs-idle policy:

| Category | Native (s) | Mixed (s) | Change (s) |
|---|---:|---:|---:|
| W8 MLP and BF16 down projections | 8.987 | 6.576 | −2.411 |
| Other measured BF16 operators | 4.617 | 5.499 | +0.882 |
| W8 Q/K | 1.350 | 1.847 | +0.497 |
| Remainder of forward | 4.777 | 5.799 | +1.022 |
| Total | 19.732 | 19.722 | −0.010 |

The remainder includes uninstrumented operators and framework work; it is not
an attention-only measurement. The MLP saving persists inside the model. Other
work offsets it. This explains the arithmetic but not the underlying cause.

## Independent probe experiment

After native or mixed MLP conditioning, execute the same prepacked BF16 GEMM
twice. Probe input and weight objects, shape (4096×4096×4096), dtype, tile size
and 18-thread setting are identical. The probe does **not** consume the MLP
output. Its weight is a fixed slice of captured BF16 down-projection weights,
so this is a diagnostic proxy, not a claim about a specific Q/V layer.

Native conditioning uses the production 12-worker fused MLP. Mixed conditioning
uses 18 workers, 15 NEON-eligible slots, M32/N64/down-N64 and a 5-µs idle request.
Orders alternate ABBA/BAAB. Probe output allocation remains inside measured
GEMM time; fixed prepacked inputs do not eliminate all allocator/cache effects.

### Matched requested conditioning duration

Three seconds per conditioning phase; the faster implementation can complete
more MLP calls. Eight groups per rest interval, 16 samples per path.

| Rest before probe | Probe | After native (ms) | After mixed (ms) | Change |
|---:|---:|---:|---:|---:|
| 0 ms | first | 44.761 | 49.875 | +11.42% |
| 0 ms | second | 44.682 | 48.098 | +7.64% |
| 20 ms | first | 45.398 | 56.035 | +23.43% |
| 20 ms | second | 45.310 | 54.754 | +20.84% |

Rest settings run sequentially; do not use their absolute difference as a
clean causal estimate of adding 20 ms. The rest did not consistently remove
the mixed/native gap.

### Matched computation count

Exactly eight MLP calls on each path, with the same input and weights.
Elapsed conditioning time differs. This removes the unequal-call-count
limitation of the duration experiment.

| Rest before probe | Probe | After native (ms) | After mixed (ms) | Change |
|---:|---:|---:|---:|---:|
| 0 ms | first | 43.927 | 58.082 | +32.22% |
| 0 ms | second | 43.942 | 55.881 | +27.17% |
| 20 ms | first | 45.340 | 56.557 | +24.74% |
| 20 ms | second | 45.404 | 55.163 | +21.49% |

All probe and conditioning outputs matched the reference bit for bit. The
effect does not require altered model activations, tensor strides or extra
quantization/dequantization on the probe. Repeating the probe does not
immediately eliminate it. The size of the effect varies with run history.

## Task tracing and backend cross-check

The existing diagnostic KAI FP32-output implementation records each tile's
start/end Mach time and CPU number. It uses the same packing and tile order,
but differs from the production BF16 register epilogue. Therefore its absolute
timings must not be directly compared with the production probe from another
run.

After native/mixed conditioning, first probe averages were 42.987/44.610 ms.
The interval from first tile start to last tile end was 42.958/44.575 ms.
Only about 0.03 ms was outside that interval. Median tile intervals also grew.
This places the extra time within parallel task execution rather than a large
Python/entry/startup cost. Tile intervals can include preemption and resource
waiting; they are not exclusive hardware compute time. CPU-number changes
were present on both paths and did not identify a unique migration mechanism.

Because the traced implementation appeared less affected, a separate **same
process** test compared both untraced probe backends. Four conditions (native
or mixed conditioning × register or standard output) used rotating mirrored
orders, four groups, eight samples per condition. Standard output includes KAI
FP32 GEMM **and** its BF16 conversion in the measured time.

| Probe backend | Probe | After native (ms) | After mixed (ms) | Change |
|---|---:|---:|---:|---:|
| Direct BF16 register epilogue | first | 44.722 | 47.557 | +6.34% |
| Direct BF16 register epilogue | second | 44.729 | 46.020 | +2.89% |
| Standard KAI + BF16 conversion | first | 44.998 | 48.069 | +6.83% |
| Standard KAI + BF16 conversion | second | 44.863 | 46.532 | +3.72% |

Both are affected. This does not support blaming an error unique to the custom
BF16 epilogue. The smaller effect in this run also demonstrates why results
from separate runs cannot establish which backend is responsible.

## What remains unresolved

The intervention changes the preceding computation and the following fixed
GEMM slows down. This establishes a reproducible cross-operator interaction
under these conditions. It does **not** distinguish frequency response, SME
resource contention, scheduling/preemption or cache effects. Earlier full-model
telemetry showed lower CPU frequencies on the mixed path; that association is
not a direct SME clock measurement or proof of the sole cause.

Further diagnosis should correlate per-core state and task timing under these
controlled inputs. More short standalone-GEMM sweeps cannot resolve this causal
gap. A root-cause fix and stable full-image speedup have not been demonstrated.

## Reproduce and inspect

Use `scripts/diagnose_mlp_handoff.py` in the prepared runtime environment. Core
inference code is unchanged. Each experiment asserts numerical equality.

```bash
python scripts/diagnose_mlp_handoff.py --capture /path/to/block2-step2-mlp.pt \
  --condition-seconds 3 --groups 8 --rest-ms 0 20 --output duration.json
python scripts/diagnose_mlp_handoff.py --capture /path/to/block2-step2-mlp.pt \
  --condition-iterations 8 --groups 8 --rest-ms 0 20 --output equal-work.json
python scripts/diagnose_mlp_handoff.py --capture /path/to/block2-step2-mlp.pt \
  --condition-iterations 8 --groups 4 --rest-ms 0 --trace-probe --output trace.json
python scripts/diagnose_mlp_handoff.py --capture /path/to/block2-step2-mlp.pt \
  --condition-iterations 8 --groups 4 --rest-ms 0 --compare-backends --output backends.json
```

Repository evidence: `benchmarks/qwen-image-handoff-0921.json`.
Local raw files under `/Users/kevin/qwen-image-2.1-repos/validation/`:
`handoff-0921.json`, `handoff-workeq-0921.json`, `handoff-trace-0921.json`, and
`handoff-backends-0921.json`, with corresponding logs. Mach timebase measured
on this host: numerator 125, denominator 3 (raw ticks are not nanoseconds).
