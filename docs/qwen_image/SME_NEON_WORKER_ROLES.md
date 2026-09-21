# Separate NEON producer eligibility from total workers

2026-09-21, M5 Pro CPU, Qwen-Image-2.1-0920 W8A8 same112.
This follows [the tile and tail experiment](SME_NEON_SCHEDULING_FOLLOWUP.md).
The public inference default remains native SME2.

Final combined runtime and FlagGems ARM regression: **472 tests passed**
(`wavefront-roles-park-0921-all-tests.log`). Repository evidence is stored in
`benchmarks/qwen-image-worker-roles-0921.json`.

## Change

The experimental row pipeline now accepts `neon_workers`. Zero retains the
previous behavior: all workers may produce NEON tiles. A positive value limits
the number of logical worker slots eligible to compute W8 gate/up and SwiGLU.
Other slots only consume ready panels with SME2 for the BF16 down projection.
Eligible producers can also consume SME2 work. Consequently this is an upper
bound on NEON concurrency, not a fixed NEON/SME partition at every instant.

All slots remain in the same ATen parallel region. Slots without available
eligible work yield in the first experiment. The follow-up `idle_us` option
requests a short sleep instead; it does not implement condition-variable
parking or CPU affinity. Actual wake-up latency is determined by the OS, not
guaranteed to equal the requested microseconds. The existing per-cluster SME claim is still a local
scheduling heuristic, not evidence of three independent hardware SME engines.
The total worker count is unchanged at 18.

The objective is to distinguish reducing NEON concurrency from reducing the
entire team, while retaining SME consumers to process completed panels.
Quantization, dot-product reduction order, BF16 rounding and model inputs are
unchanged.

## Real-input full-MLP screening

Captured block-2 / step-2 input from the 1024×1024, 40-step schedule. Production
baseline uses 12 pure-SME2 workers. Each candidate has 18 total workers,
M32 / NEON N64, with `drain_tail=False`. Six ABBA/BAAB groups per candidate,
12 timed samples per path. Policy order itself is sequential, so differences
between candidate means are screening results, not direct paired estimates.

| SME down N | NEON-eligible slots | Native (ms) | Candidate (ms) | Latency change |
|---:|---:|---:|---:|---:|
| 64 | 9 | 262.758 | 227.881 | −13.27% |
| 64 | 12 | 265.414 | 188.439 | −29.00% |
| 64 | 15 | 270.317 | 168.855 | −37.53% |
| 64 | 18 | 270.151 | 179.218 | −33.66% |
| 128 | 9 | 273.309 | 262.161 | −4.08% |
| 128 | 12 | 277.475 | 216.107 | −22.12% |
| 128 | 15 | 283.143 | 188.792 | −33.32% |
| 128 | 18 | 280.546 | 195.324 | −30.38% |

All measured BF16 outputs were bitwise equal to the control. Screening selected
12 and 15 NEON-eligible slots with down-N64 for model testing. The 12-slot
candidate explicitly trades some isolated MLP throughput for less simultaneous
NEON work; the model experiment determines whether this helps other operators.

## Model comparison method

Same resident model, fixed teacher-forced step-2 input, four warmup forwards.
The three modes are native, NEON-limit-12, and NEON-limit-15. In each group,
their order rotates and then mirrors:

```text
native, 12, 15, 15, 12, native
12, 15, native, native, 15, 12
15, native, 12, 12, native, 15
native, 12, 15, 15, 12, native
```

This gives 24 measured forwards, eight per mode. Each candidate enables all 28
W8 MLPs. Every forward checks bitwise prediction equality, exactly 112 native
W8 GEMM calls, and the expected mixed-MLP call count. Configuration changes and
weight-cache lookup occur outside the measured forward. Timings are full
Transformer measurements, not complete image-generation latency.

## First model result: yielding consumers

All 24 predictions matched exactly. Mean forward time: native 18.770 s,
NEON-limit-12 19.116 s (+1.85%), NEON-limit-15 18.248 s (−2.78%).
The late groups did not retain the saving:

| Group | Native (s) | Limit 12 (s) | Change | Limit 15 (s) | Change |
|---:|---:|---:|---:|---:|---:|
| 0 | 18.487 | 18.046 | −2.39% | 17.192 | −7.01% |
| 1 | 18.861 | 18.709 | −0.80% | 17.926 | −4.96% |
| 2 | 18.641 | 19.640 | +5.36% | 18.733 | +0.49% |
| 3 | 19.090 | 20.070 | +5.13% | 19.140 | +0.26% |

Neither policy was enabled by default or sent to a full-image comparison.
Telemetry showed approximately 97% CPU active ratio on the mixed paths versus
86% on native in the first groups. Reducing NEON eligibility did not reduce
overall CPU active time: reserved workers still repeatedly yielded while waiting.
That observation motivated testing a requested short sleep for idle workers.
It does not by itself prove those workers caused the frequency change.

## Idle-wait screening and direct comparison

The follow-up keeps M32 / NEON N64 / down N64 and 18 total workers. At 15
NEON-eligible slots, the complete-MLP screening measured:

| Requested idle wait | Native (ms) | Mixed (ms) | Latency change |
|---:|---:|---:|---:|
| 0 (yield) | 275.639 | 183.997 | −33.25% |
| 5 µs | 277.966 | 188.304 | −32.26% |
| 20 µs | 279.243 | 192.989 | −30.89% |
| 100 µs | 281.690 | 198.492 | −29.54% |

The corresponding 12-slot candidates measured 186.750, 197.286, 208.111 and
219.403 ms. All outputs remained bitwise equal. Longer requested waits increased
isolated MLP latency. The 5-µs request was selected for a full-model comparison
because it retained most of the isolated saving while allowing idle workers to
sleep. This does not establish a model-level speedup.

For that comparison, both mixed modes use 15 NEON-eligible slots. The same
rotating/mirrored three-mode schedule compares native, mixed/yield and
mixed/5-µs-request, with 24 measured forwards and four warmup forwards. This
directly compares waiting policies rather than comparing timings from separate
processes or different days.

### Direct idle-wait model result

All 24 predictions matched the control exactly. Mean native / yield / 5-µs
request: **19.410 / 19.167 / 19.131 s**. The mixed-path mean savings of 1.25%
and 1.44% were concentrated early in the run:

| Group | Native (s) | Yield (s) | Change | 5-µs request (s) | Change |
|---:|---:|---:|---:|---:|---:|
| 0 | 18.664 | 17.787 | −4.70% | 17.605 | −5.68% |
| 1 | 19.625 | 19.410 | −1.10% | 19.487 | −0.70% |
| 2 | 19.618 | 19.661 | +0.22% | 19.711 | +0.47% |
| 3 | 19.732 | 19.812 | +0.40% | 19.722 | −0.05% |

The final 5-µs sample group was effectively tied, not evidence of a meaningful
speedup. No candidate was promoted and no new full 40-step image was launched.

Final-group telemetry, 37 contained samples per mode:

| Raw macmon field | Native | Yield | 5-µs request |
|---|---:|---:|---:|
| CPU active ratio | 0.866 | 0.968 | 0.972 |
| `ecpu_freq_mhz` | 4365.9 | 3515.8 | 3522.0 |
| `pcpu_freq_mhz` | 3416.6 | 2651.2 | 2668.0 |
| CPU power (W) | 31.14 | 27.93 | 28.01 |
| Average CPU temperature (°C) | 76.16 | 74.97 | 74.88 |

The requested short sleep did not measurably lower whole-forward CPU active
ratio or restore the native path's reported CPU frequencies. This tested policy
does not support attributing the sustained regression solely to yielding idle
workers. These are whole-forward observations, not direct measurements of
individual worker idle time, SME clock frequency or a hardware bottleneck.

Across the two model experiments, all **48 predictions** matched bit for bit.
The existing native production path and quantization policy remain unchanged.

## Evidence and reproduction

Local evidence root: `/Users/kevin/qwen-image-2.1-repos/validation/`.

- `wavefront-roles-0921-tests.log`: 52 operator tests passed, including one
  producer, odd dimensions/tails, one/four workers and both tail-drain settings.
- `wavefront-roles-0921-adapter-tests.log`: adapter regression.
- `wavefront-roles-0921-mlp.json`: screening samples.
- `wavefront-roles-0921-model.json`: model samples.
- `wavefront-roles-0921-telemetry.jsonl`: CPU telemetry.
- `wavefront-roles-0921-provenance.json` and source patches: tested source state
  and native extension hash.
- `wavefront-park-0921-tests.log`: 105 targeted tests passed for role and idle
  policies, including the PyTorch adapter.
- `wavefront-park-0921-mlp.json`, `wavefront-park-0921-model.json` and telemetry:
  idle-wait screening and direct model comparison.
- `wavefront-park-0921-provenance.json` and source patches: extension/source
  state for the idle-wait experiment. The earlier role experiment used the
  previous loaded binary, recorded separately.

After activating the prepared runtime environment:

```bash
python scripts/bench_wavefront_mlp.py \
  --capture /path/to/block2-step2-mlp.pt \
  --workers 18 --row-tiles 32 --column-tiles 64 --down-tiles 64 128 \
  --neon-workers 9 12 15 18 --groups 6 --output role-screening.json

python scripts/bench_fused_hybrid_model.py \
  --model /path/to/Qwen-Image-2.1-0920-W8A8-PerChannel-112 \
  --wavefront --workers 18 --wavefront-tile 32 64 64 \
  --neon-worker-sequence 12 15 --groups 4 --output role-model.json

python scripts/bench_fused_hybrid_model.py \
  --model /path/to/Qwen-Image-2.1-0920-W8A8-PerChannel-112 \
  --wavefront --workers 18 --wavefront-tile 32 64 64 \
  --neon-workers 15 --idle-sequence 0 5 --groups 4 --output idle-model.json
```
