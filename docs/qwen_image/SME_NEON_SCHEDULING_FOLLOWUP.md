# SME2 / NEON scheduling and tile follow-up

2026-09-21, M5 Pro 18-core CPU, Qwen-Image-2.1-0920 W8A8 same112.
This follows [the fused MLP review](SME_NEON_FUSED_REVIEW.md).
Production inference still uses native SME2.

Combined runtime and FlagGems ARM regression: **412 tests passed** after these
changes (`wavefront-layout-0921-all-tests.log`). Repository timing evidence is
in `benchmarks/qwen-image-scheduling-0921.json`.

## Hypotheses and implementation

The row pipeline originally let a worker leave after all NEON tasks had been
assigned. Some producers can still be running then, and the remaining workers
must also drain their BF16 down-projection tasks. The experimental `drain_tail`
option retains the worker team until every down task is assigned, yielding when
it cannot obtain work. The parallel region still joins all outstanding work
before returning. Release/acquire publication of completed row panels is
unchanged. This option does not imply that retaining more workers is faster.

A separate tile sweep changes NEON output-column task size and SME2 down-column
task size. Full dot-product reductions and BF16 rounding points remain unchanged.
The sweep measures the entire MLP, including A8 quantization/repacking, both W8
branches, SwiGLU, and BF16 down projection.

## Worker and tail sweep

Real block-2 / step-2 captured input, 1024×1024 image-token geometry, six
ABBA/BAAB groups per policy. Pure SME2 uses the production 12-worker MLP.
Wavefront tile is M32 / NEON N128 / down N256.

| Mixed workers | Drain tail | Pure SME2 (ms) | Mixed (ms) | Latency change |
|---:|:---:|---:|---:|---:|
| 6 | no | 264.775 | 329.819 | +24.57% |
| 6 | yes | 263.029 | 342.711 | +30.29% |
| 9 | no | 264.781 | 269.427 | +1.75% |
| 9 | yes | 265.844 | 278.987 | +4.94% |
| 12 | no | 276.151 | 224.085 | −18.85% |
| 12 | yes | 273.382 | 229.508 | −16.05% |
| 18 | no | 275.448 | 198.551 | −27.92% |
| 18 | yes | 278.586 | 191.116 | −31.40% |

Each policy is alternated against the pure control. Policies themselves are
tested sequentially; differences between two mixed policies are not direct
paired estimates. All measured BF16 outputs matched the control exactly.
The 6/9-worker variants do not warrant a full-model experiment. Tail draining
is opt-in and has not demonstrated a sustained model gain.

## Tile screening

18 workers, M32, tail draining disabled; four ABBA/BAAB groups per policy.

| NEON N | Down N | Pure SME2 (ms) | Mixed (ms) | Latency change |
|---:|---:|---:|---:|---:|
| 64 | 128 | 268.831 | 164.143 | −38.94% |
| 64 | 512 | 271.381 | 175.030 | −35.50% |
| 64 | 1024 | 268.608 | 193.340 | −28.02% |
| 256 | 128 | 272.642 | 204.187 | −25.11% |
| 256 | 512 | 276.856 | 213.310 | −22.95% |
| 256 | 1024 | 278.532 | 224.350 | −19.45% |
| 512 | 128 | 280.376 | 223.924 | −20.13% |
| 512 | 512 | 283.997 | 225.711 | −20.52% |
| 512 | 1024 | 284.633 | 235.336 | −17.32% |

All outputs were bitwise equal. The smaller task sizes are a screening winner,
not evidence of a 38.94% image speedup. The selected M32/N64/down-N128 policy
was evaluated in six alternating full-Transformer groups.

## Sustained model result: rejected for default inference

24 measured forwards after four warmup forwards, same fixed teacher-forced
step-2 input from the 40-step schedule, prefix cache reused read-only. All
predictions matched the pure control bit for bit. The mixed path ran all 28
eligible W8 MLPs, with tail draining disabled.

| Group | Pure SME2 (s) | Mixed (s) | Latency change |
|---:|---:|---:|---:|
| 0 | 18.545 | 17.234 | −7.07% |
| 1 | 18.739 | 17.579 | −6.19% |
| 2 | 18.605 | 18.552 | −0.28% |
| 3 | 18.485 | 18.804 | +1.72% |
| 4 | 18.632 | 18.990 | +1.92% |
| 5 | 18.729 | 19.102 | +1.99% |

Overall mean: 18.623 → 18.377 s (−1.32%). This mean hides the persistent
late-run regression. No full 40-step image was launched for this policy, and no
new image latency is claimed. Native SME2 remains the production default.

### Where the saving went

Final group averages, measured with the same instrumented model:

| Category | Pure SME2 (s) | Mixed (s) | Change (s) |
|---|---:|---:|---:|
| Fused W8 MLP plus BF16 down projections | 8.478 | 6.626 | −1.852 |
| Other measured BF16 operators | 4.368 | 5.208 | +0.839 |
| W8 Q/K projections | 1.295 | 1.731 | +0.436 |
| Remainder of forward | 4.588 | 5.537 | +0.948 |
| Total | 18.729 | 19.102 | +0.372 |

The remainder is calculated by subtracting measured operator times from total
forward time. It includes attention, elementwise work and framework overhead;
it is not a measurement of any one of those categories.

Thus the MLP saving remains present even in the final group, but other work
more than offsets it. This is broader than a slow down-projection kernel.

### Telemetry association

Adjacent one-second samples wholly inside a timed forward were selected and
time-weighted. Final group: 35 samples per path.

| Raw macmon metric | Pure SME2 | Mixed |
|---|---:|---:|
| `ecpu_freq_mhz` | 4369.6 | 3818.6 |
| `pcpu_freq_mhz` | 3744.9 | 2712.3 |
| CPU active ratio | 0.861 | 0.922 |
| CPU power (W) | 34.89 | 31.92 |
| Average CPU temperature (°C) | 80.16 | 77.76 |

The raw group labels are retained rather than assuming macmon's E/P names
describe this chip's core types. CPU frequency is not SME clock frequency.
These measurements show that the mixed path has higher active time but lower
reported CPU frequencies in the late run. They do not establish whether
instruction-dependent power management, scheduling, cache behavior or another
mechanism causes the slowdown. Lower temperature alone does not establish or
exclude a thermal/power constraint.

Further experiments should isolate that sustained execution effect and control
NEON work distribution across operators. Repeating only short isolated GEMMs
or promoting a candidate based on its first model group is insufficient.

## Evidence and commands

Local evidence directory:
`/Users/kevin/qwen-image-2.1-repos/validation/`.

- `wavefront-tail-0921-tests.log`: 32 operator tests passed.
- `wavefront-tail-0921-adapter-tests.log`: 13 adapter tests passed.
- `wavefront-tail-0921-mlp.json`: worker/tail sweep.
- `wavefront-layout-0921-mlp.json`: tile sweep.
- `wavefront-layout-0921-model.json`: full-Transformer measurements.
- `wavefront-layout-0921-telemetry.jsonl`: rootless CPU telemetry.
- `wavefront-layout-0921-telemetry-summary.json`: contained-sample summary,
  reproduced with `scripts/summarize_wavefront_telemetry.py`.
- `wavefront-layout-0921-provenance.json` and source patches: exact development
  source state and native extension hash at benchmark start.

Use the prepared runtime environment and run CPU-heavy experiments serially:

```bash
python scripts/bench_wavefront_mlp.py \
  --capture /path/to/block2-step2-mlp.pt \
  --workers 6 9 12 18 --row-tiles 32 --compare-drain --groups 6 \
  --output worker-tail.json

python scripts/bench_wavefront_mlp.py \
  --capture /path/to/block2-step2-mlp.pt \
  --workers 18 --row-tiles 32 --column-tiles 64 256 512 \
  --down-tiles 128 512 1024 --groups 4 --output tiles.json

python scripts/bench_fused_hybrid_model.py \
  --model /path/to/Qwen-Image-2.1-0920-W8A8-PerChannel-112 \
  --wavefront --workers 18 --wavefront-tile 32 64 128 --groups 6 \
  --output full-transformer.json
```
