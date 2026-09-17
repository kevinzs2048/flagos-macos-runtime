# Historical speculative-engine measurements — not deployment instructions

These records are retained for provenance only. The current W4A8 model card
and build guide describe **target-only inference**, with no BF16 draft download
and no speculative CLI configuration. These records are not an advertised
MTP serving feature or acceptance evidence for the current target-only service.

## What was actually tested

On September 16, 2026, the developer exercised an in-process vLLM engine using
one speculative token and a separate matching BF16 source at
`/Users/kevin/xingchenNew`. The W4A8 target excludes the extra draft tensors.
Two offline installation engine benchmarks and an in-process Chinese/English/
arithmetic chat test completed. **An MTP-enabled HTTP service was not independently
tested.** The removed MTP `vllm serve` example was therefore not an API-verified
deployment procedure.

| Workload, tokens/s | Installed engine run 1 | Installed engine run 2 | Historical accepted engine |
| --- | ---: | ---: | ---: |
| PP512 | 240.18 | 238.28 | 244.55 |
| TG128 pure | 47.48 | 46.95 | 47.39 |
| TG128 raw | 44.92 | 44.54 | 45.59 |
| Combined | 115.79 | 115.80 | 126.33 |

Each run is a three-round median. Target forwards were `1/66/66`, draft W4A8
MoE hits 206, and Q4-to-W8 refinement hits 410. These rates must not be
presented as non-speculative W4A8 throughput.

## Other retained engine runs and unresolved differences

| Workload, tokens/s | Earlier old-model control | Earlier new-model construction run | Later construction-path recheck |
| --- | ---: | ---: | ---: |
| PP512 | 244.78 | 255.09 | 183.19 |
| TG128 pure | 46.41 | 46.79 | 38.35 |
| TG128 raw | 44.61 | 43.89 | 34.81 |
| Combined | 118.06 | 117.88 | 94.23 |

PP255 was not stably reproduced. The first installed-versus-construction PP
comparison failed the initial 5% floor, and that failure remains preserved.
The construction directory later slowed too. Matching file hashes/profile
variables do not establish causal separation of installation-path and host-state
effects. Background processes were active but were not proven to be the sole
cause. Historical Combined126.33 was not reproduced.

The subsequently passing verification covered integrity, token agreement,
repeatability of the two installed engine runs, and historical speculative
PP/TG floors only. It did not prove target-only performance or MTP HTTP support.

## Evidence retained unchanged

- `evidence/chat_mtp1.json`
- `evidence/benchmark_mtp1.json`
- `evidence/installed_benchmark_mtp1.json`
- `evidence/installed_benchmark_mtp1_repeat.json`
- `evidence/old_control_mtp1.json`
- `evidence/construction_recheck_mtp1.json`
- `evidence/installed_initial_verification.json`
- `evidence/installed_release_verification.json`

Actual target-only benchmark evidence is `evidence/benchmark_target_only.json`:
PP512 253.10, TG128 pure 33.08, raw 32.29, Combined 97.89 tokens/s, with
forwards `1/128/128` and zero draft hits. Actual target-only HTTP evidence is
`evidence/api_target_only.json`, explicitly recording `mtp: false`.
