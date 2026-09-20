# SME2 + NEON: fused MLP and sustained inference

Date: 2026-09-21. Apple M5 Pro, 18 CPU cores. PyTorch CPU,
Qwen-Image-2.1-0920, 112 W8A8 Linear modules, remaining computation unchanged.
The production default remains native SME2. These experiments use C++ custom
operators and KleidiAI, not Triton-generated GEMM.

## What has actually improved

Both implementations reuse the original quantized codes and scales and preserve
the original BF16 rounding points. Neither changes the quantization policy.

1. **Split output columns:** SME2 and NEON each compute part of both W8 MLP
   branches. Each worker applies SwiGLU and writes the BF16 packed input for the
   following down projection. Activation quantization happens once, with a
   lossless repack for the NEON layout.
2. **Pipeline rows:** NEON computes gate/up and SwiGLU for row panels; SME2
   consumes completed panels for the BF16 down projection. A single worker team
   handles both queues. This overlaps different stages rather than splitting
   each individual GEMM between the two instruction paths.

The CPU cluster map is a local scheduling heuristic. Three CPU clusters do not
prove the presence of three independent SME execution engines. Workers may
migrate; this implementation does not enforce CPU affinity.

## Measurements

Microbenchmarks use the captured block-2 input from step 2 of the actual
1024×1024, 40-step schedule. Complete MLP latency includes the BF16 down
projection. Alternating baseline/candidate orders are used.

| Scope and candidate | Native | Candidate | Latency change |
|---|---:|---:|---:|
| Complete MLP, split columns, 18 workers | 277.897 ms | 235.882 ms | −15.12% |
| Complete MLP, row pipeline, 18 workers | 267.527 ms | 177.034 ms | −33.83% |
| Complete MLP, row pipeline, 12 workers | 263.673 ms | 201.475 ms | −23.59% |
| Full Transformer, split columns | 18.602 s | 18.008 s | −3.20% |
| Full Transformer, row pipeline, 18 workers | 18.574 s | 17.979 s | −3.20% mean; final group **+2.71%** |
| Full Transformer, row pipeline, 12 workers | 18.821 s | 19.341 s | **+2.76%** |
| Full Transformer, row pipeline, 7 of 28 MLPs | 18.681 s | 18.545 s | −0.73% mean; final group **+1.08%** |
| Full 1024×1024 image, 40 steps, split columns | 778.933 s | 790.343 s | **+1.46%** |

Full Transformer comparisons replay a fixed teacher-forced step-2 input, with
four warmup forwards and four ABBA/BAAB groups (16 measured forwards). They
measure sustained model execution but do not replace a complete denoising run.
The image comparison ran candidate then native in one resident pipeline, with
two warmup steps before each. It is one paired image experiment, not a
confidence interval. Its result does not justify enabling the candidate.

All measured predictions matched the control bit for bit. The two complete
40-step images also had identical pixels, matching the previous README image:
`f8f54660c7df4444781b54496f455d5c3a05dfb46705742f322822c0d5d6d37e`.
This demonstrates precision preservation for these runtime changes, not W8/BF16
equivalence or quality across a prompt dataset.

## Why the operator gain did not become an image gain

In the split-column full-Transformer comparison, W8 MLP time fell by 1.284 s,
but BF16 down projection rose by 0.390 s, other measured BF16 operations by
0.094 s, and Q/K by 0.086 s. The net forward saving was only 0.594 s.
The later full image did not retain that saving.

The row-pipeline experiment also lost speed over successive groups. Rootless
telemetry from the final group of the 12-worker run showed lower CPU frequencies
on the candidate, alongside lower power and temperature. This is an observed
association, not proof of overheating, a particular scheduler mechanism, or SME
clock behavior. Cache interference, scheduling and frequency effects still need
controlled isolation.

## Process and thread tuning

Earlier local measurements tested single versus three processes with shared
packed weights, independent Torch/kernel worker counts, LLVM OpenMP waiting,
QoS, and jemalloc. None demonstrated a sustained full-image improvement that
would justify replacing the current policy. Long OpenMP waits were slower;
the three-process GEMM test was slower after IPC/output costs.

On this installed LLVM libomp, setting `GOMP_SPINCOUNT` did not change the wait
policy. `OMP_PROC_BIND=close` parsed, but the runtime reported zero places, so
the experiment did not establish actual binding. Linux environment-variable
recipes must not be reported as successful Mac affinity tuning.

Further work should test concrete scheduling changes: per-operator worker
counts, less queue contention, retaining enough workers to drain the final row
panels, and limiting the fraction of MLPs that use both paths. The completed
sparse experiment enabled seven of the 28 W8 MLPs: its four groups changed
latency by −1.37%, −1.29%, −1.34%, and +1.08%. All predictions were bitwise
equal, but this candidate also failed to retain its saving. This is an
experiment, not a shipped performance policy.

## Reproduction and evidence

Combined runtime and FlagGems ARM regression: **402 tests passed**. The log is
`/Users/kevin/qwen-image-2.1-repos/validation/wavefront-0921-all-tests.log`.
Tests cover native/hybrid/wavefront output equality, odd shapes and tails,
multiple worker counts, fallback behavior, and the existing CPU operators.
Repository copies of the timing evidence are in
`benchmarks/qwen-image-wavefront-0921.json`,
`benchmarks/qwen-image-fused-hybrid-0921.json`, and
`benchmarks/qwen-image-fused-hybrid-0921-images.json`.

Runtime scripts:

- `scripts/profile_qwen_mlp.py`
- `scripts/bench_fused_hybrid_mlp.py`
- `scripts/bench_wavefront_mlp.py`
- `scripts/bench_fused_hybrid_model.py`
- `scripts/regress_fused_hybrid_images.py`
- `scripts/watch_qwen_cpu.py`

Raw local evidence is under
`/Users/kevin/qwen-image-2.1-repos/validation/`:

- `fused-hybrid-0921-profile/profile.json`
- `fused-hybrid-0921-mlp.json`, `fused-hybrid-0921-model.json`
- `fused-hybrid-0921-images/report.json` and `provenance.json`
- `wavefront-0921-mlp.json`, `wavefront-0921-model.json`
- `wavefront12-0921-model.json`, `wavefront12-0921-telemetry.jsonl`
- `wavefront-sparse-0921-model.json`

The full-image run used FlagGems `ad457540` and runtime `db26d9e`;
the image provenance records the extension hash loaded for that experiment.
Row-pipeline experiments use subsequent development changes. They must not be
attributed to the earlier committed native binary.

Keep the public inference CLI on native SME2 until a candidate passes both
precision checks and sustained full-image performance checks.
