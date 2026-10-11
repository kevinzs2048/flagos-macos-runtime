# Cix P1 regression investigation, 2026-10-11

## Scope and measurement

The original 0.3.0 Docker package measured 1B W8A8 PP512 309.12, TG128
18.76, total 75.37 tok/s in three warmed requests. The adjacent host control
measured PP512 518.14, TG128 18.88, total 82.97. The 35B W4A8 G32 package
measured PP512 99.30, TG128 7.46, total 28.82, below the historical TG128 7.94.
These numbers prompted this investigation; functional success was insufficient.

Measurements use 512 prompt tokens and exactly 128 generated tokens, batch 1,
prefix caching off, temperature 0, a 256 MiB KV budget and CPUPhaseWorker.
TG128 uses the 127 intervals following the first token. Total is computed per
request as 640 / (prefill seconds + those 127 intervals). Separate best PP and
TG measurements are never combined. The image uses CPU operators, not Metal,
and the native/Raw backend choice is recorded rather than inferred.

Every completed generation is saved atomically, including generated token IDs.
Both text and image/OCR cases are exercised. Profiling experiments are kept
separate from qualification: strace, hot-call wrappers and perf postprocessing
can materially perturb timings.

## Established evidence

* No Docker CPU quota or observed cgroup throttling explains the gap. The
  compared executions load one GNU OpenMP runtime. Corresponding Triton LLVM
  arithmetic bodies matched; differing debug paths do not explain the gap.
* The Docker base and host both reported CPython 3.11.2, but their interpreter
  builds differed. A container using only the host interpreter and standard
  library initially recovered PP512 to approximately 520. Packaging that
  interpreter independently also recovered 35B TG128 to 7.95 in five samples.
  Longer 1B runs still dipped to 307–353: interpreter consistency is necessary
  for controlled packaging, not proof of the sole performance cause.
* A twelve-request host run also dipped to PP512 300–375. The original
  three-request host/container comparison overstated the consistency of the
  host baseline. All qualification must report ranges as well as medians.
* Torch tensor data uses embedded mimalloc 2.2.6, not the LD_PRELOAD TCMalloc
  allocator. A 67 MiB tensor did not appear in TCMalloc byte accounting.
  Disabling only TCMalloc release or omitting only request-level trims did not
  resolve the regression. Keeping mimalloc and TCMalloc pages removed repeated
  minor faults in warm requests but did not remove every slow prefill.
* Sampled slow prefills had no Python GC event. Time-aligned VM statistics did
  not show compaction or THP allocation failures in those requests.
* Sampled native CPU execution showed both slower W8/GDN arithmetic and more
  OpenMP barrier waiting. Under close binding, 1B decode uses CPUs 0,1,10,11,
  occupying only two of the four big-core frequency policies. Spread binding
  uses CPUs 0,10,6,8 across all four. These placements are specific to the
  tested P1 topology, not a portable numbering assumption.

## Rejected shortcuts

Changing the governor alone, retaining only one allocator, reducing
GOMP_SPINCOUNT to 1000/100000, or reporting only three favorable requests did
not provide a satisfactory fix. Reducing the spin count worsened decode;
the qualified default remains 1000000. Governor experiments save and restore
the host policy in a finally block. The container does not silently write
host frequency settings.

## Reproducible package changes

The Linux runtime includes a snapshot of the qualified CPython executable,
standard library, JIT headers, shared library and license. The launcher uses
that snapshot via relative interpreter links and PYTHONHOME. The snapshot
manifest records executable and file hashes. All model/operator imports remain
inside the package; no host source or independent adapter mount is required.

The validation harness distinguishes retained warm allocator state from
explicit per-request heap trimming (`--heap-policy trim`). It records the
interpreter, allocator/OpenMP environment, host governors, timing ranges,
minor faults and peak RSS. Startup heap cleanup is preserved.

The three component source SHAs and model weights are unchanged by these
packaging experiments. This is not a new claim that all execution is ordinary
Triton: native fused arithmetic remains C++/Neon/KAI with supported CPU TLE Raw
entrypoints, and Gemma RMSNorm uses ordinary CPU Triton.

## Matched 1B control

Twelve warmed requests each, identical component packages and token IDs,
spread binding, retained allocator pages, GOMP_SPINCOUNT 1000000 and the
original schedutil host governors:

| Execution | PP512 median | TG128 median | Total median |
|---|---:|---:|---:|
| Independently packaged CPython container | 647.25 | 21.387 | 94.675 |
| Host-qualified interpreter, same staged components | 640.66 | 21.431 | 95.144 |

Container median changes relative to that matched host are +1.03%, -0.20%
and -0.49%. This is an observed comparison, not a confidence interval or
proof that future measurements stay within 0.5%. Container PP ranged from
340.13 to 651.02; the host also showed intermittent slow prefills. All
benchmark and text/image/OCR token IDs matched the original native reference.

Forcing all big-core policies to performance with the same spread/retention
configuration did not remove the PP variation: median PP646.02, TG21.18,
total93.87. CPPC feedback during both fast and slow prefills remained near
the requested 2.6/2.3/2.2/2.5 GHz policy maxima. Frequency alone is therefore
not an adequate explanation of the remaining variation, and no governor
change is shipped as an implicit requirement.

## Qualification status

Final policy selection and image requalification are in progress. Do not
substitute favorable intermediate medians for final image results. Raw reports
are retained under the distribution's `audit/p1-regression-20261011` and P1
`artifacts` directories. No universal 0.5% performance-equivalence claim is made.

Allocator implementation references:
[Torch CPU allocator](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/c10/core/impl/alloc_cpu.cpp),
[mimalloc options](https://microsoft.github.io/mimalloc/group__options.html).
