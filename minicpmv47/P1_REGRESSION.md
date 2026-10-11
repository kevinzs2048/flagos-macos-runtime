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

In the 1B experiments, changing the governor alone, retaining only one allocator, reducing
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

## 35B memory-reclamation control

The first rebuilt image still failed performance qualification: five warmed
35B requests measured PP81.92/TG8.04/total29.00, despite the same threads and
token IDs as the earlier PP99.18/TG8.70 candidate. Running the old diagnostic
harness in that same image measured PP81.40/TG8.10, excluding the newer
validation script as an adequate explanation.

The slow image requests accumulated roughly 604,000–649,000 minor page faults
each. A bounded mimalloc purge delay of 100 ms plus TCMalloc release rate 0
measured PP111.59/TG8.645/total33.20 in five warmed requests. Their minor-fault
deltas were 48,124, 7,921, 8,860, 11,254 and 14,671; peak RSS was 21.67 GiB.
All benchmark and text/image/OCR token IDs matched the original reference.
Explicit startup/prefill cleanup remains enabled. This is strong evidence
for memory reclamation being a material contributor, without attributing the
entire effect to either allocator individually. No arithmetic kernel changed.

The 35B profile now records that bounded delay and TCMalloc rate as defaults;
the 1B profile uses indefinite mimalloc retention for its smaller working set.
The corrected defaults were validated in the rebuilt image as reported below. Earlier
failed and favorable reports are retained, rather than overwritten.

## Qualification status

The corrected default image (ID
`sha256:72d655d6bf308074aa53374ae32e7f1596d0493b81c1960df0209994bc54816b`)
passed twelve warmed 35B requests: PP110.84/TG8.647/total33.176. Ranges were
PP95.90–113.16, TG8.510–8.681 and total31.59–33.28. The first measured request
remains included. Minor-fault deltas were 32,398 for that request, then
7,382–10,462 per request; peak RSS was 21.61 GiB. Five 1B requests measured
PP632.88/TG21.383/total94.855, with PP391.91–651.92, TG21.316–21.418 and
peak RSS6.87 GiB. The prefill variation also observed on the host remains a
measurement limitation, not a claim of zero latency variation.

Both checkpoints passed text/image/OCR and exact benchmark token comparisons,
package-only import checks, fresh-cache native/Raw/ordinary Triton tests, and
the packaged 1B OpenAI HTTP endpoint. All component arithmetic source commits
and native library hashes are unchanged. The interpreter snapshot, profile
and measurement protocol changes are explicit and separate from arithmetic.

The frozen interpreter candidate, retested after the first final image's slow
run, measured PP99.76/TG8.57/total32.08. This reinforces the need to control
allocator/initialization state rather than inferring a compiler arithmetic
regression from one timing run. The allocator combination is demonstrated to
remove the sustained PP81 behavior in the corrected image; the precise
individual contributions of the two allocator settings are not separately
claimed.

## Sequential paired controls: acceptance remains open for 1B

Twenty measured pairs under schedutil, with both models resident and generations
strictly sequential, gave container PP448.54/TG20.855/total88.518 versus host
PP636.80/TG20.897/total92.958. The container PP median was 29.56% lower and
total 4.78% lower. This fails performance acceptance despite the favorable
single-instance figures above. Paired co-residency and single-instance results
are distinct operating conditions; neither result is discarded.

A subsequent eight-pair diagnostic added light thread-switch timing and an
external CPU sampler. The slow-prefill distribution flipped toward the host:
container PP631.63 versus host552.17. Direct phase thread setters took at most
47.8 microseconds in the container and 11.1 microseconds in the host. The idle
peer consumed approximately 0–0.01 CPU-seconds during the other's prefill;
cgroup ancestors had no quota or throttling. A slow sampled container prefill
spent 1.44 wall seconds with main-thread runtime0.851 seconds and runqueue wait
0.052 seconds; the remaining interval was blocked time, consistent with the
sampled parallel barriers. These measurements exclude direct setter cost and
idle-peer busy spinning as adequate explanations, but do not establish a
particular blocked-time cause. Instrumented timings are diagnostic only.

An uninstrumented twelve-pair run with all big-core governors temporarily set
to performance gave container PP629.46/TG20.876/total92.855 versus host
PP548.69/TG20.892/total90.725. Container PP ranged440.96–635.09;
host PP still ranged411.90–647.32. All token IDs were identical. Governors were
saved and restored in a finally block. The final container request followed
the host's final request and overlapped possible host teardown, so a subsequent
control must keep both models alive until all timed requests are complete. This run does not justify describing
Docker as inherently slower, nor does it establish frequency as the sole cause:
previous fixed-performance runs and the host continue to show variation.
A following hardware-counter/per-thread diagnostic reproduced alternating
container PP349.6–366.8 after the host and PP529.7–582.1 on consecutive
container requests; host prefills stayed626.3–636.8 until the teardown-overlap
last request. Continuous cpuinfo/scaling frequency reads showed policies6/8/10
falling toward1.1 GHz during the slow container requests, while host fast
prefills stayed at2.3/2.2/2.5 GHz. Slow requests also retired more instructions
and cycles, consistent with the sampled barrier work; generic cache-event
counts include waiting/other work and are not isolated matrix-cache misses.
This continuous trace is stronger evidence than earlier sparse/CPPC feedback
snapshots: frequency is a demonstrated material contributor, but residual
fixed-performance host variation still prevents a sole-cause claim.

A process/thread utilization-clamp intervention is being tested with governors
remaining schedutil. Both models are now held resident until every measured
request finishes, avoiding peer teardown contamination. The test preserves
scheduler class and nice level, changes only the explicit utilization minimum,
and records before/after attributes. This is experimental, not yet a shipping
profile or source change.

The corrected image's offline archive was exported and reloaded successfully,
with SHA256 `37a126ed09b2e77c5a226131f858f74da01a585b2fc831c8fef6c09300ca1a66`.
It remains a candidate rather than an accepted replacement release while the
1B discrepancy is investigated. Raw reports are retained under the distribution's
`audit/p1-regression-20261011` and P1 `artifacts` directories. No universal 0.5%
performance-equivalence claim is made.

Allocator implementation references:
[Torch CPU allocator](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/c10/core/impl/alloc_cpu.cpp),
[mimalloc options](https://microsoft.github.io/mimalloc/group__options.html).
