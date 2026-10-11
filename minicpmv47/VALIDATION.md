# Package validation: 2026-10-11

The corrected Cix P1 image is `flagos/minicpm-v47-cix-p1:0.3.1-20261011`,
ID `sha256:72d655d6bf308074aa53374ae32e7f1596d0493b81c1960df0209994bc54816b`.
Its runtime recipe commit is `0f3a6a35705ed26a22e8e92815f654cf64c8dbd1`.
The Mac archive and its earlier measurements are unchanged.

## Whole-model measurements

Single-instance medians, tok/s. Warmup is retained in reports but excluded
from these medians; every subsequent request is included.

| Platform | Model | Measured requests | PP512 | TG128 | Total |
|---|---|---:|---:|---:|---:|
| M5 Pro native, original artifact | 35B W4A8 G32 | 3 | 452.30 | 47.00 | 165.81 |
| M5 Pro native, original artifact | 1B W8A8 | 3 | 1386.81 | 114.08 | 429.94 |
| Cix P1 corrected Docker | 35B W4A8 G32 | 12 | 110.84 | 8.647 | 33.176 |
| Cix P1 corrected Docker, schedutil | 1B W8A8 | 5 | 632.88 | 21.383 | 94.855 |
| Cix P1 corrected Docker, explicit host frequency floor | 1B W8A8 | 12 | 649.39 | 21.404 | 95.062 |

P1 ranges: 35B PP95.90–113.16/TG8.510–8.681/total31.59–33.28;
1B PP391.91–651.92/TG21.316–21.418/total88.33–95.30. Peak process RSS,
including loading and image cases, was 21.61 GiB and 6.87 GiB respectively.
The first measured 35B request was slower and remains included. The 1B PP
variation remains observable; the matched host also varies.

TG128 measures 127 intervals after the first token. Total is computed for
each complete request as 640 / (prefill time + those 127 decode intervals).
Independent PP/TG best results are not combined. Both P1 profiles use the
original schedutil governors, eight prefill threads, and respectively six/four
decode threads. Neither profile writes host frequency settings. The last row separately uses
the explicit host frequency controller; its1B PP range is422.36–656.54 and
TG range21.328–21.496. This policy did not improve35B decode and is not
recommended as a shared default. A matched single-instance host1B control measured PP646.62/TG20.990/total93.450
in12 requests, so the corrected container measured +0.43%/+1.97%/+1.73% in
this control. Host PP still ranged426.78–651.49.

The corrected P1 measurements use retained warm heap state. The original
0.3.0 three-request Docker records were 35B 99.30/7.46/28.82 and
1B 309.12/18.76/75.37 with explicit cleanup before every benchmark request.
Those records are preserved, but the heap protocols must not be silently
mixed when estimating speedups. Matched host/container controls and failed
candidates are described in [P1_REGRESSION.md](P1_REGRESSION.md).

No universal 0.5% performance-equivalence claim is made. The Mac artifact's
35B TG47.00 remains below the historical 47.99; this P1 correction does not
requalify or modify the Mac artifact.

## Functional qualification

The corrected P1 image passed package-only imports, native/CPU TLE/ordinary
Triton checks using a fresh JIT cache, text/image/OCR cases for both checkpoints,
and the packaged 1B OpenAI chat HTTP server. All benchmark and functional-case
token IDs matched the original native references. Arithmetic libraries,
compiler and component source commits are unchanged by the correction.

macOS previously passed those functional checks plus extraction to a new
prefix, Mach-O dependency and symlink audits, and fresh JIT execution there.
The original P1 offline archive previously passed docker load/import checks.

## Offline artifacts

Weights and DSpark are excluded. Original artifacts are preserved:

- `flagos-minicpmv47-macos-arm64-20261011.tar.gz`, SHA256
  `cb533142f671722b2048f4912ee2eff68ba6eecc5184eb909c8da990606bf43a`.
- `flagos-minicpmv47-cix-p1-20261011.docker.tar.gz` (0.3.0), SHA256
  `936d947d0174d4cf8fd356592f2078b5158fc75f9cd1c3c4be5f73619906d499`.

The corrected 0.3.1 candidate archive passed export/reload verification:
SHA256 `37a126ed09b2e77c5a226131f858f74da01a585b2fc831c8fef6c09300ca1a66`.
The original release manifest is preserved as a separate0.3.0 record. The
replacement is qualified for the documented single-instance settings, with
no universal0.5% or tail-latency guarantee: a twenty-pair schedutil control measured container1B
PP448.54/TG20.855/total88.518 versus hostPP636.80/TG20.897/total92.958.
Later diagnostic and fixed-performance controls shifted the prefill gap toward
the host; details and limits are recorded in [P1_REGRESSION.md](P1_REGRESSION.md).
The subsequent matched single-instance control shows no container regression
in the qualified1B operating condition; the co-resident results are retained
as a separate limitation. The explicit host controller is delivered beside
the Docker archive and is not an implicit container privilege or governor
change. Shared prefill latency tails require separate investigation.
Raw reports and the release manifest remain in the local distribution's
artifacts directory; investigation reports are also retained under
`audit/p1-regression-20261011`.
