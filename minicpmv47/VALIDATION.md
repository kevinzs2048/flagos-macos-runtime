# Package validation: 2026-10-11

Both platforms passed native/CPU TLE/ordinary Triton operator checks, isolated
package import checks, text/image/OCR generation for both checkpoints, and a
1B OpenAI chat HTTP smoke test. Token IDs matched each platform's historical
native reference. macOS additionally passed extraction to a new prefix,
Mach-O dependency and symlink audits, and fresh JIT execution there. The P1
offline image passed `docker load` and subsequent component imports.

## Whole-model measurements

Three warmed requests per configuration; medians, tok/s:

| Platform | Model | PP512 | TG128 | Total |
|---|---|---:|---:|---:|
| M5 Pro native | 35B W4A8 G32 | 452.30 | 47.00 | 165.81 |
| M5 Pro native | 1B W8A8 | 1386.81 | 114.08 | 429.94 |
| Cix P1 Docker | 35B W4A8 G32 | 99.30 | 7.46 | 28.82 |
| Cix P1 Docker | 1B W8A8 | 309.12 | 18.76 | 75.37 |

TG128 measures 127 intervals after the first token. Total is
640 / (prefill time + those 127 decode intervals), measured per full request;
independent PP/TG best results are not combined. See README for thread profiles.

P1 1B using the same component packages on the host measured PP512 518.14,
TG128 18.88, total 82.97. The container prefill gap remains unresolved.
P1 35B TG128 remains below the historical 7.94; Mac 35B TG128 is below its
historical 47.99. These packages pass functional checks; they are not qualified
as regression-free. Three timing samples do not prove 0.5% performance equivalence.
The final P1 image uses GCC and an explicit 256 MiB CPU KV budget.

## Offline artifact checksums

Weights and DSpark are excluded. Artifact checksums:

- `flagos-minicpmv47-macos-arm64-20261011.tar.gz`
  SHA256: `cb533142f671722b2048f4912ee2eff68ba6eecc5184eb909c8da990606bf43a`
- `flagos-minicpmv47-cix-p1-20261011.docker.tar.gz`
  SHA256: `936d947d0174d4cf8fd356592f2078b5158fc75f9cd1c3c4be5f73619906d499`

Raw test reports and the release manifest remain in the local distribution artifacts directory.
