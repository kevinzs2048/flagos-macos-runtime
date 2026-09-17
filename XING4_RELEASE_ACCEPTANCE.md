# Xing4_0 W4A8 G128 — macOS M5 Pro prerelease

Version: `0.2.0-alpha.2-xing4-0`. vLLM: **0.24.0+cpu**.
Unsigned developer alpha for the validated M5 Pro / 64 GiB profile.
Model weights are separate and are not included in the Runtime archive.

Pinned native operators and plugin were rebuilt locally; the validated
Python/PyTorch/Triton CPU/vLLM ABI bootstrap is retained. No new inference
algorithm was introduced. Recovered component source commits and inventories
are documented in `BUILDING_MAC.md` and `sources.public.lock.json`.

Local isolated installation, Runtime integrity, target-only HTTP inference,
installer safety checks and uninstall all passed. Uninstall moves only the
managed installation to macOS Trash, without changing other environments.

## Installed-package performance

M5 Pro / 64 GiB, CPU-only, batch one, MTP off. Three-round medians;
180-second cooldown before each round, September 17, 2026.

| Metric | Tokens/s |
| --- | ---: |
| PP512 | 312.48 |
| TG128 pure decode | 39.88 |
| TG128 full request | 38.97 |
| Combined PP512 + TG128 | 118.97 |

Synthetic random-token engine benchmark, not HTTP serving throughput.
Pure decode excludes a separate measured first-token baseline. Token IDs
match the reference; target forward counts are 1/128/128, with zero draft calls.

The user accepted this candidate's measured performance. PP512 remains 7.42%
below the earlier morning reference; historical performance equivalence is not
claimed. Background storage activity remained, absolute chip temperature was
not measured, and sequential control tests do not establish host-state causality.
The supplied deployment instructions use the W4A8 target only, without MTP.
