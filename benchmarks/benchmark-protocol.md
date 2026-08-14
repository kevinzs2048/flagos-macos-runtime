# G128 HTTP benchmark protocol

This is a retained maintainer benchmark protocol for the packaged standard
`vllm serve` command; it is not an end-user Runtime subcommand. The protocol
starts this Runtime's server, waits for `/health`, and uses the
OpenAI-compatible streaming `/v1/completions` endpoint. It sends a distinct
short coverage request, discards one full pp512/tg128 priming request, then
idles for 90 seconds before each of three retained batch-one pp512/tg128
samples. Sampling is greedy (`temperature=0`, seed 42, ignore EOS). Prefix
caching is not enabled. The server explicitly selects the `uni` executor and
disables periodic stats logging.

TTFT is measured from HTTP request start to the first non-empty content chunk.
Decode TPOT covers the 127 intervals from first to last generated content
chunk. Prefill throughput is 512/TTFT. The report preserves each raw sample
and the median.

Comparable runs require at least 40% memory free, one-minute load no higher
than half the logical CPU count, and no other `vllm serve` process. Candidate
prefill and decode medians must each retain at least 97% of the pre-package
G128 baseline. The coverage request must prove Q4 G128 prefill/decode, W8
lm_head, GDN prefill/decode, and CPU attention routing with zero derived
fallback counters. Model inspection independently requires 496 packed G128
checkpoint tensors; kernel coverage requires all 304 fused Runtime Linear
modules prepared by vLLM.

The discarded full-shape request faults in the production working set before
the idle window. Thus "cold" means a thermally cold, already-loaded server;
it does not mix cold JIT/page-cache effects with thermal throttling. The old
back-to-back protocol produced 59.47 tok/s after a build-heavy session. An
aligned cooled single shot from the same packaged Runtime reached 77.24 tok/s,
versus the historical cold best of 76.02 tok/s. Evidence is retained in
`qwen38-g128-runtime-cooled-verification.json`; the historical source samples
are normalized in `qwen38-g128-cold-baseline.json`.
