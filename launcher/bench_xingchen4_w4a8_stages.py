"""One-engine PP512, TG128, and PP512+TG128 benchmark."""

from __future__ import annotations

import json
import os
import random
import statistics
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def _resolve_worker(llm: LLM):
    executor = llm.llm_engine.engine_core.engine_core.model_executor
    candidates = (
        executor,
        getattr(executor, "driver_worker", None),
        getattr(executor, "worker", None),
        getattr(getattr(executor, "driver_worker", None), "worker", None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "model_runner"):
            return candidate
    raise RuntimeError("cannot locate the in-process TeleChat4 CPU worker")


def main() -> None:
    rounds = int(os.getenv("TELECHAT4_STAGE_BENCH_ROUNDS", "3"))
    rng = random.Random(int(os.getenv("TELECHAT4_STAGE_BENCH_SEED", "20260820")))
    prompt_ids = [rng.randrange(1000, 20000) for _ in range(512)]
    scenarios = {
        "pp512": (TokensPrompt(prompt_token_ids=prompt_ids), 1),
        "tg128": (TokensPrompt(prompt_token_ids=prompt_ids[:16]), 128),
        "total": (TokensPrompt(prompt_token_ids=prompt_ids), 128),
    }
    llm_kwargs = dict(
        model=os.environ["MODEL_DIR"],
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        distributed_executor_backend="uni",
        max_model_len=648,
        max_num_seqs=1,
        max_num_batched_tokens=648,
        gpu_memory_utilization=float(
            os.getenv("TELECHAT4_STAGE_BENCH_MEMORY_UTILIZATION", "0.45")
        ),
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    mtp_model = os.environ.get("TELECHAT4_MTP_MODEL", "").strip()
    if mtp_model:
        mtp_tokens = int(os.environ.get("TELECHAT4_MTP_TOKENS", "1"))
        if mtp_tokens <= 0:
            raise ValueError("TELECHAT4_MTP_TOKENS must be positive")
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "model": mtp_model,
            "num_speculative_tokens": mtp_tokens,
        }
    llm = LLM(**llm_kwargs)
    params = {
        name: SamplingParams(
            temperature=0.0,
            max_tokens=decode_tokens,
            min_tokens=decode_tokens,
            ignore_eos=True,
        )
        for name, (_, decode_tokens) in scenarios.items()
    }
    warm_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        min_tokens=4,
        ignore_eos=True,
    )
    runner = _resolve_worker(llm).model_runner
    original_forward = runner._model_forward
    target_forward_calls = 0
    target_forward_elapsed_ns = 0

    def counted_forward(*args, **kwargs):
        nonlocal target_forward_calls, target_forward_elapsed_ns
        target_forward_calls += 1
        started = time.perf_counter_ns()
        try:
            return original_forward(*args, **kwargs)
        finally:
            target_forward_elapsed_ns += time.perf_counter_ns() - started

    runner._model_forward = counted_forward

    draft_elapsed_ns = 0
    if getattr(runner, "drafter", None) is not None:
        original_propose = runner.drafter.propose

        def timed_propose(*args, **kwargs):
            nonlocal draft_elapsed_ns
            started = time.perf_counter_ns()
            try:
                return original_propose(*args, **kwargs)
            finally:
                draft_elapsed_ns += time.perf_counter_ns() - started

        runner.drafter.propose = timed_propose

    target_moe_route_hits = None
    if os.getenv("TELECHAT4_STAGE_BENCH_ROUTE_COUNTERS", "0").lower() in {
        "1",
        "true",
        "on",
    }:
        from vllm_fl.patches import cpu_w4a8_moe

        target_moe_route_hits = {
            "routed_m1": 0,
            "routed_m2": 0,
            "generic": 0,
        }

        def count_route(name, function):
            def counted(*args, **kwargs):
                target_moe_route_hits[name] += 1
                return function(*args, **kwargs)

            return counted

        for route_name, function_name in (
            ("routed_m1", "_apply_kai_routed_m1"),
            ("routed_m2", "_apply_kai_routed_m2"),
            ("generic", "_apply_kai_w4a8_moe4"),
        ):
            setattr(
                cpu_w4a8_moe,
                function_name,
                count_route(route_name, getattr(cpu_w4a8_moe, function_name)),
            )

    def generate(prompt, sampling):
        calls_before = target_forward_calls
        target_ns_before = target_forward_elapsed_ns
        draft_ns_before = draft_elapsed_ns
        started = time.perf_counter_ns()
        result = llm.generate([prompt], sampling, use_tqdm=False)
        elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
        return (
            elapsed_ms,
            list(result[0].outputs[0].token_ids),
            target_forward_calls - calls_before,
            (target_forward_elapsed_ns - target_ns_before) / 1.0e6,
            (draft_elapsed_ns - draft_ns_before) / 1.0e6,
        )

    def run(name: str):
        prompt, _ = scenarios[name]
        return generate(prompt, params[name])

    # Match the original XingChen4 acceptance protocol exactly: specialize
    # the two prompt shapes with four generated tokens, then measure PP512,
    # a one-token prompt baseline, TG128, and the combined case in that order.
    # The baseline lets us report both raw request throughput and the historic
    # pure-decode metric instead of accidentally comparing the two.
    generate(scenarios["pp512"][0], warm_params)
    generate(scenarios["tg128"][0], warm_params)

    samples = {name: [] for name in scenarios}
    stage_times = {
        name: {"target_ms": [], "draft_ms": [], "other_ms": []}
        for name in scenarios
    }
    forward_counts = {name: [] for name in scenarios}
    baseline_samples = []
    baseline_forward_counts = []
    decode_samples = []
    references = {}
    token_exact = {name: True for name in scenarios}
    for _ in range(rounds):
        for name in ("pp512", "tg128", "total"):
            if name == "tg128":
                baseline_result = generate(
                    scenarios[name][0], params["pp512"]
                )
                baseline_ms, baseline_tokens, baseline_calls = baseline_result[:3]
                baseline_samples.append(baseline_ms)
                baseline_forward_counts.append(baseline_calls)
            elapsed_ms, tokens, calls, target_ms, draft_ms = run(name)
            if name == "tg128":
                if tokens[:1] != baseline_tokens:
                    raise RuntimeError(
                        "TG128 and its one-token baseline diverged"
                    )
                decode_samples.append(elapsed_ms - baseline_ms)
            samples[name].append(elapsed_ms)
            stage_times[name]["target_ms"].append(target_ms)
            stage_times[name]["draft_ms"].append(draft_ms)
            stage_times[name]["other_ms"].append(
                elapsed_ms - target_ms - draft_ms
            )
            forward_counts[name].append(calls)
            if name not in references:
                references[name] = tokens
            else:
                token_exact[name] = token_exact[name] and tokens == references[name]

    medians = {
        name: statistics.median(values) for name, values in samples.items()
    }
    decode_median_ms = statistics.median(decode_samples)
    rates = {
        "pp512_input_tok_s": 512.0 / (medians["pp512"] / 1.0e3),
        "tg128_output_tok_s": 127.0 / (decode_median_ms / 1.0e3),
        "tg128_raw_request_tok_s": 128.0 / (medians["tg128"] / 1.0e3),
        "total_tok_s": 640.0 / (medians["total"] / 1.0e3),
    }
    output = {
        "rounds": rounds,
        "timings_ms": samples,
        "stage_timings_ms": stage_times,
        "tg128_baseline_ms": baseline_samples,
        "tg128_decode_ms": decode_samples,
        "target_forwards": forward_counts,
        "target_moe_route_hits": target_moe_route_hits,
        "tg128_baseline_target_forwards": baseline_forward_counts,
        "median_ms": medians,
        "tg128_decode_median_ms": decode_median_ms,
        "rates": rates,
        "token_exact": token_exact,
        "token_ids": references,
    }
    try:
        from vllm_fl.ops import cpu_w8_greedy

        output["w8_cached_draft_greedy_hits"] = (
            cpu_w8_greedy.w8_cached_draft_greedy_hits()
        )
        output["w8_cached_refined_draft_greedy_hits"] = (
            cpu_w8_greedy.w8_cached_refined_draft_greedy_hits()
        )
    except (ImportError, AttributeError):
        pass
    try:
        from vllm_fl.spec_decode import telechat4_mtp

        output["mtp_w4a8_moe_hits"] = (
            telechat4_mtp.telechat4_mtp_w4a8_moe_hits()
        )
        output["mtp_w8_moe_hits"] = (
            telechat4_mtp.telechat4_mtp_w8_moe_hits()
        )
        output["mtp_bf16_moe_hits"] = (
            telechat4_mtp.telechat4_mtp_bf16_moe_hits()
        )
        output["mtp_bf16_moe_row_calls"] = (
            telechat4_mtp.telechat4_mtp_bf16_moe_row_calls()
        )
        output["mtp_bf16_shared_mlp_hits"] = (
            telechat4_mtp.telechat4_mtp_bf16_shared_mlp_hits()
        )
    except (ImportError, AttributeError):
        pass
    print(
        "TELECHAT4_W4A8_STAGE_BENCH_JSON=" + json.dumps(output),
        flush=True,
    )


if __name__ == "__main__":
    main()
