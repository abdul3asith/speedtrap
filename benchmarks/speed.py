"""Measure action latency correctly: first call, warm-up, synchronize, percentiles, memory."""

import statistics
import time

import torch

from .policy import act, make_observation
from .techniques import autocast_context


def sync(device: str) -> None:
    """Wait for the GPU to finish its queued work. Does nothing on CPU."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[round(p / 100 * (len(ordered) - 1))]


def timed_act(bundle: dict, obs: dict) -> float:
    """One action, timed in milliseconds, with the GPU synchronized on both ends."""
    device = bundle["device"]
    sync(device)
    start = time.perf_counter()
    act(bundle, obs)
    sync(device)
    return (time.perf_counter() - start) * 1000


def measure_speed(bundle: dict, n: int = 200, warmup: int = 20) -> dict:
    device = bundle["device"]
    obs = make_observation(bundle)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # 1. First call: includes one-time costs such as compilation. Reported, not hidden.
    first_call_ms = timed_act(bundle, obs)

    # 2. Warm-up: let caches and kernels settle. Not counted.
    for _ in range(warmup):
        act(bundle, obs)

    # 3. Timed steady-state runs.
    times_ms = [timed_act(bundle, obs) for _ in range(n)]

    p50 = percentile(times_ms, 50)
    result = {
        "first_call_ms": first_call_ms,
        "p50_ms": p50,
        "p99_ms": percentile(times_ms, 99),
        "mean_ms": statistics.mean(times_ms),
        "jitter_ms": statistics.stdev(times_ms),
        "hz": 1000 / p50,
        "n": n,
        "warmup": warmup,
    }
    if device.startswith("cuda"):
        result["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return result


def measure_breakdown(bundle: dict, n: int = 30) -> dict:
    """Split one action into its three stages, to see where the time goes."""
    device = bundle["device"]
    obs = make_observation(bundle)
    stages = {"pre_ms": [], "model_ms": [], "post_ms": []}
    with torch.no_grad(), autocast_context(bundle["technique"], device):
        for _ in range(n):
            bundle["policy"].reset()
            sync(device)
            t0 = time.perf_counter()
            batch = bundle["pre"](obs)
            sync(device)
            t1 = time.perf_counter()
            action = bundle["policy"].select_action(batch)
            sync(device)
            t2 = time.perf_counter()
            bundle["post"](action)
            sync(device)
            t3 = time.perf_counter()
            stages["pre_ms"].append((t1 - t0) * 1000)
            stages["model_ms"].append((t2 - t1) * 1000)
            stages["post_ms"].append((t3 - t2) * 1000)
    return {name: statistics.median(times) for name, times in stages.items()}
