"""Run the speed benchmark on a Modal cloud GPU, and record the host machine."""

import datetime
import json
import os
import pathlib
import uuid

import modal

GPU = os.environ.get("GPU", "L40S")  # set GPU=L4 on the command line to change it

LEROBOT = (
    "lerobot[smolvla,pi] @ git+https://github.com/huggingface/lerobot.git"
    "@b00571809f7cab69abeee19744d8dd3c18977aa2"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(LEROBOT, "torchao")
    .env({"TRITON_DEFAULT_FP_FUSION": "1"})
    .add_local_python_source("benchmarks")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("onset-speed", image=image)


def host_info() -> dict:
    """Describe the machine this run landed on, with three quick speed probes."""
    import subprocess
    import time
    from importlib.metadata import version

    import torch

    # CPU model name. lscpu shows nothing in Modal's sandbox, so read /proc/cpuinfo.
    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    # GPU driver version.
    try:
        driver = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        driver = "unknown"

    # Probe 1: plain Python speed.
    start = time.perf_counter()
    total = 0
    for i in range(5_000_000):
        total += i
    cpu_probe_ms = (time.perf_counter() - start) * 1000

    # Probe 2: GPU hand-off cost. 2,000 tiny GPU operations, so almost all the
    # time is the CPU launching work on the GPU, not the GPU computing.
    x = torch.ones(1, device="cuda")
    for _ in range(200):
        x = x + 1
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(2000):
        x = x + 1
    torch.cuda.synchronize()
    gpu_launch_us = (time.perf_counter() - start) / 2000 * 1e6

    # Probe 3: raw GPU compute. Big matrix multiplies; should match on every L40S.
    a = torch.randn(4096, 4096, device="cuda")
    for _ in range(3):
        a @ a
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    gpu_matmul_ms = (time.perf_counter() - start) / 20 * 1000

    return {
        "gpu": torch.cuda.get_device_name(),
        "gpu_driver": driver,
        "cpu_model": cpu_model,
        "cpu_cores": len(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "cloud": os.environ.get("MODAL_CLOUD_PROVIDER", "unknown"),
        "region": os.environ.get("MODAL_REGION", "unknown"),
        "cpu_probe_ms": round(cpu_probe_ms, 1),
        "gpu_launch_us": round(gpu_launch_us, 1),
        "gpu_matmul_ms": round(gpu_matmul_ms, 2),
        "torch": version("torch"),
        "lerobot": version("lerobot"),
        "torchao": version("torchao"),
    }


TRACE_DIR = "/root/.cache/huggingface/traces"  # on the hf-cache volume, so it persists


def profile_actions(bundle: dict, run_id: str, actions: int = 3) -> dict:
    """Profile a few actions with torch.profiler and save a Chrome trace + summary.

    Each action is wrapped in record_function("action") and synchronized inside the
    range, so charts/kernel_timeline.py can find exactly one action's GPU kernels.
    Call after measure_speed, so the model is already warm.
    """
    import time

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    from benchmarks.policy import act, make_observation

    obs = make_observation(bundle)
    walls = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(actions):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with record_function("action"):
                act(bundle, obs)
                torch.cuda.synchronize()
            walls.append((time.perf_counter() - start) * 1000)

    os.makedirs(TRACE_DIR, exist_ok=True)
    trace_path = f"{TRACE_DIR}/{run_id}_native.json"
    prof.export_chrome_trace(trace_path)

    # Summarise from the exported trace: kernel count and GPU busy time (union of
    # kernel intervals, since kernels on different streams can overlap).
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    kernels = sorted(
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("cat") == "kernel" and "dur" in e
    )
    busy_us, end = 0.0, float("-inf")
    for s, e in kernels:
        if e > end:
            busy_us += e - max(s, end)
            end = e
    summary = {
        "profile_actions": actions,
        "kernels_per_action": len(kernels) / actions,
        "gpu_busy_ms_per_action": busy_us / 1000 / actions,
        "wall_ms_per_action": sum(walls) / actions,
        "trace": f"traces/{run_id}_native.json",
    }
    with open(f"{TRACE_DIR}/{run_id}_native_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


@app.function(
    gpu=GPU,
    cpu=8.0,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200,
)
def run_speed(
    model: str,
    techniques: list[str],
    n: int = 200,
    warmup: int = 20,
    profile: bool = False,
) -> list[dict]:
    from benchmarks.models import get_model
    from benchmarks.policy import free_memory, load_policy
    from benchmarks.speed import measure_breakdown, measure_speed
    from benchmarks.techniques import get_technique

    host = host_info()
    host["run_id"] = uuid.uuid4().hex[
        :8
    ]  # one ID per container: rows that share it are comparable
    host["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows = []
    for name in techniques:
        row = {"model": model, "technique": name, **host}
        try:
            bundle = load_policy(get_model(model), get_technique(name))
            row.update(measure_speed(bundle, n=n, warmup=warmup))
            row.update(measure_breakdown(bundle))
            if profile and name == "native":
                try:  # a profiler failure shouldn't throw away the timing above
                    row.update(profile_actions(bundle, host["run_id"]))
                except Exception as e:
                    row["profile_error"] = f"{type(e).__name__}: {e}"
            del bundle
        except Exception as e:  # record the failure, then continue
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        free_memory()
    hf_cache.commit()
    return rows


@app.local_entrypoint()
def main(
    model: str = "smolvla",
    techniques: str = "native",
    repeats: int = 1,
    profile: bool = False,
):
    # Start all runs at once. Each one can land on a different machine.
    calls = [
        run_speed.spawn(model, techniques.split(","), profile=profile)
        for _ in range(repeats)
    ]

    for i, call in enumerate(calls, 1):
        rows = call.get()
        out = pathlib.Path("results/speed")
        out.mkdir(parents=True, exist_ok=True)
        path = (
            out
            / f"{rows[0]['timestamp'][:19].replace(':', '-')}_{rows[0]['run_id']}.jsonl"
        )
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"saved {path}")
        h = rows[0]
        print(f"\n--- machine {i} of {repeats} ---")
        print(
            f"GPU: {h['gpu']} (driver {h['gpu_driver']})   cloud: {h['cloud']} / {h['region']}"
        )
        print(
            f"CPU: {h['cpu_model']}   cores: {h['cpu_cores']}   torch threads: {h['torch_threads']}"
        )
        print(
            f"probes: python={h['cpu_probe_ms']} ms   gpu_launch={h['gpu_launch_us']} us   "
            f"gpu_matmul={h['gpu_matmul_ms']} ms"
        )
        for r in rows:
            if "error" in r:
                print(f"{r['technique']:24s} FAILED: {r['error'][:100]}")
                continue
            print(
                f"{r['technique']:24s} p50={r['p50_ms']:7.1f} ms  p99={r['p99_ms']:7.1f} ms  "
                f"first={r['first_call_ms']:8.0f} ms  mem={r.get('peak_mem_gb', 0):.2f} GB  "
                f"| pre={r['pre_ms']:.0f}  model={r['model_ms']:.0f}  post={r['post_ms']:.0f} ms"
            )
            if "profile_error" in r:
                print(f"{'':24s} profile FAILED: {r['profile_error'][:100]}")
            if "trace" in r:
                print(
                    f"{'':24s} profile: {r['kernels_per_action']:.0f} kernels/action, "
                    f"GPU busy {r['gpu_busy_ms_per_action']:.1f} of "
                    f"{r['wall_ms_per_action']:.1f} ms per action"
                )
                print("download the trace with:")
                print(f"  mkdir -p results/traces && modal volume get hf-cache {r['trace']} results/traces/")
                print(
                    f"  modal volume get hf-cache {r['trace'].replace('.json', '_summary.json')} results/traces/"
                )
