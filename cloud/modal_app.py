"""Run the speed benchmark on a Modal cloud GPU, recording the host machine."""

import modal

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
    """Describe the machine this run landed on."""
    import os
    import subprocess
    import time
    from importlib.metadata import version

    import torch

    # CPU model name, e.g. "AMD EPYC 7R13 Processor"
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    cpu_model = next(
        (
            line.split(":", 1)[1].strip()
            for line in lscpu.splitlines()
            if line.startswith("Model name")
        ),
        "unknown",
    )

    # How many CPU cores this container may use
    cpu_cores = len(os.sched_getaffinity(0))

    # A tiny pure-Python speed test: lower = faster CPU.
    # Python overhead is what limits SmolVLA, so this is a fair "waiter speed" check.
    start = time.perf_counter()
    total = 0
    for i in range(5_000_000):
        total += i
    cpu_probe_ms = (time.perf_counter() - start) * 1000

    return {
        "gpu": torch.cuda.get_device_name(),
        "cpu_model": cpu_model,
        "cpu_cores": cpu_cores,
        "cpu_probe_ms": round(cpu_probe_ms, 1),
        "torch": version("torch"),
        "lerobot": version("lerobot"),
        "torchao": version("torchao"),
    }


@app.function(
    gpu="L40S",
    cpu=8.0,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def run_speed(
    model: str, techniques: list[str], n: int = 200, warmup: int = 20
) -> list[dict]:
    from benchmarks.models import get_model
    from benchmarks.policy import free_memory, load_policy
    from benchmarks.speed import measure_speed
    from benchmarks.techniques import get_technique

    host = host_info()
    rows = []
    for name in techniques:
        row = {"model": model, "technique": name, **host}
        try:
            bundle = load_policy(get_model(model), get_technique(name))
            row.update(measure_speed(bundle, n=n, warmup=warmup))
            del bundle
        except Exception as e:  # record the failure, keep going
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        free_memory()
    hf_cache.commit()
    return rows


@app.local_entrypoint()
def main(model: str = "smolvla", techniques: str = "native", gpu: str = "L40S"):
    rows = run_speed.with_options(gpu=gpu).remote(model, techniques.split(","))

    h = rows[0]
    print(
        f"\nGPU: {h['gpu']}   CPU: {h['cpu_model']}   cores: {h['cpu_cores']}   "
        f"cpu_probe: {h['cpu_probe_ms']} ms"
    )
    print(f"torch {h['torch']} | lerobot {h['lerobot']} | torchao {h['torchao']}\n")

    for r in rows:
        if "error" in r:
            print(f"{r['technique']:24s} FAILED: {r['error'][:100]}")
            continue
        print(
            f"{r['technique']:24s} p50={r['p50_ms']:7.1f} ms  p99={r['p99_ms']:7.1f} ms  "
            f"hz={r['hz']:6.1f}  first={r['first_call_ms']:8.0f} ms  "
            f"mem={r.get('peak_mem_gb', 0):.2f} GB"
        )
