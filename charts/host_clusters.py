"""Section 9: same GPU, different machine. One dot per machine (run_id), grouped by driver."""

import statistics
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from _style import FONT_SIZE, clean_axes, cloud_name, load_rows, missing, render_all

MODEL = "smolvla"
MIN_PER_GROUP = 3
EXPERIMENT = "machines-2026-11-20"  # the 14-machine experiment in section 9

# One point per run_id: each run_id is one container on one machine.
machines: dict[str, dict] = {}
for r in load_rows(MODEL):
    if r["technique"] == "native" and r.get("experiment") == EXPERIMENT:
        machines.setdefault(r["run_id"], r)

groups: dict[str, list[dict]] = {}
for r in machines.values():
    groups.setdefault(r["gpu_driver"], []).append(r)
counts = {d: len(rs) for d, rs in groups.items()}
if len(groups) < 2 or min(counts.values()) < MIN_PER_GROUP:
    missing(
        f"host_clusters: need >= {MIN_PER_GROUP} machines in each of 2+ driver groups, "
        f"have {counts or 'none'}",
        "modal run cloud/modal_app.py --techniques native --repeats 8",
        "each repeat can land on a different machine; rerun until both groups have 3+",
    )

drivers = sorted(groups, key=lambda d: statistics.median(r["p50_ms"] for r in groups[d]))
medians = {d: statistics.median(r["p50_ms"] for r in groups[d]) for d in drivers}
ratio = max(medians.values()) / min(medians.values())
gpu = Counter(r["gpu"] for r in machines.values()).most_common(1)[0][0]


def label(d: str) -> str:
    cloud = Counter(cloud_name(r) for r in groups[d]).most_common(1)[0][0]
    return f"driver {d}\n({cloud})"


def draw(c):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    rng = np.random.default_rng(0)  # fixed jitter, so reruns draw identical charts
    colors = [c["baseline"], c["accent"], c["muted"]]
    for i, d in enumerate(drivers):
        ys = [r["p50_ms"] for r in groups[d]]
        xs = i + rng.uniform(-0.12, 0.12, len(ys))
        ax.scatter(xs, ys, s=70, color=colors[i % len(colors)], edgecolors="none", alpha=0.9, zorder=3)
        m = medians[d]
        ax.plot([i - 0.28, i + 0.28], [m, m], color=c["secondary"], lw=2, zorder=2)
        ax.text(i + 0.32, m, f"median {m:.0f} ms", va="center", fontsize=FONT_SIZE,
                color=c["primary"])
    ax.set_xticks(range(len(drivers)), [label(d) for d in drivers])
    ax.set_xlim(-0.6, len(drivers) - 0.1)
    ax.set_ylim(0, max(r["p50_ms"] for r in machines.values()) * 1.15)
    ax.set_ylabel("Time per action, p50 (ms)", color=c["secondary"])
    ax.yaxis.grid(True, color=c["grid"], lw=1)
    ax.set_axisbelow(True)
    clean_axes(ax, c, left=False)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0)
    return fig


per_group = ", ".join(f"{counts[d]} on driver {d}" for d in drivers)
render_all(
    "host_clusters",
    draw,
    title=f"Same GPU, different machine: {ratio:.2f}× apart",
    foot=f"{len(machines)} machines ({per_group}), all {gpu}, {MODEL} native. "
         "GPU matmul, Python and launch probes similar across groups.",
)

print(f"host_clusters: {ratio:.2f}× between group medians")
print(f"  {'driver':12s} {'n':>3s} {'min':>6s} {'median':>7s} {'max':>6s}   "
      f"{'matmul ms':>9s} {'python ms':>9s} {'launch us':>9s}")
for d in drivers:
    rs = groups[d]
    p = [r["p50_ms"] for r in rs]
    matmul, python, launch = (statistics.median(r[k] for r in rs)
                              for k in ("gpu_matmul_ms", "cpu_probe_ms", "gpu_launch_us"))
    print(f"  {d:12s} {len(rs):3d} {min(p):6.0f} {statistics.median(p):7.0f} {max(p):6.0f}   "
          f"{matmul:9.2f} {python:9.1f} {launch:9.1f}")
