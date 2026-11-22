"""Section 1 chart: memory doubles, latency doesn't. Reads saved speed results."""

import matplotlib.pyplot as plt
from _style import (
    PANEL_SIZE,
    SUB_SIZE,
    VALUE_SIZE,
    clean_axes,
    footnote,
    load_rows,
    missing,
    post_run,
    render_all,
)

MODEL = "smolvla"
BASE, TEST = "native", "fp32_cast"

run = post_run(load_rows(MODEL), [BASE, TEST])
if run is None:
    missing(f"fp32_vs_bf16: no run has both {BASE} and {TEST}",
            f"modal run cloud/modal_app.py --techniques {BASE},{TEST}")
base, test = run[BASE], run[TEST]

formats = ["bf16\n(as shipped)", "fp32"]
latency_ms = [base["p50_ms"], test["p50_ms"]]
memory_gb = [base["peak_mem_gb"], test["peak_mem_gb"]]
mem_ratio = memory_gb[1] / memory_gb[0]
lat_ratio = latency_ms[1] / latency_ms[0]


def panel(ax, c, values, title, unit, fmt, ratio):
    bars = ax.bar(formats, values, width=0.5, color=[c["baseline"], c["accent"]])
    top = max(values)
    ax.set_ylim(0, top * 1.45)
    ax.set_yticks([])
    clean_axes(ax, c)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_title(title, loc="left", fontsize=PANEL_SIZE, fontweight="semibold", color=c["primary"], pad=6)
    ax.text(0, 1.0, ratio, transform=ax.transAxes, ha="left", va="top",
            fontsize=SUB_SIZE, color=c["secondary"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.03, f"{v:{fmt}} {unit}",
                ha="center", va="bottom", fontsize=VALUE_SIZE, fontweight="semibold",
                color=c["primary"])


def draw(c):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.4))
    panel(ax1, c, memory_gb, "Peak GPU memory", "GB", ".2f", f"{mem_ratio:.1f}× memory")
    panel(ax2, c, latency_ms, "Time per action (p50)", "ms", ".0f", f"{lat_ratio:.2f}× time")
    return fig


render_all(
    "fp32_vs_bf16",
    draw,
    title="Doubling the precision doubled the memory, but not the time",
    foot=footnote(base),
)
print(f"fp32_vs_bf16 from run {base['run_id']}: p50 ms {[round(v, 1) for v in latency_ms]}, "
      f"peak GB {[round(v, 2) for v in memory_gb]}")
