"""Section 4: one action as the GPU sees it — short kernels separated by gaps.

Reads a torch.profiler Chrome trace from results/traces/<run_id>_native.json, made by
`modal run cloud/modal_app.py --techniques native --profile`. Each action in the
trace is wrapped in a record_function("action") range (with a GPU sync inside), so
the kernels for one action are the ones that start inside that range.
"""

import json
import pathlib

import matplotlib.pyplot as plt
from _style import FONT_SIZE, clean_axes, footnote, load_rows, missing, render_all
from matplotlib.patches import Rectangle

TRACES = pathlib.Path("results/traces")
ZOOM_MS = 2.0

traces = sorted(p for p in TRACES.glob("*_native.json")) if TRACES.exists() else []
if not traces:
    missing(
        "kernel_timeline: no profiler trace in results/traces/",
        "modal run cloud/modal_app.py --techniques native --profile",
        "then: modal volume get hf-cache traces/<run_id>_native.json results/traces/  "
        "(the run prints the exact command with its run_id)",
    )
trace_path = max(traces, key=lambda p: p.stat().st_mtime)
run_id = trace_path.name.removesuffix("_native.json")
events = json.loads(trace_path.read_text())["traceEvents"]

actions = sorted((e for e in events if e.get("cat") == "user_annotation" and e.get("name") == "action"),
                 key=lambda e: e["ts"])
kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
if not actions or not kernels:
    missing(f"kernel_timeline: {trace_path} has no 'action' ranges or no CUDA kernels",
            "modal run cloud/modal_app.py --techniques native --profile")

action = actions[-1]  # the last profiled action: furthest from any warm-up effects
t0, t1 = action["ts"], action["ts"] + action["dur"]
ks = sorted((k for k in kernels if t0 <= k["ts"] < t1), key=lambda k: k["ts"])
spans = [((k["ts"] - t0) / 1000, k["dur"] / 1000) for k in ks]  # (start ms, length ms)

# GPU busy time = union of kernel intervals (kernels can overlap on different streams).
busy, end = 0.0, -1.0
for s, d in spans:
    if s + d > end:
        busy += (s + d) - max(s, end)
        end = s + d
wall_ms = action["dur"] / 1000
busy_frac = busy / wall_ms

# Zoom window: the 2 ms with the most kernel starts, so individual kernels are visible.
starts = [s for s, _ in spans]
zoom_start = max(starts, key=lambda s: sum(1 for t in starts if s <= t < s + ZOOM_MS))

row = next((r for r in load_rows() if r["run_id"] == run_id and r["technique"] == "native"), None)
foot = footnote(row) if row else f"Profiler trace {trace_path.name}."
foot += f" Profiled action {len(actions)} of {len(actions)}."


def draw(c):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.4), gridspec_kw={"height_ratios": [1, 1]})
    for ax, (lo, hi) in ((ax1, (0, wall_ms)), (ax2, (zoom_start, zoom_start + ZOOM_MS))):
        ax.broken_barh(spans, (0, 1), facecolors=c["accent"], edgecolor="none")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        clean_axes(ax, c)
        ax.set_xlabel("ms from the start of the action", color=c["muted"], fontsize=FONT_SIZE - 1)
    ax1.add_patch(Rectangle((zoom_start, 0), ZOOM_MS, 1, fill=False, edgecolor=c["baseline"], lw=1.5))
    ax1.set_title(f"One action: {wall_ms:.0f} ms", loc="left", fontsize=15,
                  fontweight="semibold", color=c["primary"], pad=6)
    ax1.text(1, 1.04, f"{len(ks)} kernels per action  ·  GPU busy {busy_frac:.0%} of the time",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=FONT_SIZE, color=c["secondary"])
    ax2.set_title(f"Zoomed: {ZOOM_MS:g} ms (outlined above)", loc="left", fontsize=15,
                  fontweight="semibold", color=c["primary"], pad=6)
    return fig


render_all("kernel_timeline", draw, title="Most of each action, the GPU is waiting", foot=foot)
print(f"kernel_timeline from {trace_path.name}: {len(ks)} kernels, wall {wall_ms:.1f} ms, "
      f"GPU busy {busy:.1f} ms ({busy_frac:.1%}), zoom at {zoom_start:.2f} ms")
