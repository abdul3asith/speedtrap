"""Section 1 chart: memory doubles, latency doesn't. Reads saved speed results.

Writes a light and a dark version (transparent background) to charts/out/.
"""

import json
import pathlib

import matplotlib.pyplot as plt

RESULTS = pathlib.Path("results/speed")
OUT = pathlib.Path("charts/out")
MODEL = "smolvla"
BASE, TEST = "native", "fp32_cast"

# Site palette (light, dark).
THEMES = {
    "light": {
        "primary": "#1b1a17",
        "secondary": "#4a4843",
        "muted": "#85827a",
        "accent": "#9a3412",
        "axis": "#c3c0b5",
    },
    "dark": {
        "primary": "#ebe7de",
        "secondary": "#b9b4a8",
        "muted": "#807b70",
        "accent": "#f0a272",
        "axis": "#403d37",
    },
}
FONT = ["Inter", "Inter Variable", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pick_run(rows: list[dict]) -> dict[str, dict]:
    """The newest container that ran both techniques without errors."""
    runs: dict[str, dict[str, dict]] = {}
    for r in rows:
        if (
            r.get("model") == MODEL
            and "error" not in r
            and r["technique"] in (BASE, TEST)
        ):
            runs.setdefault(r["run_id"], {})[r["technique"]] = r
    complete = [run for run in runs.values() if BASE in run and TEST in run]
    if not complete:
        raise SystemExit(f"No run has both {BASE} and {TEST}. Run modal first.")
    return max(complete, key=lambda run: run[BASE]["timestamp"])


run = pick_run(load_rows())
base, test = run[BASE], run[TEST]

formats = ["bf16\n(as shipped)", "fp32"]
latency_ms = [base["p50_ms"], test["p50_ms"]]
memory_gb = [base["peak_mem_gb"], test["peak_mem_gb"]]
mem_ratio = memory_gb[1] / memory_gb[0]
lat_ratio = latency_ms[1] / latency_ms[0]
footnote = (
    f"{MODEL}, {base['gpu']}, {base['cloud'].replace('CLOUD_PROVIDER_', '')} host, "
    f"driver {base['gpu_driver']}, one container (run {base['run_id']}). "
    f"Warm-up {base['warmup']}, {base['n']} timed actions, GPU synchronized."
)


def panel(ax, c, values, title, unit, fmt, ratio):
    bars = ax.bar(formats, values, width=0.5, color=[c["muted"], c["accent"]])
    top = max(values)
    ax.set_ylim(0, top * 1.32)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["axis"])
    ax.tick_params(axis="x", length=0, pad=8, colors=c["secondary"], labelsize=11)
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold", color=c["primary"], pad=6)
    ax.text(0, 1.0, ratio, transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color=c["secondary"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.03, f"{v:{fmt}} {unit}",
                ha="center", va="bottom", fontsize=12, fontweight="semibold",
                color=c["primary"])


def render(name: str) -> None:
    c = THEMES[name]
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": FONT,
                         "svg.fonttype": "none"}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8), dpi=200)
        fig.patch.set_alpha(0)
        for ax in (ax1, ax2):
            ax.patch.set_alpha(0)
        panel(ax1, c, memory_gb, "Peak GPU memory", "GB", ".2f", f"{mem_ratio:.1f}× memory")
        panel(ax2, c, latency_ms, "Time per action (p50)", "ms", ".0f", f"{lat_ratio:.2f}× time")
        fig.suptitle("Doubling the precision doubled the memory, but not the time",
                     x=0.02, ha="left", fontsize=15, fontweight="semibold", color=c["primary"])
        fig.text(0.02, 0.01, footnote, fontsize=8.5, color=c["muted"])
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        OUT.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg"):
            fig.savefig(OUT / f"fp32_vs_bf16_{name}.{ext}", bbox_inches="tight", transparent=True)
        plt.close(fig)


for theme in THEMES:
    render(theme)
print(f"chart from run {base['run_id']}: latency {latency_ms}, memory {memory_gb}")
