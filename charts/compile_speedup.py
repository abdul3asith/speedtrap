"""Section 6: torch.compile — much faster per action, after a long first call."""

import statistics

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
BASE, TEST = "native", "compile"

run = post_run(load_rows(MODEL), [BASE, TEST])
if run is None:
    missing(f"compile_speedup: no run has both {BASE} and {TEST}",
            f"modal run cloud/modal_app.py --techniques {BASE},{TEST}",
            "the first compiled call can take ~20 min: keep the modal timeout at >= 7200 s "
            "(run_speed currently has timeout=7200)")
base, test = run[BASE], run[TEST]

labels = ["original", "compiled"]
first_labels = list(labels)
p50 = [base["p50_ms"], test["p50_ms"]]
speedup = p50[0] / p50[1]
foot = footnote(base)

# If this run didn't log the original's first call, use the median over other native runs.
base_first = base.get("first_call_ms")
if base_first is None:
    others = [r["first_call_ms"] for r in load_rows(MODEL)
              if r["technique"] == BASE and r.get("first_call_ms")]
    if not others:
        missing("compile_speedup: no first-call time for the original model",
                f"modal run cloud/modal_app.py --techniques {BASE},{TEST}")
    base_first = statistics.median(others)
    first_labels[0] = "original*"
    foot += f" *Original first call: median of {len(others)} other runs (not logged for this run)."
first_s = [base_first / 1000, test["first_call_ms"] / 1000]
first_ratio = first_s[1] / first_s[0]


def fmt_duration(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def bars(ax, c, names, values, texts, colors, title, note):
    top = max(values)
    # A bar under ~2% of the tallest would be invisible on a linear scale (1.4 s next
    # to 18.5 min is 0.1%), so draw it at a visible minimum. Its label keeps the real value.
    shown = [max(v, top * 0.02) for v in values]
    b = ax.bar(names, shown, width=0.5, color=colors)
    ax.set_ylim(0, top * 1.45)
    ax.set_yticks([])
    clean_axes(ax, c)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_title(title, loc="left", fontsize=PANEL_SIZE, fontweight="semibold", color=c["primary"], pad=6)
    ax.text(0, 1.0, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=SUB_SIZE, color=c["secondary"])
    for bar, v, t in zip(b, shown, texts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.03, t, ha="center", va="bottom",
                fontsize=VALUE_SIZE, fontweight="semibold", color=c["primary"])


def draw(c):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.4))
    bars(ax1, c, labels, p50, [f"{v:.0f} ms" for v in p50], [c["baseline"], c["accent"]],
         "Time per action (p50)", f"{speedup:.1f}× faster")
    bars(ax2, c, first_labels, first_s, [fmt_duration(v) for v in first_s], [c["baseline"], c["accent"]],
         "First call (one-time startup)", f"about {first_ratio:,.0f}× longer, once per restart")
    return fig


render_all(
    "compile_speedup",
    draw,
    title=f"Fewer, bigger orders: {speedup:.0f}× faster, after a long first call",
    foot=foot,
)
print(f"compile_speedup from run {base['run_id']}: p50 ms {[round(v, 1) for v in p50]} "
      f"({speedup:.2f}×), first call {[fmt_duration(v) for v in first_s]}")
