"""Section 5: every technique from one machine, as a Markdown table and a LinkedIn image."""

import pathlib

import matplotlib.pyplot as plt
from _style import (
    FONT_SIZE,
    FOOT_SIZE,
    OUT,
    THEMES,
    TITLE_SIZE,
    footnote,
    load_rows,
    missing,
    newest_complete_run,
    theme_rc,
)

MODEL = "smolvla"
TECHNIQUES = {
    "native": "As shipped (bf16)",
    "fp32_cast": "Twice the precision",
    "fp16_autocast": "Math in 16-bit",
    "bf16_autocast": "Math in bf16",
    "int8_weight": "Weights in 8-bit integers",
}
TABLE_MD = pathlib.Path("results/tables/technique_table.md")

run = newest_complete_run(load_rows(MODEL), list(TECHNIQUES))
if run is None:
    missing("technique_table: no run has all five techniques without errors",
            "modal run cloud/modal_app.py --techniques " + ",".join(TECHNIQUES))

native = run["native"]
rows = []
for tech, what in TECHNIQUES.items():
    r = run[tech]
    ratio = native["p50_ms"] / r["p50_ms"]  # >1 means faster than the original
    rows.append((tech, what, r["p50_ms"], r["peak_mem_gb"], ratio))

# 1. Markdown, for the post.
TABLE_MD.parent.mkdir(parents=True, exist_ok=True)
lines = [
    "| Technique | What it does | p50 (ms) | Peak memory (GB) | vs. original |",
    "|---|---|---:|---:|---:|",
]
lines += [f"| {t} | {w} | {p:.0f} | {m:.2f} | {x:.2f}× |" for t, w, p, m, x in rows]
lines += ["", f"<!-- {footnote(native)} -->", ""]
TABLE_MD.write_text("\n".join(lines))

# 2. The same table as an image for LinkedIn.
c = THEMES["linkedin"]
headers = ["Technique", "What it does", "p50 (ms)", "Peak memory (GB)", "vs. original"]
col_x = [0.02, 0.24, 0.64, 0.83, 0.98]          # left edge (text) or right edge (numbers)
align = ["left", "left", "right", "right", "right"]
row_h = 0.14

with theme_rc():
    fig = plt.figure(figsize=(10, 3.6))
    ax = fig.add_axes((0.01, 0.08, 0.98, 0.8))  # full width, aligned with the title
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    top = 0.95
    ax.add_patch(plt.Rectangle((0, top - row_h), 1, row_h, color=c["tint"], lw=0))
    for x, h, a in zip(col_x, headers, align):
        ax.text(x, top - row_h / 2, h, ha=a, va="center",
                fontsize=FONT_SIZE - 1, fontweight="semibold", color=c["secondary"])
    for i, (t, w, p, m, x) in enumerate(rows):
        y = top - row_h * (i + 1.5)
        ax.plot([0, 1], [y - row_h / 2] * 2, color=c["grid"], lw=1)
        if t == "native":
            vs_color = c["primary"]
        else:
            vs_color = c["accent"] if x > 1 else c["warning"]
        cells = [t, w, f"{p:.0f}", f"{m:.2f}", f"{x:.2f}×"]
        colors = [c["primary"], c["secondary"], c["primary"], c["primary"], vs_color]
        for cx, text, a, col in zip(col_x, cells, align, colors):
            ax.text(cx, y, text, ha=a, va="center",
                    fontsize=FONT_SIZE, color=col,
                    fontweight="semibold" if col in (c["accent"], c["warning"]) else "normal",
                    family="monospace" if a == "left" and cx == col_x[0] else None)
    faster = [t for t, _, _, _, x in rows if t != "native" and x > 1.05]  # >5% faster
    title = ("None of these tricks made it meaningfully faster" if not faster
             else f"Only {', '.join(faster)} beat the original")
    fig.suptitle(title, x=0.01, ha="left",
                 fontsize=TITLE_SIZE, fontweight="bold", color=c["primary"])
    fig.text(0.01, 0.01, footnote(native) + "  vs. original = original p50 ÷ technique p50.",
             fontsize=FOOT_SIZE, color=c["muted"])
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "technique_table_linkedin.png"
    fig.savefig(png, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)

print(f"  wrote {TABLE_MD}\n  wrote {png}")
print(f"technique_table from run {native['run_id']}:")
for t, _, p, m, x in rows:
    print(f"  {t:15s} p50 {p:7.1f} ms   peak {m:.2f} GB   {x:.2f}×")
