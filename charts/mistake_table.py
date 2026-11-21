"""Section 8: the same CPU setting gave both ~355 ms and ~570 ms.

These five numbers come from earlier run LOGS, not from results/speed: those runs
didn't record the host machine, so they were never saved as result rows. They are
hard-coded here on purpose, and the footnote says so.
"""

import matplotlib.pyplot as plt
from _style import FONT_SIZE, render_all

RUNS = [  # (run, CPU setting, p50 ms) — from run logs
    ("First", "default", 573),
    ("Second", "8 cores", 354),
    ("Third", "8 cores", 357),
    ("Fourth", "8 cores", 575),
    ("Fifth", "8 cores", 560),
]
FAST_BELOW_MS = 450  # splits the two clusters (~355 vs ~570)


def draw(c):
    fig = plt.figure(figsize=(7.5, 3.6))
    ax = fig.add_axes((0.01, 0.08, 0.98, 0.84))  # full width, aligned with the title
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    row_h = 1 / (len(RUNS) + 1)
    cols = [(0.03, "left", "Run"), (0.35, "left", "CPU setting"), (0.97, "right", "Time per action (p50)")]
    y = 1 - row_h / 2
    for x, a, h in cols:
        ax.text(x, y, h, ha=a, va="center", fontsize=FONT_SIZE - 1, fontweight="semibold",
                color=c["secondary"])
    ax.plot([0, 1], [1 - row_h] * 2, color=c["primary"], lw=1.2)
    for i, (run, setting, ms) in enumerate(RUNS):
        y = 1 - row_h * (i + 1.5)
        if setting == "8 cores":  # the rows that should all match, but don't
            ax.add_patch(plt.Rectangle((0, y - row_h / 2), 1, row_h, color=c["tint"], lw=0))
        ax.plot([0, 1], [y - row_h / 2] * 2, color=c["grid"], lw=1)
        fast = ms < FAST_BELOW_MS
        ax.text(0.03, y, run, ha="left", va="center", color=c["primary"])
        ax.text(0.35, y, setting, ha="left", va="center", color=c["primary"])
        ax.text(0.97, y, f"{ms} ms", ha="right", va="center", fontweight="semibold",
                color=c["accent"] if fast else (c["warning"] if setting == "8 cores" else c["primary"]))
    return fig


render_all(
    "mistake_table",
    draw,
    title="Same setting, different machines",
    foot="From run logs; the host machine was not recorded for these runs.",
)
print("mistake_table (hard-coded from logs):", [(r, s, ms) for r, s, ms in RUNS])
