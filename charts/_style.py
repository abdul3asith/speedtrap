"""Shared look and helpers for every chart in charts/.

Colours are the blog's palette (abdulbasith site), not matplotlib defaults.
Each chart is saved three ways:
  <name>_light.svg, <name>_dark.svg  transparent, for the website (no title or
                                     footnote: the post's caption carries them)
  <name>_linkedin.png                white background, dpi 200, title and footnote
Files starting with "_" are helpers, not chart scripts (the Makefile skips them).
"""

import json
import pathlib
from contextlib import contextmanager

import matplotlib.pyplot as plt

# results/speed: written by the harness. results/logs: older runs transcribed from terminal logs.
RESULTS = [pathlib.Path("results/speed"), pathlib.Path("results/logs")]
# The run that sections 1, 5 and 6 of the post describe.
POST_RUN = "first-run"
OUT = pathlib.Path("charts/out")

# Two element colours, as on the site: ink (near-black / off-white) for the
# reference, maroon (the site accent) for the thing the chart is about.
THEMES = {
    "light": {
        "primary": "#1b1a17",    # main text
        "secondary": "#4a4843",  # labels
        "muted": "#85827a",      # footnotes, ticks
        "baseline": "#1b1a17",   # reference elements: site ink
        "accent": "#9a3412",     # site maroon
        "axis": "#c3c0b5",
        "grid": "#e6e3da",
        "tint": "#f3e3d8",       # highlighted table rows: site accent-soft
    },
    "dark": {
        "primary": "#ebe7de",
        "secondary": "#b9b4a8",
        "muted": "#807b70",
        "baseline": "#ebe7de",
        "accent": "#f0a272",     # the site's dark-mode accent
        "axis": "#403d37",
        "grid": "#2c2a26",
        "tint": "#3a2519",
    },
}
# LinkedIn PNGs sit on white, so they use the light palette.
THEMES["linkedin"] = THEMES["light"]

FONT = ["Inter", "Inter Variable", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
FONT_SIZE = 14     # ticks and labels
VALUE_SIZE = 16    # numbers on bars
SUB_SIZE = 14      # panel subtitles ("2.0× memory")
PANEL_SIZE = 15    # panel titles
TITLE_SIZE = 17    # LinkedIn headline
FOOT_SIZE = 11


def load_rows(model: str | None = None) -> list[dict]:
    """Every result row without an error, optionally for one model."""
    rows = []
    for path in sorted(p for d in RESULTS for p in d.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r or (model and r.get("model") != model):
                continue
            rows.append(r)
    return rows


def newest_complete_run(rows: list[dict], techniques: list[str]) -> dict[str, dict] | None:
    """Rows keyed by technique, from the newest run_id that has all `techniques`.

    Techniques are only compared within one run_id (one container, one machine).
    """
    runs: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r["technique"] in techniques:
            runs.setdefault(r["run_id"], {})[r["technique"]] = r
    complete = [run for run in runs.values() if all(t in run for t in techniques)]
    if not complete:
        return None
    return max(complete, key=lambda run: run[techniques[0]]["timestamp"])


def post_run(rows: list[dict], techniques: list[str]) -> dict[str, dict] | None:
    """The post's run (POST_RUN) if it has all `techniques`, else the newest complete run."""
    pinned = newest_complete_run([r for r in rows if r["run_id"] == POST_RUN], techniques)
    return pinned or newest_complete_run(rows, techniques)


def cloud_name(row: dict) -> str:
    return str(row.get("cloud", "unknown")).replace("CLOUD_PROVIDER_", "")


def footnote(row: dict) -> str:
    """Where the numbers came from, built from the result row itself."""
    where = [row["model"], row["gpu"]]
    if row.get("cloud"):
        where.append(f"{cloud_name(row)} host")
    if row.get("gpu_driver"):
        where.append(f"driver {row['gpu_driver']}")
    where.append(f"run {row['run_id']}")
    text = f"{', '.join(where)}. Warm-up {row['warmup']}, {row['n']} timed actions, GPU synchronized."
    if row.get("source"):
        text += f" {row['source'].capitalize()}."
    return text


def missing(what: str, command: str, extra: str = "") -> None:
    """Data isn't there yet: say exactly how to produce it, then skip this chart."""
    print(f"SKIPPED: {what}")
    print(f"  run: {command}")
    if extra:
        print(f"  {extra}")
    raise SystemExit(0)


@contextmanager
def theme_rc():
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "font.size": FONT_SIZE,
        "svg.fonttype": "none",  # keep text as text, so the site's Inter renders it
        "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        yield


def clean_axes(ax, c: dict, left: bool = False) -> None:
    ax.patch.set_alpha(0)
    ax.spines["left"].set_visible(left)
    ax.spines["left"].set_color(c["axis"])
    ax.spines["bottom"].set_color(c["axis"])
    ax.tick_params(colors=c["secondary"], labelsize=FONT_SIZE)


def render_all(name: str, draw, title: str, foot: str) -> list[pathlib.Path]:
    """Draw the chart for each theme and save the three files.

    `draw(c)` must create and return a figure using colours `c`.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in ("light", "dark", "linkedin"):
        c = THEMES[theme]
        with theme_rc():
            fig = draw(c)
            if theme == "linkedin":  # the website's figure caption replaces the footnote
                fig.text(0.01, 0.005, foot, fontsize=FOOT_SIZE, color=c["muted"], ha="left",
                         va="bottom", wrap=True)
                fig.suptitle(title, x=0.01, ha="left", fontsize=TITLE_SIZE,
                             fontweight="bold", color=c["primary"])
                fig.tight_layout(rect=(0, 0.05, 1, 0.93))
                path = OUT / f"{name}_linkedin.png"
                fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
            else:
                fig.patch.set_alpha(0)
                fig.tight_layout(rect=(0, 0.05, 1, 1))
                path = OUT / f"{name}_{theme}.svg"
                fig.savefig(path, transparent=True, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    for p in paths:
        print(f"  wrote {p}")
    return paths
