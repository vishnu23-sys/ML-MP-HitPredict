"""Shared matplotlib style and figure helpers (thin marks, hairline grid, legend for >= 2 series)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import COLORS, FIG_DIR  # noqa: E402


def setup():
    plt.rcParams.update({
        "figure.facecolor": COLORS["surface"],
        "axes.facecolor": COLORS["surface"],
        "savefig.facecolor": COLORS["surface"],
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.edgecolor": COLORS["axis"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": COLORS["ink2"],
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLORS["ink"],
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "xtick.labelcolor": COLORS["ink2"],
        "ytick.labelcolor": COLORS["ink2"],
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(FIG_DIR.parent.parent)).replace("\\", "/")


def grouped_bars(ax, categories, series: dict, colors: dict, fmt="{:.1%}", label_series=(),
                 ylim=None, err: dict | None = None):
    """Grouped vertical bars with a small gap between bars.

    Only the series named in `label_series` get value labels (selective direct labelling); the
    full numbers live in results.json and the README tables.
    """
    n = len(series)
    x = np.arange(len(categories))
    width = 0.8 / n
    for i, (name, vals) in enumerate(series.items()):
        pos = x - 0.4 + width * (i + 0.5)
        vals = np.asarray(vals, float)
        yerr, tops = None, vals
        if err and name in err:
            lo, hi = np.asarray(err[name], float).T
            yerr = np.vstack([vals - lo, hi - vals])
            tops = hi
        bars = ax.bar(pos, vals, width * 0.88, color=colors[name], label=name, yerr=yerr,
                      error_kw={"ecolor": COLORS["ink2"], "elinewidth": 1, "capsize": 2})
        if name in label_series:
            for b, v, t in zip(bars, vals, tops):
                if np.isfinite(v):
                    ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, t),
                                xytext=(0, 3), textcoords="offset points", ha="center",
                                va="bottom", fontsize=7.5, color=COLORS["ink2"])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    if n >= 2:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=n, fontsize=9)
    return ax
