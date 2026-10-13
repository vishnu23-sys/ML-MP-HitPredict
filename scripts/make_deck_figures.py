"""Charts for the review deck, drawn from results/results.json in the deck's palette.

Each figure is sized in slide pixels (figsize = width/100, height/100 inches at 100 dpi, saved at
2x), so a 20 pt label prints at about 28 px on the 1920x1080 slide.

Usage:
    python scripts/make_deck_figures.py [--font-dir DIR]   # DIR with IBM Plex Sans .ttf files
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from hitpredict.config import PAPER, ROOT  # noqa: E402

PLUM, BLUE, RED, SAGE = "#503156", "#6EA4BF", "#A31621", "#BCD8C1"
LIGHT, INK, MUTED, GRID = "#FCF7F8", "#2E1C31", "#6B5A6E", "#E6DCE2"
OUT = ROOT / "report" / "deck"
R = json.loads((ROOT / "results" / "results.json").read_text())


def style(font_dir):
    family = "sans-serif"
    if font_dir:
        for f in Path(font_dir).glob("*.ttf"):
            font_manager.fontManager.addfont(str(f))
        family = "IBM Plex Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 19, "text.color": INK,
        "figure.facecolor": LIGHT, "axes.facecolor": LIGHT, "savefig.facecolor": LIGHT,
        "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "axes.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1.2,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
        "ytick.labelcolor": MUTED, "legend.frameon": False,
    })


def fig(w, h):
    return plt.subplots(figsize=(w / 100, h / 100), dpi=100)


def save(f, name):
    OUT.mkdir(parents=True, exist_ok=True)
    f.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(f)
    print("wrote", OUT / f"{name}.png")


def label_bars(ax, bars, vals, fmt="{:.1f}", color=INK, dy=0.6):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt.format(v), ha="center", va="bottom",
                fontsize=18, color=color, fontweight="semibold")


def reproduction():
    rep = R["supervised"]["paper_protocol"]
    keys = ["LR", "GDA", "SVM linear", "SVM RBF", "SVM polynomial", "DT", "NN"]
    names = ["Logistic\nregression", "GDA", "SVM\nlinear", "SVM\nRBF", "SVM\npolynomial",
             "Decision\ntree", "Neural\nnetwork"]
    paper = [100 * PAPER["table2"][k]["val_acc"] for k in keys]
    ours = [100 * (rep[k]["val_acc_best_epoch_as_reported"] if k == "NN"
                   else rep[k]["val"]["accuracy"]) for k in keys]
    f, ax = fig(1664, 600)
    x = range(len(keys))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], paper, w * 0.94, color=BLUE, label="Paper's number")
    b2 = ax.bar([i + w / 2 for i in x], ours, w * 0.94, color=PLUM, label="Our rerun")
    label_bars(ax, b1, paper, color=MUTED)
    label_bars(ax, b2, ours)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(50, 86)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2, fontsize=19)
    f.tight_layout(pad=0.4)
    save(f, "reproduction")


def bugs():
    sup, em = R["supervised"]["paper_protocol"], R["unsupervised"]["unsupervised"]
    bag = next(b for b in R["overfitting"]["bagging"]
               if b["model"] == "DT" and b["samples_per_bag"] == 200 and b["bags"] == 50)
    names = ["SVM accuracy on\nits own training songs", "Grouping songs\nwithout labels (EM)",
             "Voting trees\n(50 trees)"]
    paper = [100 * PAPER["table2"]["SVM linear"]["train_acc"], 100 * PAPER["em_unsupervised_acc"],
             100 * bag["paper_val_acc"]]
    fixed = [100 * sup["SVM linear"]["train"]["accuracy"], 100 * em["best_label_matching_mean"],
             100 * bag["majority_vote_val"]]
    f, ax = fig(1664, 600)
    x = range(3)
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], paper, w * 0.94, color=BLUE, label="What the paper reported")
    b2 = ax.bar([i + w / 2 for i in x], fixed, w * 0.94, color=RED, label="Counted correctly")
    label_bars(ax, b1, paper, color=MUTED, dy=1.2)
    label_bars(ax, b2, fixed, dy=1.2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2, fontsize=19, bbox_to_anchor=(0, 1.08))
    f.tight_layout(pad=0.4)
    save(f, "bugs")


def months():
    counts = R["temporal"]["seasonal"]["month_counts"]
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    vals = [counts.get(str(m), 0) for m in range(1, 13)]
    f, ax = fig(800, 560)
    bars = ax.bar(range(12), vals, 0.78, color=[RED] + [PLUM] * 11)
    ax.text(0, vals[0] + 25, f"{vals[0]:,}", ha="center", va="bottom", fontsize=19, color=RED,
            fontweight="semibold")
    ax.set_xticks(range(12))
    ax.set_xticklabels(labels, fontsize=17)
    ax.set_ylabel("Songs", fontsize=18)
    ax.set_ylim(0, vals[0] * 1.15)
    ax.grid(axis="x", visible=False)
    f.tight_layout(pad=0.4)
    save(f, "months")


def holiday():
    fo = R["temporal"]["seasonal"]["fair_oof_by_period"]
    names = ["Nov and Dec", "Feb to Oct", "January"]
    vals = [100 * fo[k]["oof_accuracy"] for k in ("Nov-Dec only", "Feb-Oct", "January only")]
    f, ax = fig(800, 560)
    bars = ax.bar(range(3), vals, 0.6, color=[PLUM, PLUM, RED])
    label_bars(ax, bars, vals, dy=0.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names)
    ax.set_ylim(60, 82)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.set_ylabel("Accuracy on unseen artists", fontsize=18)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="x", visible=False)
    f.tight_layout(pad=0.4)
    save(f, "holiday")


def models():
    mc = R["model_comparison"]["models"]
    plain = {"Majority class": "Always \"not a hit\"", "Artist score rule": "Artist score only",
             "Naive Bayes": "Naive Bayes", "GDA (shared covariance)": "GDA",
             "Logistic regression": "Logistic regression", "Linear SVM": "Linear SVM",
             "k-nearest neighbours": "Nearest neighbours", "Decision tree": "Decision tree",
             "RBF SVM": "SVM (RBF)", "Neural network (6 sigmoid units)": "Small neural net",
             "Neural network (32-16 ReLU)": "Larger neural net", "Random forest": "Random forest",
             "Gradient boosting": "Gradient boosting"}
    order = sorted(mc, key=lambda k: mc[k]["test"]["accuracy"])
    vals = [100 * mc[k]["test"]["accuracy"] for k in order]
    cols = [RED if k == R["model_comparison"]["best_by_cv"] else BLUE if k == "Artist score rule"
            else MUTED if k == "Majority class" else PLUM for k in order]
    f, ax = fig(1000, 780)
    bars = ax.barh(range(len(order)), vals, 0.7, color=cols)
    for b, v in zip(bars, vals):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=17,
                color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([plain[k] for k in order], fontsize=17)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlim(55, 80)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="y", visible=False)
    f.tight_layout(pad=0.4)
    save(f, "models")


def steps():
    st = R["improvements"]["steps"]
    names = ["Paper's\nmodel", "+ more\nsongs", "+ better\nfeatures", "+ boosted\ntrees"]
    vals = [100 * s["test"]["accuracy"] for s in st]
    lo = [100 * s["test_acc_ci95"][0] for s in st]
    hi = [100 * s["test_acc_ci95"][1] for s in st]
    f, ax = fig(960, 600)
    bars = ax.bar(range(4), vals, 0.62, color=[BLUE, PLUM, PLUM, RED],
                  yerr=[[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]],
                  error_kw={"ecolor": MUTED, "elinewidth": 1.6, "capsize": 5})
    for b, v, h in zip(bars, vals, hi):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.3, f"{v:.1f}", ha="center", va="bottom",
                fontsize=19, color=INK, fontweight="semibold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(names)
    ax.set_ylim(66, 82)
    ax.set_ylabel("Accuracy on unseen artists", fontsize=18)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="x", visible=False)
    f.tight_layout(pad=0.4)
    save(f, "steps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--font-dir")
    style(ap.parse_args().font_dir)
    for fn in (reproduction, bugs, months, holiday, models, steps):
        fn()
