"""Experiment family 5: seasonal and era effects (paper Figures 10 and 11).

Seasonal: the paper trains logistic regression on songs released outside a period and tests it on
songs released inside (summer = Jun-Aug, holiday = Nov-Jan), and concludes that holiday hits differ.
Two problems: about a third of all songs carry a January release month because Spotify stores
year-only release dates as YYYY-01-01, so the 'holiday' set is mostly songs of unknown month; and the
in-period and out-of-period sets have different hit rates, which changes accuracy by itself.

Time periods: the paper trains a separate model per five-year period (80/20 split in file order) and
reads higher accuracy as evidence that hits are period-specific. With 31 to 243 validation songs per
period, one split is noisy, and hit rates differ a lot between periods. We repeat each split 30
times by artist and compare against a pooled model and the majority-class baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import data, plotting
from ..config import AUTHOR_SUBSETS, COLORS, PAPER, PAPER_FEATURES, SEED
from ..metrics import group_bootstrap_ci
from ..models import PaperNN
from .common import grouped_cv, paper_lr

SEASONS = {"Summer": [6, 7, 8], "Holiday": [11, 12, 1]}
PERIODS = {"1990-2018": (1990, 2018), "1990-1994": (1990, 1994), "1995-1999": (1995, 1999),
           "2000-2004": (2000, 2004), "2005-2009": (2005, 2009), "2010-2014": (2010, 2014),
           "2015-2018": (2015, 2018)}


def _fair_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def _acc(m, X, y):
    return float(np.mean(m.predict(X) == y))


def _paper_nn_within(df, feats, epochs=18000):
    """project_nn.py protocol: 80/20 split in file order, best validation accuracy over epochs."""
    n_tr = int(np.ceil(0.8 * len(df)))
    Xtr, ytr = data.xy(df.iloc[:n_tr], feats)
    Xva, yva = data.xy(df.iloc[n_tr:], feats)
    return float(max(PaperNN(feats, epochs=epochs).fit(Xtr, ytr, Xva, yva).history["val_acc"]))


# --------------------------------------------------------------------------- seasonal

def seasonal():
    raw = data.load_raw_dated()
    lr_feats, nn_feats = AUTHOR_SUBSETS["lr"], AUTHOR_SUBSETS["nn"]
    paper_proto = {}
    for season, months in SEASONS.items():
        inside = raw[raw.Month.isin(months)]
        outside = raw[~raw.Month.isin(months)]
        Xo, yo = data.xy(outside, lr_feats)
        Xi, yi = data.xy(inside, lr_feats)
        m = paper_lr().fit(Xo, yo)
        paper_proto[season] = {
            "n_inside": len(inside), "n_outside": len(outside),
            "LR_test_inside": _acc(m, Xi, yi), "LR_train_outside": _acc(m, Xo, yo),
            "NN_within_inside": _paper_nn_within(inside, nn_feats),
            "NN_within_outside": _paper_nn_within(outside, nn_feats),
        }

    # Fair: out-of-fold predictions for every dated song from artist-grouped 5-fold CV,
    # then accuracy and hit rate broken down by release period.
    df = data.load_master()
    df = df[df.Dated].reset_index(drop=True)
    X, y = data.xy(df, PAPER_FEATURES)
    g = df.Group.to_numpy()
    pred = np.zeros(len(df), int)
    for a, b in grouped_cv(5, SEED).split(X, y, g):
        pred[b] = _fair_lr().fit(X[a], y[a]).predict(X[b])
    month = df.Month.to_numpy()
    subsets = {
        "Summer (Jun-Aug)": np.isin(month, [6, 7, 8]),
        "Non-summer": ~np.isin(month, [6, 7, 8]),
        "Holiday (Nov-Jan)": np.isin(month, [11, 12, 1]),
        "Non-holiday": ~np.isin(month, [11, 12, 1]),
        "January only": month == 1,
        "Nov-Dec only": np.isin(month, [11, 12]),
        "Feb-Oct": ~np.isin(month, [11, 12, 1]),
    }
    fair = {}
    for name, mask in subsets.items():
        fair[name] = {
            "n": int(mask.sum()), "hit_rate": float(y[mask].mean()),
            "majority_baseline": float(max(y[mask].mean(), 1 - y[mask].mean())),
            "oof_accuracy": float(np.mean(pred[mask] == y[mask])),
            "oof_accuracy_ci95": group_bootstrap_ci(y[mask], pred[mask], g[mask]),
        }
    month_counts = {int(k): int(v) for k, v in df.Month.value_counts().sort_index().items()}
    return {"paper_protocol": paper_proto, "fair_oof_by_period": fair,
            "month_counts": month_counts,
            "january_share": float(np.mean(month == 1))}


# --------------------------------------------------------------------------- five-year periods

def time_periods(n_repeats: int = 30):
    raw = data.load_raw_dated()
    lr_feats, nn_feats = AUTHOR_SUBSETS["lr"], AUTHOR_SUBSETS["nn"]
    paper_proto = {}
    for name, (lo, hi) in PERIODS.items():
        sub = raw[(raw.Year >= lo) & (raw.Year <= hi)]
        n_tr = int(round(0.8 * len(sub)))
        Xtr, ytr = data.xy(sub.iloc[:n_tr], lr_feats)
        Xva, yva = data.xy(sub.iloc[n_tr:], lr_feats)
        paper_proto[name] = {"n": len(sub), "n_val": len(sub) - n_tr,
                             "LR_val": _acc(paper_lr().fit(Xtr, ytr), Xva, yva),
                             "NN_val_best_epoch": _paper_nn_within(sub, nn_feats)}

    df = data.load_master()
    df = df[df.Dated].reset_index(drop=True)
    fair = {}
    for name, (lo, hi) in PERIODS.items():
        if name == "1990-2018":
            continue
        in_p = (df.Year >= lo) & (df.Year <= hi)
        sub, others = df[in_p], df[~in_p]
        specific, pooled, majority = [], [], []
        for r in range(n_repeats):
            tr, te = data.group_split(sub, 0.2, seed=SEED + r)
            # drop other-period songs by artists in this test fold, so pooled training is fair too
            oth = others[~others.Group.isin(te.Group)]
            Xtr, ytr = data.xy(tr, PAPER_FEATURES)
            Xte, yte = data.xy(te, PAPER_FEATURES)
            specific.append(_acc(_fair_lr().fit(Xtr, ytr), Xte, yte))
            Xp, yp = data.xy(oth, PAPER_FEATURES)
            pooled.append(_acc(_fair_lr().fit(np.vstack([Xtr, Xp]), np.concatenate([ytr, yp])),
                               Xte, yte))
            majority.append(float(np.mean(yte == int(ytr.mean() >= 0.5))))
        fair[name] = {"n": int(in_p.sum()), "hit_rate": float(sub.Label.mean()),
                      "period_model_mean": float(np.mean(specific)),
                      "period_model_sd": float(np.std(specific)),
                      "pooled_model_mean": float(np.mean(pooled)),
                      "pooled_model_sd": float(np.std(pooled)),
                      "majority_mean": float(np.mean(majority)),
                      "pooled_minus_period": float(np.mean(pooled) - np.mean(specific))}
    return {"paper_protocol": paper_proto, "fair_repeated": fair, "n_repeats": n_repeats}


# --------------------------------------------------------------------------- figures

def figures(seas, per):
    import matplotlib.pyplot as plt

    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    counts = seas["month_counts"]
    months = list(range(1, 13))
    ax.bar(months, [counts.get(m, 0) for m in months], 0.8, color=COLORS["ours"])
    ax.set_xticks(months)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                        "Nov", "Dec"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Songs")
    ax.set_title(f"Release month: {seas['january_share']:.0%} of songs fall in January "
                 "(year-only dates)", fontsize=11)
    ax = axes[1]
    f = seas["fair_oof_by_period"]
    names = ["Summer (Jun-Aug)", "Non-summer", "Holiday (Nov-Jan)", "Non-holiday",
             "Nov-Dec only"]
    plotting.grouped_bars(
        ax, [n.replace(" (", "\n(").replace(" only", "\nonly") for n in names],
        {"Model accuracy (grouped CV)": [f[n]["oof_accuracy"] for n in names],
         "Majority-class baseline": [f[n]["majority_baseline"] for n in names]},
        {"Model accuracy (grouped CV)": COLORS["ours"], "Majority-class baseline": COLORS["muted"]},
        ylim=(0.4, 0.9), fmt="{:.0%}", label_series=("Model accuracy (grouped CV)",),
        err={"Model accuracy (grouped CV)": [f[n]["oof_accuracy_ci95"] for n in names]})
    ax.set_title("Accuracy by release period", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    paths.append(plotting.save(fig, "07_seasonal"))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = [n for n in PERIODS if n != "1990-2018"]
    fr = per["fair_repeated"]
    plotting.grouped_bars(
        ax, [f"{n}\n{fr[n]['hit_rate']:.0%} hits, n={fr[n]['n']}" for n in names],
        {"Paper, period model (one split)": [PAPER["time_periods"]["LR"][n] for n in names],
         "Ours, period model (30 splits)": [fr[n]["period_model_mean"] for n in names],
         "Ours, pooled model (30 splits)": [fr[n]["pooled_model_mean"] for n in names]},
        {"Paper, period model (one split)": COLORS["paper"],
         "Ours, period model (30 splits)": COLORS["ours"],
         "Ours, pooled model (30 splits)": COLORS["third"]},
        ylim=(0.4, 1.0), fmt="{:.0%}", label_series=("Ours, period model (30 splits)",),
        err={"Ours, period model (30 splits)": [
            (fr[n]["period_model_mean"] - fr[n]["period_model_sd"],
             fr[n]["period_model_mean"] + fr[n]["period_model_sd"]) for n in names]})
    x = np.arange(len(names))
    ax.scatter(x, [fr[n]["majority_mean"] for n in names], marker="_", s=900, color=COLORS["ink2"],
               zorder=5, label="Majority-class baseline")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=4, fontsize=8.5)
    ax.set_ylabel("Accuracy")
    ax.set_title("Five-year periods: logistic regression accuracy (error bars: +-1 sd over splits)")
    paths.append(plotting.save(fig, "08_time_periods"))
    return paths


def run():
    seas = seasonal()
    per = time_periods()
    return {"seasonal": seas, "time_periods": per, "figures": figures(seas, per)}
