"""Experiment family 6: how much does the split strategy change the score?

Songs by the same artist share a sound and, through the artist score, a large part of their label:
of the 883 lead artists with more than one song, 757 have only hits or only non-hits. If an
artist's songs sit on both sides of the split, a model could recognise the artist rather than learn
what makes a hit. The paper's split puts 56% of validation songs next to a song by the same artist
in training. Here we measure how much that matters, over 20 seeds and with fixed hyperparameters:

* random row-level split (what the paper effectively did),
* artist-grouped split (our protocol), and
* a temporal split (train on releases up to 2006, test on 2007-2018) as a stricter extra check.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import data, plotting
from ..config import COLORS, PAPER_FEATURES, SEED


def _models():
    return {
        "Logistic regression": lambda: make_pipeline(StandardScaler(),
                                                     LogisticRegression(max_iter=2000)),
        "k-NN (k=15)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
        "Random forest": lambda: RandomForestClassifier(300, random_state=SEED, n_jobs=-1),
        "Gradient boosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
    }


def run(n_seeds: int = 20):
    df = data.load_master()
    raw = data.load_raw_dated()
    tr0, va0 = data.paper_split(raw)
    leak = {
        "paper_split_val_songs_with_artist_in_train": float(va0.Artist.isin(tr0.Artist).mean()),
        "artists_with_multiple_songs": int((df.groupby("Group").size() > 1).sum()),
        "of_which_single_label": int(df.groupby("Group").Label.agg(
            lambda s: len(s) > 1 and s.nunique() == 1).sum()),
    }
    res = {name: {"random": [], "grouped": []} for name in _models()}
    for s in range(n_seeds):
        splits = {"random": data.random_split(df, 0.25, SEED + s),
                  "grouped": data.group_split(df, 0.25, SEED + s)}
        for kind, (tr, te) in splits.items():
            Xtr, ytr = data.xy(tr, PAPER_FEATURES)
            Xte, yte = data.xy(te, PAPER_FEATURES)
            for name, fac in _models().items():
                res[name][kind].append(float(np.mean(fac().fit(Xtr, ytr).predict(Xte) == yte)))

    dated = df[df.Dated]
    tr, te = dated[dated.Year <= 2006], dated[dated.Year >= 2007]
    tr = tr[~tr.Group.isin(te.Group)]
    Xtr, ytr = data.xy(tr, PAPER_FEATURES)
    Xte, yte = data.xy(te, PAPER_FEATURES)
    temporal = {name: float(np.mean(fac().fit(Xtr, ytr).predict(Xte) == yte))
                for name, fac in _models().items()}
    temporal_info = {"n_train": len(tr), "n_test": len(te),
                     "train_hit_rate": float(ytr.mean()), "test_hit_rate": float(yte.mean()),
                     "test_majority_baseline": float(max(yte.mean(), 1 - yte.mean()))}

    summary = {}
    for name, r in res.items():
        summary[name] = {
            "random_mean": float(np.mean(r["random"])), "random_sd": float(np.std(r["random"])),
            "grouped_mean": float(np.mean(r["grouped"])), "grouped_sd": float(np.std(r["grouped"])),
            "inflation_points": float(100 * (np.mean(r["random"]) - np.mean(r["grouped"]))),
            "temporal": temporal[name],
        }
    out = {"leakage_facts": leak, "summary": summary, "per_seed": res,
           "temporal_split": temporal_info, "n_seeds": n_seeds}
    out["figures"] = [figure(summary, temporal_info)]
    return out


def figure(summary, temporal_info):
    import matplotlib.pyplot as plt

    names = list(summary)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        s = summary[n]
        ax.plot([s["grouped_mean"], s["random_mean"]], [yi, yi], color=COLORS["axis"], lw=2,
                zorder=1)
    ax.scatter([summary[n]["temporal"] for n in names], y, s=50, color=COLORS["third"],
               label="Temporal split (train <=2006, test >=2007)", zorder=3, marker="D",
               edgecolors=COLORS["surface"], linewidths=1.5)
    ax.scatter([summary[n]["grouped_mean"] for n in names], y, s=60, color=COLORS["ours"],
               label="Artist-grouped split", zorder=3, edgecolors=COLORS["surface"], linewidths=1.5)
    ax.scatter([summary[n]["random_mean"] for n in names], y, s=60, color=COLORS["paper"],
               label="Random row split", zorder=3, edgecolors=COLORS["surface"], linewidths=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\nrandom - grouped: {summary[n]['inflation_points']:+.1f} pts"
                        for n in names], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Test accuracy (mean of 20 seeds; temporal is a single split)")
    ax.set_title("Random vs artist-grouped split: no meaningful inflation on this dataset")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, fontsize=8.5)
    lo = min(min(summary[n]["temporal"], summary[n]["grouped_mean"]) for n in names)
    hi = max(summary[n]["random_mean"] for n in names)
    ax.set_xlim(lo - 0.01, max(hi, max(summary[n]["temporal"] for n in names)) + 0.035)
    return plotting.save(fig, "09_split_inflation")
