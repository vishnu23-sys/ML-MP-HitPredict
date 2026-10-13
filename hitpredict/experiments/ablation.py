"""Experiment family 4: feature ablation / forward selection (paper Section 5, Figure 9).

The paper ranks features by greedy forward selection, scoring every candidate subset on the
validation set, keeps the subset with the highest validation accuracy and then reports that same
validation accuracy as the model's performance. Selecting and reporting on the same data is
optimistic. We (a) reproduce the paper's ranking, (b) redo the selection with grouped
cross-validation on the training artists only, and (c) measure the optimism directly over 20
random train/validation/test splits.
"""

from __future__ import annotations

import numpy as np

from .. import data, plotting
from ..config import AUTHOR_SUBSETS, COLORS, PAPER, PAPER_FEATURES, SEED
from ..models import PaperNN
from .common import grouped_cv, paper_lr


def forward_selection(features, score_fn):
    """Greedy forward selection over all features. Returns [(feature, score), ...] in order."""
    chosen, remaining, path = [], list(features), []
    while remaining:
        best = max(remaining, key=lambda f: score_fn(chosen + [f]))
        chosen.append(best)
        remaining.remove(best)
        path.append((best, float(score_fn(chosen))))
    return path


def best_prefix(path):
    k = int(np.argmax([s for _, s in path])) + 1
    return [f for f, _ in path[:k]], path[k - 1][1]


def _val_scorer(model_fn, tr, va):
    cache = {}

    def score(feats):
        key = tuple(feats)
        if key not in cache:
            Xtr, ytr = data.xy(tr, feats)
            Xva, yva = data.xy(va, feats)
            cache[key] = float(np.mean(model_fn().fit(Xtr, ytr).predict(Xva) == yva))
        return cache[key]

    return score


def _cv_scorer(model_fn, tr):
    g = tr.Group.to_numpy()
    folds = list(grouped_cv(5, SEED).split(tr, tr.Label, g))
    cache = {}

    def score(feats):
        key = tuple(feats)
        if key not in cache:
            X, y = data.xy(tr, feats)
            cache[key] = float(np.mean([np.mean(model_fn().fit(X[a], y[a]).predict(X[b]) == y[b])
                                        for a, b in folds]))
        return cache[key]

    return score


def paper_protocol(nn_epochs: int = 6000):
    raw = data.load_raw_dated()
    tr, va = data.paper_split(raw)
    lr_path = forward_selection(PAPER_FEATURES, _val_scorer(paper_lr, tr, va))

    # The NN ablation in project_nn.py uses the undated file and an 80/20 split in file order,
    # and scores each subset by its best validation accuracy over training epochs.
    und = data.load_raw_undated()
    n_tr = int(np.ceil(0.8 * len(und)))
    ntr, nva = und.iloc[:n_tr], und.iloc[n_tr:]
    cache = {}

    def nn_score(feats):
        key = tuple(feats)
        if key not in cache:
            Xtr, ytr = data.xy(ntr, feats)
            Xva, yva = data.xy(nva, feats)
            h = PaperNN(feats, epochs=nn_epochs).fit(Xtr, ytr, Xva, yva).history
            cache[key] = max(h["val_acc"])
        return cache[key]

    nn_path = forward_selection(PAPER_FEATURES, nn_score)
    return {"LR": lr_path, "NN": nn_path, "nn_epochs_used": nn_epochs}


def fair_selection():
    tr, te = data.main_split()
    path = forward_selection(PAPER_FEATURES, _cv_scorer(paper_lr, tr))
    subset, cv_acc = best_prefix(path)
    out = {"cv_path": path, "selected": subset, "selected_cv_acc": cv_acc}
    for name, feats in [("selected_subset", subset), ("all_10_features", PAPER_FEATURES),
                        ("authors_lr_subset", AUTHOR_SUBSETS["lr"])]:
        Xtr, ytr = data.xy(tr, feats)
        Xte, yte = data.xy(te, feats)
        out[f"test_acc_{name}"] = float(np.mean(paper_lr().fit(Xtr, ytr).predict(Xte) == yte))
    return out


def selection_optimism(n_repeats: int = 20):
    """Repeat: artist-grouped 60/20/20 split, select features on validation (paper style),
    then compare the reported validation accuracy with accuracy on untouched test artists."""
    df = data.load_master()
    rows = []
    for r in range(n_repeats):
        rest, test = data.group_split(df, 0.2, seed=SEED + r)
        train, val = data.group_split(rest, 0.25, seed=SEED + r)
        subset, val_acc = best_prefix(forward_selection(PAPER_FEATURES,
                                                        _val_scorer(paper_lr, train, val)))
        Xtr, ytr = data.xy(train, subset)
        Xte, yte = data.xy(test, subset)
        test_acc = float(np.mean(paper_lr().fit(Xtr, ytr).predict(Xte) == yte))
        Xtr, ytr = data.xy(train, PAPER_FEATURES)
        Xva, yva = data.xy(val, PAPER_FEATURES)
        Xte, yte = data.xy(test, PAPER_FEATURES)
        m = paper_lr().fit(Xtr, ytr)
        rows.append({"seed": SEED + r, "n_selected": len(subset), "selected_val_acc": val_acc,
                     "selected_test_acc": test_acc,
                     "all_features_val_acc": float(np.mean(m.predict(Xva) == yva)),
                     "all_features_test_acc": float(np.mean(m.predict(Xte) == yte))})
    keys = ["selected_val_acc", "selected_test_acc", "all_features_val_acc", "all_features_test_acc"]
    summary = {k: float(np.mean([row[k] for row in rows])) for k in keys}
    summary["optimism_selected"] = summary["selected_val_acc"] - summary["selected_test_acc"]
    summary["optimism_all_features"] = (summary["all_features_val_acc"]
                                        - summary["all_features_test_acc"])
    return {"repeats": rows, "summary": summary}


def figure(paper_rep, opt):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, model in zip(axes[:2], ["LR", "NN"]):
        ours = paper_rep[model]
        paper = PAPER["ablation"][model]
        k = np.arange(1, len(ours) + 1)
        ax.plot(k, [s for _, s in paper], color=COLORS["paper"], marker="o", ms=4, label="Paper")
        ax.plot(k, [s for _, s in ours], color=COLORS["ours"], marker="o", ms=4,
                label="Ours (paper protocol)")
        ax.set_xticks(k)
        ax.set_xticklabels([f"{i}. {f}" for i, (f, _) in zip(k, ours)], rotation=40, ha="right",
                           fontsize=8)
        ax.set_xlabel("Feature added at each step (our order; the paper's order is in the README)")
        ax.set_ylabel("Validation accuracy")
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        ax.margins(y=0.15)
        ax.set_title(f"{model}: forward selection on the validation set", fontsize=11)
        ax.legend(loc="lower right", fontsize=8)
    s = opt["summary"]
    plotting.grouped_bars(
        axes[2], ["Features chosen\non validation", "All 10 features"],
        {"Reported (validation)": [s["selected_val_acc"], s["all_features_val_acc"]],
         "Fresh test artists": [s["selected_test_acc"], s["all_features_test_acc"]]},
        {"Reported (validation)": COLORS["paper"], "Fresh test artists": COLORS["ours"]},
        ylim=(0.6, 0.8), label_series=("Reported (validation)", "Fresh test artists"))
    axes[2].set_title("Selection optimism (mean of 20 splits)", fontsize=11)
    axes[2].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    axes[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return plotting.save(fig, "06_feature_ablation")


def run():
    rep = paper_protocol()
    fair = fair_selection()
    opt = selection_optimism()
    return {"paper_protocol": rep, "fair_selection": fair, "selection_optimism": opt,
            "figures": [figure(rep, opt)]}
