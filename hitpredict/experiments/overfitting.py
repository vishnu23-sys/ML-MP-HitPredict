"""Experiment family 3: the paper's overfitting studies.

* Bagging (paper Figure 5 / Table): SVM linear, SVM polynomial and decision tree trained on
  random subsets of 200 or 500 songs, 10 or 50 bags. The authors average the 0/1 votes and then
  test `average == label`, which only counts a song as correct when every bag agrees. We report
  that number and the real majority-vote accuracy.
* Decision-tree depth (paper Figure 6): max depth = k * n_features.
* Neural-network epochs (paper Figure 7): validation accuracy against epochs. The paper picks the
  epoch with the best validation accuracy and reports that same accuracy, which is optimistic.
  We show the size of that optimism with a proper three-way split.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .. import data, plotting
from ..config import AUTHOR_SUBSETS, COLORS, PAPER, SEED
from ..metrics import scores
from ..models import LegacySVC, PaperNN
from .common import grouped_cv


def _bag_votes(factory, Xtr, ytr, Xva, n_samples, n_bags, rng):
    """Share of bags voting 'hit' for every training and validation song."""
    from joblib import Parallel, delayed

    bags = [rng.choice(len(Xtr), n_samples, replace=False) for _ in range(n_bags)]

    def one(idx):
        m = factory().fit(Xtr[idx], ytr[idx])
        return m.predict(Xtr), m.predict(Xva)

    preds = Parallel(n_jobs=-1)(delayed(one)(idx) for idx in bags)
    vt = np.mean([p[0] for p in preds], axis=0)
    vv = np.mean([p[1] for p in preds], axis=0)
    return vt, vv


def bagging():
    raw = data.load_raw_dated()
    tr, va = data.paper_split(raw)
    feats = AUTHOR_SUBSETS["svm"]
    Xtr, ytr = data.xy(tr, feats)
    Xva, yva = data.xy(va, feats)
    factories = {
        "SVM linear": lambda: LegacySVC(kernel="linear"),
        "SVM polynomial": lambda: LegacySVC(kernel="poly"),
        "DT": lambda: DecisionTreeClassifier(random_state=SEED),
    }
    rows = []
    for name, fac in factories.items():
        for n_samples in (200, 500):
            for n_bags in (10, 50):
                rng = np.random.default_rng(SEED)
                vt, vv = _bag_votes(fac, Xtr, ytr, Xva, n_samples, n_bags, rng)
                p_train, p_val = PAPER["bagging"][(name, n_samples, n_bags)]
                rows.append({
                    "model": name, "samples_per_bag": n_samples, "bags": n_bags,
                    "paper_train_acc": p_train, "paper_val_acc": p_val,
                    "authors_metric_train": float(np.mean(vt == ytr)),
                    "authors_metric_val": float(np.mean(vv == yva)),
                    "majority_vote_train": float(np.mean((vt > 0.5) == ytr)),
                    "majority_vote_val": float(np.mean((vv > 0.5) == yva)),
                })
    return rows


def tree_depth():
    raw = data.load_raw_dated()
    tr, va = data.paper_split(raw)
    feats = AUTHOR_SUBSETS["svm"]
    n = len(feats)
    Xtr, ytr = data.xy(tr, feats)
    Xva, yva = data.xy(va, feats)
    paper_rows = []
    for k, (p_tr, p_va) in PAPER["dt_depth"].items():
        m = DecisionTreeClassifier(max_depth=int(k * n), random_state=SEED).fit(Xtr, ytr)
        paper_rows.append({"depth_multiple": k, "max_depth": int(k * n),
                           "train_acc": float(m.score(Xtr, ytr)), "val_acc": float(m.score(Xva, yva)),
                           "paper_train_acc": p_tr, "paper_val_acc": p_va})

    # Fair: choose the depth by grouped CV on the training artists, evaluate once on test.
    trf, tef = data.main_split()
    Xtr, ytr = data.xy(trf, feats)
    Xte, yte = data.xy(tef, feats)
    g = trf.Group.to_numpy()
    curve = []
    for depth in range(1, 21):
        tr_acc, cv_acc = [], []
        for a, b in grouped_cv().split(Xtr, ytr, g):
            m = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(Xtr[a], ytr[a])
            tr_acc.append(m.score(Xtr[a], ytr[a]))
            cv_acc.append(m.score(Xtr[b], ytr[b]))
        curve.append({"max_depth": depth, "train_acc": float(np.mean(tr_acc)),
                      "cv_acc": float(np.mean(cv_acc)), "cv_sd": float(np.std(cv_acc))})
    best = max(curve, key=lambda r: r["cv_acc"])
    m = DecisionTreeClassifier(max_depth=best["max_depth"], random_state=SEED).fit(Xtr, ytr)
    return {"paper_protocol": paper_rows, "fair_cv_curve": curve,
            "fair_chosen_depth": best["max_depth"],
            "fair_test": scores(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1])}


def nn_epochs(epochs: int = 30000):
    # Paper protocol: undated file, 80/20 split in file order, with and without the penalty term.
    und = data.load_raw_undated()
    n_tr = int(np.ceil(0.8 * len(und)))
    feats = AUTHOR_SUBSETS["nn"]
    Xtr, ytr = data.xy(und.iloc[:n_tr], feats)
    Xva, yva = data.xy(und.iloc[n_tr:], feats)
    curves = {}
    for reg in (True, False):
        nn = PaperNN(feats, epochs=epochs, regularise=reg).fit(Xtr, ytr, Xva, yva)
        curves["with_penalty" if reg else "without_penalty"] = nn.history

    # Fair: pick the epoch on an inner validation split of the training artists, then test once.
    trf, tef = data.main_split()
    inner_tr, inner_va = data.group_split(trf, test_size=0.2, seed=SEED)
    Xi, yi = data.xy(inner_tr, feats)
    Xv, yv = data.xy(inner_va, feats)
    Xt, yt = data.xy(trf, feats)
    Xte, yte = data.xy(tef, feats)
    inner = PaperNN(feats, epochs=epochs).fit(Xi, yi, Xv, yv).history
    best_epoch = inner["epoch"][int(np.argmax(inner["val_acc"]))]
    final = PaperNN(feats, epochs=best_epoch + 1).fit(Xt, yt)
    # For comparison only (never used for a decision): the curve on the test set itself.
    peek = PaperNN(feats, epochs=epochs).fit(Xt, yt, Xte, yte).history
    return {
        "paper_protocol": {k: {"best_val_acc": float(max(v["val_acc"])),
                               "best_epoch": v["epoch"][int(np.argmax(v["val_acc"]))],
                               "final_val_acc": v["val_acc"][-1]} for k, v in curves.items()},
        "curves": curves,
        "fair": {"epoch_chosen_on_inner_validation": best_epoch,
                 "inner_val_acc_at_choice": float(max(inner["val_acc"])),
                 "test_acc": float(np.mean(final.predict(Xte) == yte)),
                 "max_test_acc_over_epochs_if_peeking": float(max(peek["val_acc"])),
                 "peek_curve": peek},
    }


def figures(bag, depth, nn):
    import matplotlib.pyplot as plt

    paths = []
    # Bagging
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, name in zip(axes, ["SVM linear", "SVM polynomial", "DT"]):
        rows = [r for r in bag if r["model"] == name]
        cats = [f"{r['samples_per_bag']} songs\n{r['bags']} bags" for r in rows]
        plotting.grouped_bars(
            ax, cats,
            {"Paper (validation)": [r["paper_val_acc"] for r in rows],
             "Authors' metric (ours)": [r["authors_metric_val"] for r in rows],
             "Real majority vote": [r["majority_vote_val"] for r in rows]},
            {"Paper (validation)": COLORS["paper"], "Authors' metric (ours)": COLORS["third"],
             "Real majority vote": COLORS["ours"]}, ylim=(0, 1), fmt="{:.0%}",
            label_series=("Paper (validation)", "Real majority vote"))
        ax.set_title(name, fontsize=11)
        ax.tick_params(axis="x", labelsize=8)
        ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Bagging: the paper's low accuracies come from an averaging bug", x=0.01,
                 ha="left", fontweight="bold", color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower left", ncol=3, fontsize=9, bbox_to_anchor=(0.01, 0))
    paths.append(plotting.save(fig, "03_bagging"))

    # Tree depth
    fig, ax = plt.subplots(figsize=(7, 3.8))
    c = depth["fair_cv_curve"]
    d = [r["max_depth"] for r in c]
    ax.plot(d, [r["train_acc"] for r in c], color=COLORS["paper"], label="Training folds")
    ax.plot(d, [r["cv_acc"] for r in c], color=COLORS["ours"], label="Held-out artists (grouped CV)")
    cv = np.array([r["cv_acc"] for r in c])
    sd = np.array([r["cv_sd"] for r in c])
    ax.fill_between(d, cv - sd, cv + sd, color=COLORS["ours"], alpha=0.15, linewidth=0)
    ax.axvline(depth["fair_chosen_depth"], color=COLORS["muted"], lw=1)
    ax.annotate(f"chosen depth: {depth['fair_chosen_depth']}\n(a single split on artist score)"
                if depth["fair_chosen_depth"] == 1 else f"chosen depth: {depth['fair_chosen_depth']}",
                (depth["fair_chosen_depth"], 0.9), xytext=(5, 0), textcoords="offset points",
                fontsize=8, color=COLORS["ink2"])
    ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(2))
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Decision tree: depth chosen by grouped cross-validation")
    ax.legend(loc="center right")
    paths.append(plotting.save(fig, "04_tree_depth"))

    # NN epochs
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    ax = axes[0]
    cw, cn = nn["curves"]["with_penalty"], nn["curves"]["without_penalty"]
    ax.plot(cw["epoch"], cw["train_acc"], color=COLORS["muted"], lw=1.2, label="Training")
    ax.plot(cw["epoch"], cw["val_acc"], color=COLORS["ours"], label="Validation, with penalty")
    ax.plot(cn["epoch"], cn["val_acc"], color=COLORS["paper"], label="Validation, without penalty")
    ax.set_title("Paper protocol (80/20 split in file order)", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    pk = nn["fair"]["peek_curve"]
    ax.plot(pk["epoch"], pk["train_acc"], color=COLORS["muted"], lw=1.2, label="Training")
    ax.plot(pk["epoch"], pk["val_acc"], color=COLORS["ours"], label="Test artists")
    ep = nn["fair"]["epoch_chosen_on_inner_validation"]
    ax.axvline(ep, color=COLORS["third"], lw=1)
    ax.annotate(f"epoch chosen on inner validation: {ep}\n"
                f"test accuracy {nn['fair']['test_acc']:.1%} "
                f"(best possible by peeking {nn['fair']['max_test_acc_over_epochs_if_peeking']:.1%})",
                (ep, 0.62), xytext=(6, 0), textcoords="offset points", fontsize=8,
                color=COLORS["ink2"])
    ax.set_title("Fair protocol (artist-grouped split)", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.legend(loc="lower right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0.55, 0.85)
    fig.suptitle("Neural network accuracy against training epochs", x=0.01, ha="left",
                 fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    paths.append(plotting.save(fig, "05_nn_epochs"))
    return paths


def run():
    bag = bagging()
    depth = tree_depth()
    nn = nn_epochs()
    out = {"bagging": bag, "tree_depth": depth,
           "nn_epochs": {k: v for k, v in nn.items() if k != "curves"}}
    out["figures"] = figures(bag, depth, nn)
    # keep JSON small: drop the per-epoch curves after plotting
    out["nn_epochs"]["fair"].pop("peek_curve", None)
    return out
