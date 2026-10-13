"""Experiment family 1: the paper's five supervised models (Table 2 / Figure 8).

Part A reproduces the authors' protocol exactly: their CSV in file order, the first 2,416 rows
for training and the remaining 805 for validation, their hand-picked feature subsets, raw
(unscaled) features and the same model settings.

Part B runs the same fixed model settings under the fair protocol (artist-grouped split of the
de-duplicated data, test set used once) so that the two can be compared directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .. import data, plotting
from ..config import AUTHOR_SUBSETS, COLORS, PAPER, SEED
from ..metrics import group_bootstrap_ci, scores
from ..models import GDA, LegacySVC, PaperNN
from .common import evaluate, labels, paper_lr


def _with_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


def _svm_buggy_train_accuracy(pred, y):
    """svm.py compares an (m,1) prediction array with an (m,) label array, so numpy broadcasts an
    m-by-m matrix and the 'accuracy' becomes the share of matching (prediction, label) pairs over
    all pairs of songs. This explains the paper's SVM training accuracies of about 0.55."""
    return float(np.mean(pred.reshape(-1, 1) == y.reshape(1, -1)))


def _paper_models():
    """name -> (factory, feature subset, add intercept column)."""
    return {
        "LR": (paper_lr, AUTHOR_SUBSETS["lr"], False),
        "GDA": (GDA, AUTHOR_SUBSETS["gda"], False),
        "SVM linear": (lambda: LegacySVC(kernel="linear"), AUTHOR_SUBSETS["svm"], True),
        "SVM RBF": (lambda: LegacySVC(kernel="rbf"), AUTHOR_SUBSETS["svm"], True),
        "SVM polynomial": (lambda: LegacySVC(kernel="poly"), AUTHOR_SUBSETS["svm"], True),
        "DT": (lambda: DecisionTreeClassifier(random_state=SEED), AUTHOR_SUBSETS["svm"], False),
    }


def reproduce_paper_protocol():
    raw = data.load_raw_dated()
    tr, va = data.paper_split(raw)
    out = {}
    for name, (factory, feats, icpt) in _paper_models().items():
        Xtr, ytr = data.xy(tr, feats)
        Xva, yva = data.xy(va, feats)
        if icpt:
            Xtr, Xva = _with_intercept(Xtr), _with_intercept(Xva)
        m = factory().fit(Xtr, ytr)
        res = {"train": evaluate(m, Xtr, ytr), "val": evaluate(m, Xva, yva),
               "features": feats}
        if name.startswith("SVM"):
            res["train_acc_as_computed_by_authors"] = _svm_buggy_train_accuracy(m.predict(Xtr), ytr)
        out[name] = res

    # Neural network: project_nn.py uses the undated file and an 80/20 split in file order,
    # and reports the best validation accuracy seen during training.
    und = data.load_raw_undated()
    n_tr = int(np.ceil(0.8 * len(und)))
    tr, va = und.iloc[:n_tr], und.iloc[n_tr:]
    feats = AUTHOR_SUBSETS["nn"]
    Xtr, ytr = data.xy(tr, feats)
    Xva, yva = data.xy(va, feats)
    nn = PaperNN(feats, epochs=18000).fit(Xtr, ytr, Xva, yva)
    h = nn.history
    best = int(np.argmax(h["val_acc"]))
    out["NN"] = {
        "train": evaluate(nn, Xtr, ytr), "val": evaluate(nn, Xva, yva), "features": feats,
        "val_acc_best_epoch_as_reported": h["val_acc"][best],
        "best_epoch": h["epoch"][best],
        "note": "val accuracy here is at the last epoch; the paper reports the maximum over epochs",
    }
    return out


def fair_protocol():
    """Same fixed models, artist-grouped split, evaluated once on the held-out artists."""
    tr, te = data.main_split()
    out = {}
    for name, (factory, feats, icpt) in _paper_models().items():
        Xtr, ytr = data.xy(tr, feats)
        Xte, yte = data.xy(te, feats)
        if icpt:
            Xtr, Xte = _with_intercept(Xtr), _with_intercept(Xte)
        m = factory().fit(Xtr, ytr)
        res = {"train": evaluate(m, Xtr, ytr), "test": evaluate(m, Xte, yte)}
        res["test_acc_ci95"] = group_bootstrap_ci(yte, m.predict(Xte), te.Group)
        out[name] = res
    feats = AUTHOR_SUBSETS["nn"]
    Xtr, ytr = data.xy(tr, feats)
    Xte, yte = data.xy(te, feats)
    nn = PaperNN(feats, epochs=18000).fit(Xtr, ytr)  # fixed epoch count, no peeking at test
    out["NN"] = {"train": evaluate(nn, Xtr, ytr), "test": evaluate(nn, Xte, yte),
                 "test_acc_ci95": group_bootstrap_ci(yte, nn.predict(Xte), te.Group)}
    # Reference point: the artist-score rule alone.
    rule = (te.ArtistScore.to_numpy() > 0.5).astype(int)
    out["Artist score rule"] = {"test": scores(labels(te), rule, te.ArtistScore.to_numpy()),
                                "test_acc_ci95": group_bootstrap_ci(labels(te), rule, te.Group)}
    return out


def figure(rep, fair):
    import matplotlib.pyplot as plt

    names = ["LR", "GDA", "SVM linear", "SVM RBF", "SVM polynomial", "DT", "NN"]
    paper_val = [PAPER["table2"][n]["val_acc"] for n in names]
    ours_val = [rep[n]["val"]["accuracy"] if n != "NN"
                else rep[n]["val_acc_best_epoch_as_reported"] for n in names]
    fair_test = [fair[n]["test"]["accuracy"] for n in names]
    err = {"Ours, fair artist-grouped test": [fair[n]["test_acc_ci95"] for n in names]}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    plotting.grouped_bars(
        ax, [n.replace("SVM ", "SVM\n") for n in names],
        {"Paper (validation)": paper_val, "Ours, paper protocol (validation)": ours_val,
         "Ours, fair artist-grouped test": fair_test},
        {"Paper (validation)": COLORS["paper"], "Ours, paper protocol (validation)": COLORS["ours"],
         "Ours, fair artist-grouped test": COLORS["third"]},
        ylim=(0.4, 0.9), err=err, label_series=("Ours, fair artist-grouped test",))
    rule = fair["Artist score rule"]["test"]["accuracy"]
    ax.axhline(rule, color=COLORS["muted"], lw=1.2,
               label=f"Artist score rule alone, fair test ({rule:.1%})")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Supervised models: paper vs our reproduction vs fair evaluation")
    return plotting.save(fig, "01_supervised_models")


def run():
    rep = reproduce_paper_protocol()
    fair = fair_protocol()
    return {"paper_protocol": rep, "fair_protocol": fair, "figures": [figure(rep, fair)]}
