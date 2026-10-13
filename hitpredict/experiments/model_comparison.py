"""Experiment family 7: a range of models, from trivial baselines to ensembles, tuned fairly.

Every model is tuned with 5-fold artist-grouped cross-validation on the training artists only. The
held-out test artists are scored once per model after tuning; the 'best model' is the one with the
highest cross-validation accuracy, never the highest test accuracy.
"""

from __future__ import annotations

import re
import time

import joblib
import numpy as np

from .. import data, plotting
from ..config import COLORS, MODEL_DIR, PAPER, PAPER_FEATURES
from ..metrics import group_bootstrap_ci, scores
from ..models import model_zoo
from .common import tune


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run(features=PAPER_FEATURES, save_models: bool = True):
    tr, te = data.main_split()
    Xtr, ytr = data.xy(tr, features)
    Xte, yte = data.xy(te, features)
    groups = tr.Group.to_numpy()
    rows = {}
    for name, est, grid, family in model_zoo(features):
        t0 = time.time()
        search = tune(est, grid, Xtr, ytr, groups)
        i = search.best_index_
        model = search.best_estimator_
        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:, 1]
        rows[name] = {
            "family": family,
            "best_params": {k: (v if isinstance(v, (int, float, str)) or v is None else str(v))
                            for k, v in search.best_params_.items()},
            "cv_acc_mean": float(search.cv_results_["mean_test_score"][i]),
            "cv_acc_sd": float(search.cv_results_["std_test_score"][i]),
            "train_acc": float(np.mean(model.predict(Xtr) == ytr)),
            "test": scores(yte, pred, prob),
            "test_acc_ci95": group_bootstrap_ci(yte, pred, te.Group),
            "fit_seconds": round(time.time() - t0, 1),
        }
        if save_models:
            (MODEL_DIR / "zoo").mkdir(parents=True, exist_ok=True)
            joblib.dump({"name": name, "model": model, "features": list(features)},
                        MODEL_DIR / "zoo" / f"{slug(name)}.joblib")
        print(f"  {name:36s} cv {rows[name]['cv_acc_mean']:.3f}  test "
              f"{rows[name]['test']['accuracy']:.3f}  ({rows[name]['fit_seconds']}s)")
    best = max(rows, key=lambda n: rows[n]["cv_acc_mean"])
    return {"features": list(features), "n_train": len(tr), "n_test": len(te), "models": rows,
            "best_by_cv": best, "figures": [figure(rows)]}


def figure(rows):
    import matplotlib.pyplot as plt

    names = sorted(rows, key=lambda n: rows[n]["cv_acc_mean"])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    test = np.array([rows[n]["test"]["accuracy"] for n in names])
    ci = np.array([rows[n]["test_acc_ci95"] for n in names])
    cv = np.array([rows[n]["cv_acc_mean"] for n in names])
    ax.barh(y, test, 0.6, color=COLORS["ours"], label="Test accuracy (95% CI, held-out artists)",
            xerr=np.vstack([test - ci[:, 0], ci[:, 1] - test]),
            error_kw={"ecolor": COLORS["ink2"], "elinewidth": 1, "capsize": 2})
    ax.scatter(cv, y, color=COLORS["paper"], zorder=3, s=36, label="Cross-validation accuracy",
               edgecolors=COLORS["surface"], linewidths=1.5)
    for yi, v in zip(y, test):
        ax.annotate(f"{v:.1%}", (0.505, yi), va="center", fontsize=8, color=COLORS["surface"])
    paper = PAPER["table2"]["LR"]["val_acc"]
    ax.axvline(paper, color=COLORS["muted"], lw=1)
    ax.annotate(f"Paper's reported best (LR, {paper:.1%})", (paper, len(names) - 0.4), xytext=(3, 0),
                textcoords="offset points", fontsize=8, color=COLORS["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.5, 0.82)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Model comparison, tuned with grouped CV (sorted by CV accuracy)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2, fontsize=8.5)
    return plotting.save(fig, "10_model_comparison")
