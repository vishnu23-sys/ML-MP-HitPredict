"""Experiment family 8: our improvements over the paper, added one at a time.

All steps are trained on the training artists and scored on the same held-out test artists. Every
choice (hyperparameters, which ensemble to keep, the decision threshold) is made with grouped
cross-validation on the training artists. The list of steps was fixed before looking at test scores.

  B0  Paper baseline: unregularised logistic regression, the authors' 7-feature subset, trained only
      on songs with a release date (as the paper's main experiments were).
  I1  More data: also train on the 813 labelled songs whose Spotify release date has no month. The
      paper left them out of its LR, GDA and SVM experiments, although the date is never a model
      input (only its neural network used them).
  I2  Better features: all 12 Spotify fields, log10 of the four skewed features, Key dropped and
      ArtistScore x audio interactions, with an L2 penalty tuned by grouped CV.
  I3  Nonlinear models: tuned gradient boosting and a soft-voting ensemble (engineered LR +
      gradient boosting + RBF SVM); the variant with the best grouped-CV accuracy is kept.
  I4  Decision threshold for the use case: the paper notes that false positives are costly for a
      label. We pick, on out-of-fold training predictions, the lowest threshold that reaches 80%
      precision, then report test precision and recall at that threshold.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .. import data, plotting
from ..config import ALL_FEATURES, AUTHOR_SUBSETS, COLORS, MODEL_DIR, SEED
from ..metrics import group_bootstrap_ci, paired_group_bootstrap, scores
from ..models import HitFeatures, engineered_lr
from .common import grouped_cv, oof_proba, paper_lr, tune

TARGET_PRECISION = 0.80


def _cv_acc(model, X, y, g):
    accs = []
    for a, b in grouped_cv().split(X, y, g):
        accs.append(np.mean(model.fit(X[a], y[a]).predict(X[b]) == y[b]))
    return float(np.mean(accs)), float(np.std(accs))


def _step(name, model, Xtr, ytr, gtr, Xte, yte, te, cv=None):
    if cv is None:
        cv = _cv_acc(model, Xtr, ytr, gtr)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {"name": name, "cv_acc_mean": cv[0], "cv_acc_sd": cv[1],
            "test": scores(yte, pred, model.predict_proba(Xte)[:, 1]),
            "test_acc_ci95": group_bootstrap_ci(yte, pred, te.Group), "_pred": pred,
            "_model": model}


def run(save_models: bool = True):
    tr, te = data.main_split()
    yte = te.Label.to_numpy()
    steps = []

    # B0: paper baseline, dated training songs only
    feats = AUTHOR_SUBSETS["lr"]
    trd = tr[tr.Dated]
    steps.append(_step("B0 Paper baseline (LR, 7 features, dated songs)", paper_lr(),
                       *data.xy(trd, feats), trd.Group.to_numpy(), *data.xy(te, feats), te))

    # I1: + the undated songs
    steps.append(_step("I1 + all labelled songs", paper_lr(), *data.xy(tr, feats),
                       tr.Group.to_numpy(), *data.xy(te, feats), te))

    # I2: + engineered features, tuned L2 logistic regression
    Xtr, ytr = data.xy(tr, ALL_FEATURES)
    Xte, _ = data.xy(te, ALL_FEATURES)
    g = tr.Group.to_numpy()
    lr_search = tune(engineered_lr(ALL_FEATURES), {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3]},
                     Xtr, ytr, g)
    i = lr_search.best_index_
    lr_cv = (float(lr_search.cv_results_["mean_test_score"][i]),
             float(lr_search.cv_results_["std_test_score"][i]))
    steps.append(_step("I2 + engineered features (tuned LR)", lr_search.best_estimator_,
                       Xtr, ytr, g, Xte, yte, te, cv=lr_cv))

    # I3: nonlinear candidates, keep the best by grouped CV
    hgb_search = tune(HistGradientBoostingClassifier(random_state=SEED),
                      {"learning_rate": [0.03, 0.1], "max_depth": [2, 3],
                       "max_leaf_nodes": [7, 15], "l2_regularization": [0.0, 1.0],
                       "max_iter": [100, 300]}, Xtr, ytr, g)
    svm_search = tune(Pipeline([("features", HitFeatures(ALL_FEATURES, interactions=False)),
                                ("scale", StandardScaler()),
                                ("clf", SVC(probability=True, random_state=SEED))]),
                      {"clf__C": [0.3, 1, 3], "clf__gamma": ["scale", 0.03]}, Xtr, ytr, g)
    hgb = hgb_search.best_estimator_
    ens = VotingClassifier([("lr", lr_search.best_estimator_), ("hgb", hgb),
                            ("svm", svm_search.best_estimator_)], voting="soft")
    candidates = {
        "gradient boosting": (hgb, (float(hgb_search.best_score_),
                                    float(hgb_search.cv_results_["std_test_score"][
                                        hgb_search.best_index_]))),
        "soft-voting ensemble": (ens, _cv_acc(ens, Xtr, ytr, g)),
    }
    chosen = max(candidates, key=lambda k: candidates[k][1][0])
    i3_candidates = {k: {"cv_acc_mean": v[1][0], "cv_acc_sd": v[1][1]}
                     for k, v in candidates.items()}
    model, cv = candidates[chosen]
    steps.append(_step(f"I3 + nonlinear model ({chosen})", model, Xtr, ytr, g, Xte, yte, te,
                       cv=cv))

    # Best model by CV across I2 and I3 (the deployable model)
    final = max(steps[2:], key=lambda s: s["cv_acc_mean"])

    # I4: threshold for 80% precision, chosen on out-of-fold training predictions
    oof = oof_proba(final["_model"], Xtr, ytr, g)
    thr = 0.5
    for t in np.round(np.arange(0.30, 0.951, 0.01), 2):
        hit = oof >= t
        if hit.sum() and ytr[hit].mean() >= TARGET_PRECISION:
            thr = float(t)
            break
    final["_model"].fit(Xtr, ytr)
    prob = final["_model"].predict_proba(Xte)[:, 1]
    at_thr = scores(yte, (prob >= thr).astype(int), prob)
    at_half = scores(yte, (prob >= 0.5).astype(int), prob)
    threshold = {"target_precision": TARGET_PRECISION, "threshold_from_oof": thr,
                 "oof_precision_at_threshold": float(ytr[oof >= thr].mean()),
                 "test_at_0.5": at_half, "test_at_threshold": at_thr}

    # Where the final model goes wrong on the test songs, split by artist score
    pred = (prob >= 0.5).astype(int)
    a_s = te.ArtistScore.to_numpy()
    errors = {}
    for val, name in [(0, "new_artists"), (1, "established_artists")]:
        m = a_s == val
        errors[name] = {
            "songs": int(m.sum()), "hits": int(yte[m].sum()),
            "accuracy": float(np.mean(pred[m] == yte[m])),
            "missed_hits": int(np.sum((pred == 0) & (yte == 1) & m)),
            "false_alarms": int(np.sum((pred == 1) & (yte == 0) & m)),
            "hits_found_share": float(np.mean(pred[m & (yte == 1)] == 1)) if yte[m].sum() else None,
        }

    # Paired comparisons on the test artists
    b0 = steps[0]["_pred"]
    for k, s in enumerate(steps):
        s["vs_baseline"] = dict(zip(["diff", "lo", "hi", "p_not_better"],
                                    paired_group_bootstrap(yte, s["_pred"], b0, te.Group)))
        if k:
            s["vs_previous"] = dict(zip(["diff", "lo", "hi", "p_not_better"],
                                        paired_group_bootstrap(yte, s["_pred"],
                                                               steps[k - 1]["_pred"], te.Group)))

    if save_models:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"name": final["name"], "model": final["_model"], "features": ALL_FEATURES,
                     "threshold_precision80": thr, "cv_acc": final["cv_acc_mean"],
                     "test_acc": final["test"]["accuracy"]}, MODEL_DIR / "best_model.joblib")
        base = steps[0]["_model"]
        joblib.dump({"name": "Paper baseline LR (reproduced)", "model": base, "features": feats,
                     "threshold_precision80": None, "cv_acc": steps[0]["cv_acc_mean"],
                     "test_acc": steps[0]["test"]["accuracy"]}, MODEL_DIR / "paper_lr.joblib")
        te.to_csv(MODEL_DIR / "test_songs.csv", index=False)

    clean = [{k: v for k, v in s.items() if not k.startswith("_")} for s in steps]
    out = {"steps": clean, "i3_candidates": i3_candidates, "i3_chosen": chosen,
           "lr_best_C": lr_search.best_params_["clf__C"],
           "hgb_best_params": hgb_search.best_params_, "svm_best_params": svm_search.best_params_,
           "final_model": final["name"], "threshold": threshold,
           "final_model_errors_by_artist_score": errors,
           "lr_coefficients": _lr_coefficients(lr_search.best_estimator_),
           "danceability_logodds_per_sd": _danceability_effect(lr_search.best_estimator_,
                                                               tr.Danceability.std())}
    out["figures"] = [figure(clean, threshold)]
    return out


def _lr_coefficients(pipe):
    names = pipe.named_steps["features"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coef))
    return [{"feature": str(names[i]), "coef_per_sd": float(coef[i])} for i in order]


def _danceability_effect(pipe, sd):
    """Change in log-odds of a hit for +1 training SD of danceability, by artist score."""
    names = list(pipe.named_steps["features"].get_feature_names_out())
    w = pipe.named_steps["clf"].coef_[0] / pipe.named_steps["scale"].scale_
    main, inter = w[names.index("Danceability")], w[names.index("ArtistScore*Danceability")]
    return {"new_artists": float(main * sd), "established_artists": float((main + inter) * sd)}


def figure(steps, threshold):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    labels = [s["name"].split(" ", 1)[0] + "\n" + s["name"].split(" ", 1)[1].replace(" (", "\n(")
              for s in steps]
    plotting.grouped_bars(
        ax, labels,
        {"Grouped CV on training artists": [s["cv_acc_mean"] for s in steps],
         "Held-out test artists": [s["test"]["accuracy"] for s in steps]},
        {"Grouped CV on training artists": COLORS["paper"], "Held-out test artists": COLORS["ours"]},
        ylim=(0.6, 0.82), err={"Held-out test artists": [s["test_acc_ci95"] for s in steps]},
        label_series=("Held-out test artists",))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(0.02))
    ax.set_title("Improvements, added one at a time", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    a, b = threshold["test_at_0.5"], threshold["test_at_threshold"]
    plotting.grouped_bars(
        ax, ["Precision", "Recall", "Accuracy"],
        {"Threshold 0.5": [a["precision"], a["recall"], a["accuracy"]],
         f"Threshold {threshold['threshold_from_oof']:.2f} (80% precision on OOF)":
             [b["precision"], b["recall"], b["accuracy"]]},
        {"Threshold 0.5": COLORS["muted"],
         f"Threshold {threshold['threshold_from_oof']:.2f} (80% precision on OOF)": COLORS["ours"]},
        ylim=(0, 1), fmt="{:.0%}",
        label_series=(f"Threshold {threshold['threshold_from_oof']:.2f} (80% precision on OOF)",))
    ax.set_title("I4: operating point for a cautious label (test set)", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return plotting.save(fig, "11_improvements")
