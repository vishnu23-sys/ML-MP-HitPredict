"""Experiment family 2: unsupervised and semi-supervised EM (paper Section 3.2 and Figure 1).

The paper reports 50.1% (unsupervised EM) and 46.9% (semi-supervised EM). In both scripts the
predictions have shape (m,) and the labels shape (m, 1), so `np.mean(h == y)` broadcasts to an
m-by-m matrix and measures how often a prediction matches the label of a *different, random* song.
We compute that quantity too, to show it reproduces the paper, next to the real accuracy.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .. import data, plotting
from ..config import AUTHOR_SUBSETS, COLORS, SEED


def pairwise_match_rate(pred, y):
    """What the authors' accuracy line actually computes."""
    return float(np.mean(np.asarray(pred).reshape(-1, 1) == np.asarray(y).reshape(1, -1)))


def semi_supervised_em(X_unl, X_lab, z_lab, alpha=20.0, max_iter=1000, eps=1e-3, reg=1e-6,
                       seed=SEED):
    """Two-component Gaussian mixture EM where labelled points are fixed to their class and
    weighted by alpha (the CS229 problem-set formulation the authors followed)."""
    rng = np.random.default_rng(seed)
    K, n = 2, X_unl.shape[1]
    init = rng.integers(0, K, len(X_unl))
    mu = [X_unl[init == j].mean(0) for j in range(K)]
    sigma = [np.cov(X_unl[init == j].T, bias=True) + reg * np.eye(n) for j in range(K)]
    phi = np.array([np.mean(init == j) for j in range(K)])
    prev = None
    for _ in range(max_iter):
        dens = np.column_stack([phi[j] * multivariate_normal.pdf(X_unl, mu[j], sigma[j],
                                                                 allow_singular=True)
                                for j in range(K)])
        w = dens / np.clip(dens.sum(1, keepdims=True), 1e-300, None)
        for j in range(K):
            lab = X_lab[z_lab == j]
            wj = w[:, j]
            denom = wj.sum() + alpha * len(lab)
            phi[j] = denom / (len(X_unl) + alpha * len(X_lab))
            mu[j] = (wj @ X_unl + alpha * lab.sum(0)) / denom
            du, dl = X_unl - mu[j], lab - mu[j]
            sigma[j] = ((du * wj[:, None]).T @ du + alpha * dl.T @ dl) / denom + reg * np.eye(n)
        ll = np.log(np.clip(dens.sum(1), 1e-300, None)).sum() + alpha * sum(
            multivariate_normal.logpdf(X_lab[z_lab == j], mu[j], sigma[j], allow_singular=True).sum()
            for j in range(K))
        if prev is not None and abs(ll - prev) < eps:
            break
        prev = ll
    return w.argmax(1)


def run():
    df = data.load_raw_dated()  # the paper ran EM on the dated file
    out = {}

    # ---- unsupervised EM: 2-component full-covariance GMM, several random restarts
    feats = AUTHOR_SUBSETS["em"]
    X, y = data.xy(df, feats)
    runs = []
    for s in range(10):
        gm = GaussianMixture(2, covariance_type="full", random_state=s, reg_covar=1e-6).fit(X)
        h = gm.predict(X)
        runs.append({"seed": s,
                     "cluster_id_equals_label": float(np.mean(h == y)),
                     "best_label_matching": float(max(np.mean(h == y), np.mean(h != y))),
                     "as_computed_by_authors": pairwise_match_rate(h, y)})
    Xs = StandardScaler().fit_transform(X)
    gm_s = GaussianMixture(2, covariance_type="full", random_state=SEED, n_init=10).fit(Xs)
    hs = gm_s.predict(Xs)
    out["unsupervised"] = {
        "features": feats,
        "per_seed": runs,
        "best_label_matching_mean": float(np.mean([r["best_label_matching"] for r in runs])),
        "as_computed_by_authors_mean": float(np.mean([r["as_computed_by_authors"] for r in runs])),
        "standardised_best_label_matching": float(max(np.mean(hs == y), np.mean(hs != y))),
        "majority_class_rate": float(max(y.mean(), 1 - y.mean())),
    }

    # ---- semi-supervised EM, labelled share 20% (paper text) and 5% (authors' code)
    feats = AUTHOR_SUBSETS["ss_em"]
    X, y = data.xy(df, feats)
    ss = {}
    for share in (0.05, 0.20):
        rng = np.random.default_rng(SEED)
        lab = rng.random(len(X)) < share
        h = semi_supervised_em(X[~lab], X[lab], y[lab])
        ss[f"{int(share * 100)}pct_labelled"] = {
            "accuracy_on_unlabelled": float(np.mean(h == y[~lab])),
            "as_computed_by_authors": pairwise_match_rate(h, y),
            "share_predicted_hit": float(h.mean()),
        }
    out["semi_supervised"] = {"features": feats, "alpha": 20, **ss}
    out["figures"] = [figure(df, out)]
    return out


def figure(df, out):
    import matplotlib.pyplot as plt

    feats = AUTHOR_SUBSETS["em"]
    X, y = data.xy(df, feats)
    gm = GaussianMixture(2, covariance_type="full", random_state=0, reg_covar=1e-6).fit(X)
    h = gm.predict(X)
    if np.mean(h == y) < 0.5:
        h = 1 - h
    d, a = df["Danceability"], df["Acousticness"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, lab, title in [(axes[0], y, "True labels"), (axes[1], h, "EM clusters (best matching)")]:
        for v, c, name in [(0, COLORS["ours"], "Non-hit"), (1, COLORS["paper"], "Hit")]:
            ax.scatter(d[lab == v], a[lab == v], s=6, alpha=0.5, color=c, label=name, linewidths=0)
        ax.set_xlabel("Danceability")
        ax.set_ylabel("Acousticness")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", fontsize=8, markerscale=2)
    u, s = out["unsupervised"], out["semi_supervised"]
    names = ["EM\n(unsup.)", "SS-EM\n5% labels", "SS-EM\n20% labels", "Majority\nclass"]
    paper = [0.501, np.nan, 0.469, np.nan]
    authors = [u["as_computed_by_authors_mean"], s["5pct_labelled"]["as_computed_by_authors"],
               s["20pct_labelled"]["as_computed_by_authors"], np.nan]
    real = [u["best_label_matching_mean"], s["5pct_labelled"]["accuracy_on_unlabelled"],
            s["20pct_labelled"]["accuracy_on_unlabelled"], u["majority_class_rate"]]
    plotting.grouped_bars(axes[2], names,
                          {"Paper": paper, "Authors' metric (ours)": authors, "Real accuracy": real},
                          {"Paper": COLORS["paper"], "Authors' metric (ours)": COLORS["third"],
                           "Real accuracy": COLORS["ours"]}, ylim=(0, 1), fmt="{:.0%}",
                          label_series=("Paper", "Real accuracy"))
    axes[2].set_title("Accuracy", fontsize=11)
    axes[2].legend(loc="upper left", fontsize=8, ncol=1)
    fig.suptitle("EM clustering: the paper's ~50% is a scoring artifact; real accuracy is higher",
                 x=0.01, ha="left",
                 fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    return plotting.save(fig, "02_em_clustering")
