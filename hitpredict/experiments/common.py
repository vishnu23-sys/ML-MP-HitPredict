"""Helpers shared by the experiment modules."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from ..config import SEED, TARGET
from ..metrics import scores


def paper_lr():
    """Unregularised logistic regression fitted by Newton's method, as in logistic_regression.py."""
    return LogisticRegression(C=np.inf, solver="newton-cholesky", max_iter=500)


def grouped_cv(n_splits: int = 5, seed: int = SEED):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def tune(estimator, grid, X, y, groups, n_splits=5, seed=SEED, scoring="accuracy"):
    """Grid search with grouped CV on the training data only. Returns the refitted search object."""
    search = GridSearchCV(clone(estimator), grid or {}, cv=grouped_cv(n_splits, seed),
                          scoring=scoring, n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    return search


def oof_proba(estimator, X, y, groups, n_splits=5, seed=SEED):
    """Out-of-fold P(hit) for every training row, from grouped CV."""
    out = np.zeros(len(y))
    for tr, te in grouped_cv(n_splits, seed).split(X, y, groups):
        m = clone(estimator).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def evaluate(model, X, y):
    prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return scores(y, model.predict(X), prob)


def train_eval(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    return {"train": evaluate(model, Xtr, ytr), "test": evaluate(model, Xte, yte)}


def labels(df):
    return df[TARGET].to_numpy(dtype=int)
