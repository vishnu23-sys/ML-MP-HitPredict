"""Evaluation metrics and confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .config import SEED


def scores(y_true, y_pred, y_prob=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion"] = {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}
    return out


def group_bootstrap_ci(y_true, y_pred, groups, n_boot: int = 2000, alpha: float = 0.05,
                       seed: int = SEED):
    """95% CI for accuracy, resampling whole artists (songs by one artist are not independent)."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    correct = (y_true == y_pred).astype(float)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=correct)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    accs = sums[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.quantile(accs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_group_bootstrap(y_true, pred_a, pred_b, groups, n_boot: int = 2000, seed: int = SEED):
    """Bootstrap the accuracy difference (a - b) by artist. Returns (diff, lo, hi, p_le_0)."""
    y_true, pred_a, pred_b, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    d = (pred_a == y_true).astype(float) - (pred_b == y_true).astype(float)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    diffs = sums[draws].sum(1) / counts[draws].sum(1)
    return (float(d.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)),
            float(np.mean(diffs <= 0)))
