"""Smoke tests. Data-dependent tests are skipped until scripts/download_data.py has been run;
the CLI test is skipped until the models exist (scripts/run_all.py)."""

import numpy as np
import pytest

from hitpredict import config, data
from hitpredict.experiments.unsupervised import pairwise_match_rate
from hitpredict.metrics import group_bootstrap_ci, scores
from hitpredict.models import GDA, ArtistScoreRule, HitFeatures, PaperNN

needs_data = pytest.mark.skipif(not data.data_present(), reason="run scripts/download_data.py")
needs_models = pytest.mark.skipif(not (config.MODEL_DIR / "best_model.joblib").exists(),
                                  reason="run scripts/run_all.py")


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 3)) + y[:, None] * 1.5
    return X, y


def test_primary_artist_groups_featured_artists():
    assert data.primary_artist("Drake feat. Rihanna") == "drake"
    assert data.primary_artist("maroon 5 ft. cardi b") == "maroon 5"
    assert data.primary_artist("Halsey") == "halsey"


def test_gda_and_paper_nn_learn_a_separable_problem():
    X, y = _toy()
    assert np.mean(GDA().fit(X, y).predict(X) == y) > 0.85
    nn = PaperNN(["a", "b", "c"], epochs=1500, eval_every=500).fit(X, y, X, y)
    assert np.mean(nn.predict(X) == y) > 0.8
    assert len(nn.history["val_acc"]) == 3


def test_hit_features_shape_and_names():
    X = np.tile([[1, 0.5, 0.6, 0.1, 0.2, 0.0, 0.1, 0.5, -6, 120, 5, 1]], (4, 1)).astype(float)
    hf = HitFeatures(config.ALL_FEATURES)
    out = hf.transform(X)
    assert out.shape == (4, len(hf.get_feature_names_out()))
    assert np.isfinite(out).all()


def test_metrics_and_bootstrap():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 0, 1, 1, 1, 1])
    s = scores(y, p, p.astype(float))
    assert s["accuracy"] == pytest.approx(0.75)
    lo, hi = group_bootstrap_ci(y, p, np.arange(len(y)) // 2, n_boot=200)
    assert 0 <= lo <= 0.75 <= hi <= 1


def test_pairwise_match_rate_is_not_accuracy():
    y = np.array([0, 1] * 50)
    assert pairwise_match_rate(y, y) == pytest.approx(0.5)  # perfect predictions score 0.5
    assert np.mean(y == y) == 1.0


def test_artist_score_rule():
    X = np.array([[1.0, 0.3], [0.0, 0.9]])
    assert ArtistScoreRule(column=0).fit(X, [1, 0]).predict(X).tolist() == [1, 0]


@needs_data
def test_dataset_and_group_split():
    df = data.load_master()
    assert len(df) == 4019 and int(df.Label.sum()) == 1595 and int(df.Dated.sum()) == 3206
    tr, te = data.main_split()
    assert len(tr) + len(te) == len(df)
    assert not set(tr.Group) & set(te.Group)


@needs_data
def test_paper_logistic_regression_is_reproduced():
    from hitpredict.experiments.common import paper_lr

    tr, va = data.paper_split(data.load_raw_dated())
    feats = config.AUTHOR_SUBSETS["lr"]
    m = paper_lr().fit(*data.xy(tr, feats))
    Xva, yva = data.xy(va, feats)
    s = scores(yva, m.predict(Xva))
    assert round(s["accuracy"], 3) == config.PAPER["table2"]["LR"]["val_acc"]
    assert s["confusion"] == config.PAPER["lr_confusion_val"]


@needs_models
def test_cli_runs(capsys):
    from hitpredict import predict

    predict.main(["--random", "2"])
    predict.main(["--song", "ArtistScore=1", "Danceability=0.8"])
    out = capsys.readouterr().out
    assert "P(hit)" in out
