"""Model definitions: faithful re-implementations of the paper's models and a tuned model zoo."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import SEED


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class GDA(BaseEstimator, ClassifierMixin):
    """Gaussian discriminant analysis with a shared covariance (maximum-likelihood estimates).

    Matches gda.py from the authors: Sigma = (1/m) sum (x - mu_y)(x - mu_y)^T, linear boundary.
    """

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        self.classes_ = np.array([0, 1])
        phi = y.mean()
        mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
        centred = X - np.where(y[:, None] == 1, mu1, mu0)
        sigma = centred.T @ centred / len(X)
        inv = np.linalg.pinv(sigma)
        self.coef_ = inv @ (mu1 - mu0)
        self.intercept_ = (0.5 * mu0 @ inv @ mu0 - 0.5 * mu1 @ inv @ mu1
                           - np.log((1 - phi) / phi))
        return self

    def predict_proba(self, X):
        p = _sigmoid(np.asarray(X, float) @ self.coef_ + self.intercept_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class LegacySVC(BaseEstimator, ClassifierMixin):
    """SVC with gamma='scale' as defined in scikit-learn 0.20 (2018): 1 / (n_features * X.std()).

    scikit-learn 0.22 changed 'scale' to use X.var(). With the paper's unscaled features
    (tempo is around 120) the two give very different kernels; the old definition reproduces the
    paper's RBF and polynomial SVM numbers exactly.
    """

    def __init__(self, kernel="rbf", C=1.0):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, float)
        gamma = 1.0 / (X.shape[1] * X.std())
        self.svc_ = SVC(kernel=self.kernel, C=self.C, gamma=gamma).fit(X, y)
        self.classes_ = self.svc_.classes_
        return self

    def predict(self, X):
        return self.svc_.predict(X)

    def decision_function(self, X):
        return self.svc_.decision_function(X)


class ArtistScoreRule(BaseEstimator, ClassifierMixin):
    """Predict 'hit' exactly when the artist already had a Hot 100 hit. Needs no training."""

    def __init__(self, column: int = 0):
        self.column = column

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(X, float)[:, self.column].clip(0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (np.asarray(X, float)[:, self.column] > 0.5).astype(int)


class PaperNN:
    """The authors' network from project_nn.py, re-implemented.

    One hidden layer of 6 sigmoid units, no hidden bias (an intercept column is added to the
    input), full-batch updates with separate learning rates for the two weight matrices.
    Their 'regularisation' adds lam0*||W0|| + lam1*||W1|| to the output error, which shifts the
    error by a constant rather than penalising the weights; it is kept here to match the paper.
    Inputs are rescaled by the same hand-picked constants as the original script.
    """

    SCALE = {"Speechiness": 10.0, "Acousticness": 10.0, "Instrumentalness": 5.0, "Tempo": 0.015}

    def __init__(self, features, lr0=0.005, lr1=0.003, regularise=True, epochs=18000,
                 eval_every=100, seed=1):
        self.features = list(features)
        self.lr0, self.lr1 = lr0, lr1
        self.regularise = regularise
        self.epochs, self.eval_every, self.seed = epochs, eval_every, seed

    def _prep(self, X):
        X = np.array(X, float)
        for j, f in enumerate(self.features):
            X[:, j] *= self.SCALE.get(f, 1.0)
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.RandomState(self.seed)
        x0 = self._prep(X)
        y = np.asarray(y, float).reshape(-1, 1)
        xv = self._prep(X_val) if X_val is not None else None
        self.W0 = 2 * rng.random_sample((x0.shape[1], 6)) - 1
        self.W1 = 2 * rng.random_sample((6, 1)) - 1
        self.history = {"epoch": [], "train_acc": [], "val_acc": []}
        for ep in range(self.epochs):
            h = _sigmoid(x0 @ self.W0)
            out = _sigmoid(h @ self.W1)
            err = y - out
            if self.regularise:
                err = err + self.lr0 * np.linalg.norm(self.W0) + self.lr1 * np.linalg.norm(self.W1)
            d_out = err * out * (1 - out)
            d_h = (d_out @ self.W1.T) * h * (1 - h)
            self.W0 += self.lr0 * x0.T @ d_h
            self.W1 += self.lr1 * h.T @ d_out
            if ep % self.eval_every == 0:
                self.history["epoch"].append(ep)
                self.history["train_acc"].append(float(np.mean((out >= 0.5) == y)))
                if xv is not None:
                    pv = _sigmoid(_sigmoid(xv @ self.W0) @ self.W1)
                    self.history["val_acc"].append(
                        float(np.mean((pv[:, 0] >= 0.5) == np.asarray(y_val))))
        return self

    def predict_proba(self, X):
        p = _sigmoid(_sigmoid(self._prep(X) @ self.W0) @ self.W1)[:, 0]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- tuned model zoo

def _scaled(est):
    return Pipeline([("scale", StandardScaler()), ("clf", est)])


def model_zoo(feature_names):
    """(name, estimator, param_grid, family) for every model in the comparison.

    Grids are searched with grouped cross-validation on the training set only.
    """
    as_col = list(feature_names).index("ArtistScore") if "ArtistScore" in feature_names else 0
    return [
        ("Majority class", DummyClassifier(strategy="most_frequent"), {}, "baseline"),
        ("Artist score rule", ArtistScoreRule(column=as_col), {}, "baseline"),
        ("Naive Bayes", _scaled(GaussianNB()), {}, "linear"),
        ("GDA (shared covariance)", _scaled(GDA()), {}, "linear"),
        ("Logistic regression", _scaled(LogisticRegression(max_iter=2000)),
         {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}, "linear"),
        ("Linear SVM", _scaled(SVC(kernel="linear", probability=True, random_state=SEED)),
         {"clf__C": [0.01, 0.1, 1]}, "linear"),
        ("k-nearest neighbours", _scaled(KNeighborsClassifier()),
         {"clf__n_neighbors": [5, 15, 31, 61, 101], "clf__weights": ["uniform", "distance"]},
         "nonlinear"),
        ("Decision tree", DecisionTreeClassifier(random_state=SEED),
         {"max_depth": [2, 3, 4, 6, 8, None], "min_samples_leaf": [1, 10, 30, 60]}, "nonlinear"),
        ("RBF SVM", _scaled(SVC(kernel="rbf", probability=True, random_state=SEED)),
         {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}, "nonlinear"),
        ("Neural network (6 sigmoid units)",
         _scaled(MLPClassifier(hidden_layer_sizes=(6,), activation="logistic", max_iter=3000,
                               random_state=SEED)),
         {"clf__alpha": [1e-4, 1e-2, 1e-1, 1, 3]}, "nonlinear"),
        ("Neural network (32-16 ReLU)",
         _scaled(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000, early_stopping=True,
                               random_state=SEED)),
         {"clf__alpha": [1e-3, 1e-1, 1, 3]}, "nonlinear"),
        ("Random forest", RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1),
         {"max_depth": [4, 8, None], "min_samples_leaf": [1, 5, 20],
          "max_features": ["sqrt", 0.5]}, "ensemble"),
        ("Gradient boosting", HistGradientBoostingClassifier(random_state=SEED),
         {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, None],
          "max_leaf_nodes": [7, 15], "l2_regularization": [0.0, 1.0],
          "max_iter": [100, 300]}, "ensemble"),
    ]


class HitFeatures(BaseEstimator):
    """Feature engineering used by improvement 2.

    * log10 transform of the four heavily right-skewed Spotify features (most songs sit near 0,
      a few near 1), so a linear model sees their order of magnitude rather than raw values;
    * Key is dropped (an unordered 0-11 code is meaningless as a number); Mode is kept as 0/1;
    * ArtistScore x audio-feature interactions, so a linear model can learn a different audio
      profile for established artists and newcomers.
    Input columns must be in the order of `feature_names`.
    """

    SKEWED = ("Speechiness", "Acousticness", "Instrumentalness", "Liveness")

    def __init__(self, feature_names=None, interactions=True):
        self.feature_names = feature_names
        self.interactions = interactions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        names = list(self.feature_names)
        X = np.array(X, float)
        for i, f in enumerate(names):
            if f in self.SKEWED:
                X[:, i] = np.log10(X[:, i] + 1e-3)
        keep = [i for i, f in enumerate(names) if f != "Key"]
        out = X[:, keep]
        if self.interactions and "ArtistScore" in names:
            a = X[:, names.index("ArtistScore")][:, None]
            audio = [i for i, f in enumerate(names) if f not in ("ArtistScore", "Key", "Mode")]
            out = np.hstack([out, a * X[:, audio]])
        return out

    def get_feature_names_out(self, input_features=None):
        names = [f for f in self.feature_names if f != "Key"]
        if self.interactions and "ArtistScore" in self.feature_names:
            names += [f"ArtistScore*{f}" for f in self.feature_names
                      if f not in ("ArtistScore", "Key", "Mode")]
        return np.array(names)


def engineered_lr(feature_names):
    return Pipeline([("features", HitFeatures(feature_names)), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000))])


__all__ = ["GDA", "LegacySVC", "ArtistScoreRule", "PaperNN", "HitFeatures", "model_zoo",
           "engineered_lr"]
