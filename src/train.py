"""
train.py - Unsupervised detectors: Isolation Forest, DBSCAN, Local Outlier Factor
=================================================================================

Common contract
---------------
All three detectors are wrapped in classes exposing the same API so the
dashboard, the CLI and the evaluator treat them interchangeably:

    detector.fit(X)              learn from an (n_samples, n_features) scaled matrix
    detector.train_scores_       anomaly score of every training row (HIGHER = MORE ANOMALOUS)
    detector.threshold_          decision cut-off:  anomaly  <=>  score > threshold_
    detector.score(X_new)        score unseen rows
    detector.predict(X_new)      1 = anomaly, 0 = normal
    detector.fit_predict(X)      convenience: (labels, scores) on the training data

Using a single "higher = worse" score convention (scikit-learn's own
score_samples is "higher = more normal", so we negate it) makes ROC-AUC,
histograms, severity ranking and thresholds work identically for every model.

Model intuition
---------------
Isolation Forest : anomalies are "few and different", so random axis-parallel
    cuts isolate them in very few splits. Score = 2^(-E[path length]/c(n)).
DBSCAN           : anomalies are points not density-reachable from any dense
    region (label -1 = noise). Purely density/distance based.
LOF              : compares a point's local density with that of its
    k neighbours; LOF >> 1 means "much sparser than the neighbourhood".
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from src.preprocessor import NetworkPreprocessor, extract_labels, load_dataset

RANDOM_STATE = 42   # fixed seed: identical results on every run


def auto_group_size(n_samples: int, contamination: float, low: int, high: int = 1000) -> int:
    """Neighbourhood size for the density-based detectors when the user gives none.

    Density methods only flag a group of similar flows if the group is SMALLER
    than the neighbourhood they look at (k for LOF, min_samples for DBSCAN).
    Attack traffic is bursty: a port scan or flood produces hundreds of nearly
    identical flows, and if k is smaller than that burst the burst looks like an
    ordinary dense cluster (LOF ~ 1) and slips through. The expected number of
    anomalies is contamination * n, so we take HALF of that as the default
    (bounded to [low, high] to keep runtime sane and never above n - 1).
    """
    return int(min(max(low, 0.5 * contamination * n_samples), high, max(n_samples - 1, 2)))


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class BaseDetector:
    """Shared behaviour: thresholding is always `score > threshold_`."""

    name = "base"

    def fit(self, X: np.ndarray):                       # pragma: no cover - abstract
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:       # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary decision for NEW data: 1 = anomaly, 0 = normal."""
        return (self.score(X) > self.threshold_).astype(int)

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and return (labels, scores) for the training rows themselves.

        Uses `train_scores_` computed during fit (NOT score(X)) because for
        LOF re-scoring a training point would count the point as its own neighbour.
        """
        self.fit(X)
        return (self.train_scores_ > self.threshold_).astype(int), self.train_scores_


# --------------------------------------------------------------------------- #
# 1. Isolation Forest
# --------------------------------------------------------------------------- #
class IsolationForestDetector(BaseDetector):
    """Ensemble of random isolation trees.

    Hyper-parameters
    ----------------
    contamination : expected fraction of anomalies. Sets the decision threshold
        at the (1 - contamination) quantile of training scores. 0.02-0.05 is
        typical for network data where attacks are rare.
    n_estimators  : number of trees. Variance of the score falls ~1/sqrt(T);
        100 is the paper default, 200 gives more stable rankings for little cost.
    max_samples   : rows per tree. 'auto' = min(256, n): the paper shows 256 is
        enough, and small sub-samples REDUCE "swamping/masking" of anomalies.
    """

    name = "Isolation Forest"

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 max_samples="auto", random_state: int = RANDOM_STATE):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X):
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators, max_samples=self.max_samples,
            contamination=self.contamination, random_state=self.random_state, n_jobs=-1,
        ).fit(X)
        # sklearn: score_samples is higher = more normal, and offset_ is the
        # contamination-quantile of it. Negate both to get "higher = anomalous".
        self.train_scores_ = -self.model_.score_samples(X)
        self.threshold_ = float(-self.model_.offset_)
        return self

    def score(self, X):
        return -self.model_.score_samples(X)


# --------------------------------------------------------------------------- #
# 2. DBSCAN
# --------------------------------------------------------------------------- #
class DBSCANDetector(BaseDetector):
    """Density-based clustering where noise (label -1) is the anomaly class.

    DBSCAN has no native predict(), so we turn it into a scorer:
        score(x) = distance from x to the nearest CORE sample.
    A point is noise exactly when that distance exceeds eps, hence
    threshold_ = eps and `score > eps` reproduces DBSCAN's own -1 labels on the
    training set while also giving a continuous severity and supporting new data.

    Hyper-parameters
    ----------------
    min_samples : neighbours (incl. itself) required within eps to be a core
        point. Rule of thumb: >= 2 * n_dimensions, larger for noisy data.
        NOTE: an attack burst with MORE than min_samples similar flows (e.g. a
        big port scan) becomes a legitimate dense cluster and is NOT flagged -
        a known DBSCAN weakness. None (default) = auto_group_size(): half the
        expected anomaly count, so bursts smaller than that are still noise.
    eps : neighbourhood radius. If None, chosen automatically with the
        "k-distance" heuristic: distance to the min_samples-th neighbour is
        computed for every point and eps is set to its (1 - contamination)
        quantile, so roughly the sparsest `contamination` share of points are
        non-core. (Result is an UPPER bound on flagged share - border points
        near dense areas are rescued.)
    """

    name = "DBSCAN"

    def __init__(self, contamination: float = 0.05, eps: Optional[float] = None,
                 min_samples: Optional[int] = None):
        self.contamination = contamination
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.min_samples_ = (self.min_samples if self.min_samples is not None
                             else auto_group_size(len(X), self.contamination, low=10))
        k = min(self.min_samples_, len(X))
        if self.eps is None:
            kdist = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X).kneighbors(X)[0][:, -1]
            q = float(np.clip(1.0 - self.contamination, 0.5, 0.999))
            self.eps_ = float(np.quantile(kdist, q))
        else:
            self.eps_ = float(self.eps)
        self.model_ = DBSCAN(eps=self.eps_, min_samples=self.min_samples_, n_jobs=-1).fit(X)
        core = self.model_.core_sample_indices_
        if len(core) == 0:
            raise ValueError("DBSCAN found no core samples: increase eps or lower min_samples.")
        self.core_nn_ = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X[core])
        self.n_clusters_ = int(len(set(self.model_.labels_) - {-1}))
        self.threshold_ = self.eps_
        self.train_scores_ = self.score(X)
        return self

    def score(self, X):
        return self.core_nn_.kneighbors(X)[0][:, 0]


# --------------------------------------------------------------------------- #
# 3. Local Outlier Factor
# --------------------------------------------------------------------------- #
class LOFDetector(BaseDetector):
    """Local Outlier Factor in novelty mode (so unseen flows can be scored).

    Hyper-parameters
    ----------------
    n_neighbors   : neighbourhood size k. Must exceed the size of the largest
        group you still want called "abnormal" (otherwise an attack burst looks
        like a normal dense cluster) yet stay below the smallest legitimate
        cluster. None (default) = auto_group_size(): half the expected number
        of anomalies (contamination * n), at least 35, at most 1000.
    contamination : quantile used to convert scores into a hard decision.
    """

    name = "Local Outlier Factor"

    def __init__(self, contamination: float = 0.05, n_neighbors: Optional[int] = None):
        self.contamination = contamination
        self.n_neighbors = n_neighbors

    def fit(self, X):
        self.n_neighbors_ = int(min(self.n_neighbors, len(X) - 1)) if self.n_neighbors is not None \
            else auto_group_size(len(X), self.contamination, low=35)
        self.model_ = LocalOutlierFactor(n_neighbors=self.n_neighbors_, contamination=self.contamination,
                                         novelty=True, n_jobs=-1).fit(X)
        # negative_outlier_factor_ = -LOF of the training rows (leave-one-out, unbiased).
        self.train_scores_ = -self.model_.negative_outlier_factor_
        self.threshold_ = float(-self.model_.offset_)
        return self

    def score(self, X):
        return -self.model_.score_samples(X)


MODEL_REGISTRY = {
    IsolationForestDetector.name: IsolationForestDetector,
    DBSCANDetector.name: DBSCANDetector,
    LOFDetector.name: LOFDetector,
}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class TrainedModel:
    """Everything produced by fitting one detector on one dataset."""
    name: str
    detector: BaseDetector
    labels: np.ndarray        # 1 = anomaly
    scores: np.ndarray        # higher = more anomalous
    fit_seconds: float


def train_model(name: str, X: np.ndarray, contamination: float = 0.05, **params) -> TrainedModel:
    """Fit the named detector on X and return labels, scores and timing."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    detector = MODEL_REGISTRY[name](contamination=contamination, **params)
    t0 = time.perf_counter()
    labels, scores = detector.fit_predict(X)
    return TrainedModel(name, detector, labels, scores, time.perf_counter() - t0)


def slugify(name: str) -> str:
    """'Local Outlier Factor' -> 'local_outlier_factor' (file-name safe)."""
    return name.lower().replace(" ", "_")


def save_artifact(path: Path, preprocessor: NetworkPreprocessor, model: TrainedModel,
                  contamination: float) -> None:
    """Persist preprocessor + detector together so inference can never mismatch scaling."""
    joblib.dump({"preprocessor": preprocessor, "detector": model.detector,
                 "meta": {"name": model.name, "contamination": contamination,
                          "fit_seconds": model.fit_seconds}}, path)


def load_artifact(path) -> dict:
    """Load an artifact saved by save_artifact (requires the `src` package importable)."""
    return joblib.load(path)


def score_dataframe(artifact: dict, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a saved artifact to new raw flows -> (labels, scores)."""
    features, _, _ = extract_labels(df)
    X = artifact["preprocessor"].transform(features)
    scores = artifact["detector"].score(X)
    return (scores > artifact["detector"].threshold_).astype(int), scores


def train_all_models(df: pd.DataFrame, contamination: float = 0.05,
                     save_dir: Optional[Path] = None
                     ) -> Tuple[NetworkPreprocessor, np.ndarray, Dict[str, TrainedModel], Optional[np.ndarray]]:
    """Preprocess once, train all three detectors, optionally save `.pkl` artifacts.

    Returns (preprocessor, X, {model_name: TrainedModel}, y_true_or_None).
    """
    features, y, _ = extract_labels(df)              # labels are split off: models never see them
    pre = NetworkPreprocessor().fit(features)
    X = pre.transform(features)
    results = {n: train_model(n, X, contamination) for n in MODEL_REGISTRY}
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for n, m in results.items():
            save_artifact(save_dir / f"{slugify(n)}.pkl", pre, m, contamination)
    return pre, X, results, y


def main() -> None:
    """CLI: python -m src.train --data data/synthetic_traffic.csv --contamination 0.05"""
    p = argparse.ArgumentParser(description="Train all anomaly detectors and save artifacts.")
    p.add_argument("--data", required=True)
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--save-dir", default="models")
    a = p.parse_args()
    _, _, results, _ = train_all_models(load_dataset(a.data), a.contamination, Path(a.save_dir))
    for n, m in results.items():
        print(f"{n:22s} flagged {int(m.labels.sum()):6d} flows in {m.fit_seconds:.1f}s")


if __name__ == "__main__":
    main()
