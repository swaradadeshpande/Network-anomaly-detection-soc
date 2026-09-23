"""
evaluator.py - Metrics, severity scoring and static report figures
==================================================================

Two families of metrics
-----------------------
Unsupervised (always available, no labels needed)
    * Silhouette score of the normal-vs-anomaly partition in the scaled feature
      space: s = (b - a) / max(a, b) in [-1, 1], where a = mean intra-group
      distance and b = mean distance to the other group. Higher means the flagged
      flows are better separated from normal traffic. It is only a *geometric*
      sanity check - it cannot tell you whether the flagged flows are malicious.
    * Number / percentage of flagged flows.

Supervised (only when the file has a label column - e.g. synthetic data, CICIDS2017)
    * Precision = TP/(TP+FP)  : how many alerts were real  (analyst workload)
    * Recall    = TP/(TP+FN)  : how many attacks were caught (missed breaches)
    * F1        = harmonic mean of the two
    * ROC-AUC   : threshold-free ranking quality of the raw scores
    * Average precision (PR-AUC): like ROC-AUC but honest under heavy class imbalance
    * Confusion matrix
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, silhouette_score)


def compute_unsupervised_metrics(X: np.ndarray, labels: np.ndarray,
                                 sample_size: int = 5000, seed: int = 42) -> Dict[str, float]:
    """Flag counts + silhouette of the normal/anomaly split.

    Silhouette is O(n^2), so it is computed on a random sample of `sample_size`
    rows. It is NaN when fewer than two groups exist.
    """
    n_anom = int(np.sum(labels))
    out = {"n_flows": int(len(labels)), "n_anomalies": n_anom,
           "anomaly_pct": 100.0 * n_anom / max(len(labels), 1), "silhouette": float("nan")}
    if 1 < n_anom < len(labels) - 1:
        try:
            out["silhouette"] = float(silhouette_score(X, labels, sample_size=min(sample_size, len(labels)),
                                                       random_state=seed))
        except ValueError:      # the random sample happened to miss one group
            pass
    return out


def compute_supervised_metrics(y_true: np.ndarray, labels: np.ndarray,
                               scores: np.ndarray) -> Dict[str, object]:
    """Precision/recall/F1/ROC-AUC/PR-AUC/confusion matrix against ground truth."""
    out: Dict[str, object] = {
        "precision": precision_score(y_true, labels, zero_division=0),
        "recall": recall_score(y_true, labels, zero_division=0),
        "f1": f1_score(y_true, labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, labels, labels=[0, 1]),  # [[TN FP],[FN TP]]
        "roc_auc": float("nan"), "avg_precision": float("nan"),
    }
    if len(np.unique(y_true)) == 2:     # AUC is undefined with a single class
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["avg_precision"] = average_precision_score(y_true, scores)
    return out


def evaluate_model(X, labels, scores, y_true: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Merge unsupervised and (if labels exist) supervised metrics into one dict."""
    metrics = compute_unsupervised_metrics(X, labels)
    if y_true is not None:
        metrics.update(compute_supervised_metrics(y_true, labels, scores))
    return metrics


def compare_models(results: dict, X, y_true: Optional[np.ndarray] = None) -> pd.DataFrame:
    """One row per model (TrainedModel objects) - ready for printing or st.dataframe."""
    rows = []
    for name, m in results.items():
        met = evaluate_model(X, m.labels, m.scores, y_true)
        row = {"model": name, "flagged": met["n_anomalies"], "flagged_%": round(met["anomaly_pct"], 2),
               "silhouette": met["silhouette"], "fit_s": round(m.fit_seconds, 2)}
        if y_true is not None:
            row.update({k: met[k] for k in ("precision", "recall", "f1", "roc_auc", "avg_precision")})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").round(3)


def compute_severity(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Map raw anomaly scores to a 0-100 severity (0 for flows below the cut-off).

    Raw scores differ wildly between models (IF ~0.4-0.9, LOF up to 100+), and
    the excess over the threshold is heavy-tailed. We therefore:
        excess = score - threshold                      (only for flagged flows)
        severity = 100 * log1p(excess/m) / log1p(ref/m)
    where m = median excess (natural scale) and ref = 99th percentile of the
    excess. The log keeps a few extreme flows from flattening everyone else to ~0.
    """
    excess = np.clip(np.asarray(scores, dtype=float) - threshold, 0, None)
    flagged = excess > 0
    sev = np.zeros_like(excess)
    if flagged.any():
        m = float(np.median(excess[flagged])) or 1e-9
        ref = float(np.quantile(excess[flagged], 0.99)) or m
        sev[flagged] = np.clip(100 * np.log1p(excess[flagged] / m) / np.log1p(ref / m), 0, 100)
    return sev


def threat_level(anomaly_pct: float):
    """(label, hex colour) for the dashboard indicator, based on the share of flagged flows.

    Caveat: for Isolation Forest and LOF the flagged share is *set* by the
    contamination parameter, so treat this indicator as relative to that setting.
    """
    for limit, label, colour in ((1, "LOW", "#2ca02c"), (3, "GUARDED", "#8fbf3f"),
                                 (8, "ELEVATED", "#f0a30a"), (15, "HIGH", "#e8590c")):
        if anomaly_pct < limit:
            return label, colour
    return "CRITICAL", "#d62728"


def save_report_figures(results: dict, y_true: Optional[np.ndarray], out_dir) -> list:
    """Write PNG figures (score histograms, confusion matrices) with matplotlib/seaborn."""
    import matplotlib
    matplotlib.use("Agg")                       # headless backend: works on servers without a display
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    n = len(results)

    # Figure 1: score distribution per model with the decision threshold (log-y shows the tail).
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.8))
    for ax, (name, m) in zip(np.atleast_1d(axes), results.items()):
        sns.histplot(m.scores, bins=80, ax=ax, color="#4C78A8")
        ax.axvline(m.detector.threshold_, color="red", ls="--", label="cut-off")
        ax.set(title=name, xlabel="anomaly score (higher = worse)", yscale="log")
        ax.legend()
    fig.tight_layout()
    paths.append(out_dir / "score_distributions.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    # Figure 2: confusion matrices (only if ground truth is available).
    if y_true is not None:
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
        for ax, (name, m) in zip(np.atleast_1d(axes), results.items()):
            cm = confusion_matrix(y_true, m.labels, labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
            ax.set(title=name, xlabel="predicted", ylabel="actual")
        fig.tight_layout()
        paths.append(out_dir / "confusion_matrices.png")
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig)
    return paths
