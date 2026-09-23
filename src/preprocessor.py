"""
preprocessor.py - Loading, cleaning, skew reduction and scaling
===============================================================

Why preprocessing matters so much for *unsupervised* anomaly detection
----------------------------------------------------------------------
Distance/density based detectors (DBSCAN, LOF) and even tree isolation are
driven by raw feature geometry. Without care:

1. Features with huge numeric range (bytes/s ~ 1e7, flags ~ 1) dominate every
   Euclidean distance -> we STANDARDISE (zero mean, unit variance).
2. Network features are heavy-tailed (log-normal). A single 1e9 bytes/s flow
   stretches the scaler and squashes all normal traffic into a tiny interval
   -> we apply log(1 + x) to skewed non-negative columns FIRST, then scale.
3. Identifiers (IPs, ports, timestamps, flow IDs) let a model "memorise" hosts
   instead of learning behaviour and do not generalise -> we DROP them.
4. Real datasets contain +/-inf and NaN (CICIDS2017 famously has 'Infinity'
   in Flow Bytes/s) -> we convert inf to NaN and impute with the column MEDIAN
   (robust to outliers, unlike the mean).

Dataset compatibility
---------------------
* Synthetic data from generate_data.py.
* CICIDS2017 CSVs (headers like ' Flow Duration', 'Flow Bytes/s'): column names
  are normalised to snake_case, so they just work.
* NSL-KDD (KDDTrain+.txt, header-less, 43 columns): detected automatically and
  named using the official schema; symbolic columns (protocol_type, service,
  flag) are ignored because they are non-numeric.
* Any other CSV: every numeric, non-identifier column becomes a feature.
"""

from __future__ import annotations

import re
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# --------------------------------------------------------------------------- #
# Schema knowledge
# --------------------------------------------------------------------------- #
# Columns (after snake_case normalisation) that identify WHO/WHEN, not HOW.
ID_COLUMNS = {
    "flow_id", "id", "unnamed_0", "timestamp", "time", "date",
    "src_ip", "source_ip", "dst_ip", "destination_ip",
    "src_port", "source_port", "dst_port", "destination_port", "protocol",
}
# Ground-truth columns: NEVER a feature, only used for evaluation.
LABEL_COLUMNS = {"label", "class", "attack_type", "attack_cat", "attack", "difficulty"}

# CICIDS2017 names that differ from our canonical synthetic names.
COLUMN_ALIASES = {
    "total_backward_packets": "total_bwd_packets",
    "total_length_of_fwd_packets": "total_fwd_bytes",
    "total_length_of_bwd_packets": "total_bwd_bytes",
    "flow_bytes_s": "flow_bytes_s",
    "flow_packets_s": "flow_packets_s",
}

# Official NSL-KDD column order (41 features + label + difficulty score).
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

# Label strings that mean "not an attack" across the supported datasets.
BENIGN_VALUES = {"benign", "normal", "0", "0.0"}


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise headers to snake_case and map known aliases to canonical names.

    ' Flow Bytes/s' -> 'flow_bytes_s', 'Total Length of Fwd Packets' -> 'total_fwd_bytes'.
    Idempotent: applying it twice changes nothing.
    """
    out = df.copy(deep=False)                       # shallow copy: never mutate the caller's frame
    out.columns = [re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_")
                   for c in out.columns]
    out = out.rename(columns=COLUMN_ALIASES)
    return out.loc[:, ~out.columns.duplicated()]    # some CICIDS files repeat a column name


def load_dataset(path_or_buffer, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read a CSV / NSL-KDD text file (path or file-like) into a standardised DataFrame."""
    df = pd.read_csv(path_or_buffer, nrows=nrows, low_memory=False)
    std = standardize_columns(df)
    # NSL-KDD ships WITHOUT a header row, so pandas mistakes the first record for names.
    # Signature: 42/43 columns and none of the expected schema names present.
    if df.shape[1] in (42, 43) and "src_bytes" not in std.columns:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)                  # rewind uploaded-file buffers before re-reading
        df = pd.read_csv(path_or_buffer, header=None, nrows=nrows,
                         names=NSL_KDD_COLUMNS[: df.shape[1]], low_memory=False)
        std = standardize_columns(df)
    return std


def extract_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[pd.Series]]:
    """Split a standardised frame into (features_frame, binary_labels, attack_names).

    * binary_labels: 0 = benign, 1 = attack (None when the file has no label column).
    * attack_names : original attack strings when available (for per-attack recall).
    The returned features frame still contains identifier columns; the
    preprocessor drops them itself so the dashboard can keep them for display.
    """
    df = standardize_columns(df)
    label_col = next((c for c in ("label", "class") if c in df.columns), None)
    y, attack = None, None
    if label_col is not None:
        as_text = df[label_col].astype(str).str.strip().str.lower()
        y = (~as_text.isin(BENIGN_VALUES)).astype(int).to_numpy()
        if "attack_type" in df.columns:
            attack = df["attack_type"].astype(str)
        elif not pd.api.types.is_numeric_dtype(df[label_col]):   # string labels (works for object AND pandas-3 'str' dtype)
            attack = df[label_col].astype(str).str.strip()
    features = df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns])
    return features, y, attack


# --------------------------------------------------------------------------- #
# The transformer
# --------------------------------------------------------------------------- #
class NetworkPreprocessor(BaseEstimator, TransformerMixin):
    """Fit-once, apply-anywhere preprocessing pipeline.

    fit():
        1. drop identifier / label columns
        2. coerce everything else to numbers; keep columns that are >= 90 % numeric
        3. inf -> NaN, drop all-NaN and constant columns
        4. remember per-column medians and impute NaN with them
        5. detect skewed columns (skew > `skew_threshold`, min >= 0) -> log1p
        6. fit StandardScaler on the (partly logged) matrix
    transform(): replays exactly the same steps using the *fitted* statistics, so
    new traffic is scaled with the training baseline (no leakage, no drift in scale).

    Parameters
    ----------
    skew_threshold : float
        Sample skewness above which a non-negative column is log-transformed.
        |skew| > 1 is the usual rule of thumb for "highly skewed".
    """

    def __init__(self, skew_threshold: float = 1.0):
        self.skew_threshold = skew_threshold

    # -- internal helpers ---------------------------------------------------- #
    @staticmethod
    def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
        """Coerce to float; unparsable strings (e.g. 'abc') become NaN, 'Infinity' becomes inf."""
        num = frame.apply(pd.to_numeric, errors="coerce").astype("float64")
        return num.replace([np.inf, -np.inf], np.nan)   # inf would poison mean/std -> treat as missing

    def _log(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log(1+x) to the fitted skewed columns. clip(0) guards unseen negative values."""
        if self.log_cols_:
            X = X.copy()
            X[self.log_cols_] = np.log1p(X[self.log_cols_].clip(lower=0))
        return X

    # -- sklearn API --------------------------------------------------------- #
    def fit(self, df: pd.DataFrame, y=None):
        df = standardize_columns(df)
        candidates = [c for c in df.columns if c not in ID_COLUMNS and c not in LABEL_COLUMNS]
        raw = df[candidates].apply(pd.to_numeric, errors="coerce")
        # Text columns (IPs, 'service' names...) turn into all-NaN -> removed by this rule.
        keep = [c for c in candidates if raw[c].notna().mean() >= 0.9]
        X = self._numeric(df[keep])
        X = X.dropna(axis=1, how="all")                 # nothing to learn from an empty column
        X = X.loc[:, X.nunique(dropna=True) > 1]        # constant columns have zero information
        if X.shape[1] == 0:
            raise ValueError("No usable numeric feature columns were found in the data.")
        self.feature_names_ = list(X.columns)
        self.medians_ = X.median()                      # robust central value for imputation
        X = X.fillna(self.medians_)
        skew = X.skew()
        self.log_cols_ = [c for c in X.columns if skew[c] > self.skew_threshold and X[c].min() >= 0]
        self.scaler_ = StandardScaler().fit(self._log(X))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "scaler_")
        df = standardize_columns(df)
        missing = [c for c in self.feature_names_ if c not in df.columns]
        if missing:
            warnings.warn(f"Columns missing from input, imputed with training medians: {missing}")
        X = self._numeric(df.reindex(columns=self.feature_names_))
        X = X.fillna(self.medians_)
        return self.scaler_.transform(self._log(X))

    def get_feature_report(self) -> pd.DataFrame:
        """Small table describing every feature and whether it was log-transformed."""
        return pd.DataFrame({"feature": self.feature_names_,
                             "log1p_applied": [c in self.log_cols_ for c in self.feature_names_],
                             "train_median": [self.medians_[c] for c in self.feature_names_]})
