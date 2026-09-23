"""
app.py - Interactive Streamlit Security Operations Center (SOC) dashboard
=========================================================================

Run with:   streamlit run app.py

Workflow implemented here
-------------------------
1. The analyst uploads a CSV of network flows (or uses the built-in synthetic
   traffic), chooses a detector and a contamination rate in the sidebar.
2. The data is cleaned/log-transformed/scaled (NetworkPreprocessor) and the
   chosen unsupervised detector is FIT ON THE UPLOADED BATCH. This is the usual
   SOC "hunt" setting: no attack labels are needed - the model learns what is
   typical *in this capture* and reports what deviates.
3. Results are shown as KPI cards, a PCA map, a score histogram with the
   decision cut-off, and a filterable alert table ranked by severity.

Streamlit re-executes this script top-to-bottom on every interaction, so
expensive work is wrapped in @st.cache_data: moving a filter never re-trains.
"""

import io
import sys
from pathlib import Path

# Make `import src...` work no matter which directory `streamlit run` is started from.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from src.evaluator import (compute_severity, compute_supervised_metrics,
                           compute_unsupervised_metrics, threat_level)
from src.generate_data import generate_synthetic_traffic
from src.preprocessor import (NetworkPreprocessor, extract_labels, load_dataset,
                              standardize_columns)
from src.train import MODEL_REGISTRY, train_model

# st.set_page_config must be the first Streamlit call in the script.
st.set_page_config(page_title="SOC | Network Anomaly Detection", page_icon="🛡️", layout="wide")

MAX_PLOT_POINTS = 8000      # WebGL scatter stays smooth up to ~10k points; anomalies are always kept


# --------------------------------------------------------------------------- #
# Cached computation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading uploaded file...")
def read_upload(raw: bytes, name: str) -> pd.DataFrame:
    """Parse the uploaded bytes (CICIDS2017 / NSL-KDD / synthetic layouts are auto-detected)."""
    return load_dataset(io.BytesIO(raw))


@st.cache_data(show_spinner="Generating synthetic traffic...")
def make_synthetic(n_flows: int) -> pd.DataFrame:
    """Built-in demo traffic (5 % attacks) so the dashboard is usable with zero setup."""
    return standardize_columns(generate_synthetic_traffic(n_flows))


@st.cache_data(show_spinner="Training detector...")
def run_detection(df: pd.DataFrame, model_name: str, contamination: float, params: tuple) -> dict:
    """Preprocess -> fit detector -> project to 2-D -> compute metrics. Returns plain arrays."""
    features, y, attack = extract_labels(df)               # labels never reach the model
    pre = NetworkPreprocessor().fit(features)
    X = pre.transform(features)
    trained = train_model(model_name, X, contamination, **dict(params))
    # PCA to 2 components purely for VISUALISATION: shows how separable the flagged flows are.
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    out = {
        "labels": trained.labels, "scores": trained.scores,
        "threshold": float(trained.detector.threshold_), "coords": coords,
        "explained": pca.explained_variance_ratio_, "fit_seconds": trained.fit_seconds,
        "features": pre.get_feature_report(), "y": y,
        "attack": None if attack is None else attack.to_numpy(),
        "unsup": compute_unsupervised_metrics(X, trained.labels),
        "sup": None if y is None else compute_supervised_metrics(y, trained.labels, trained.scores),
        "detail": {k: v for k, v in vars(trained.detector).items()
                   if k in ("eps_", "min_samples_", "n_neighbors_", "n_clusters_", "n_estimators")},
    }
    return out


# --------------------------------------------------------------------------- #
# Sidebar: data source + model controls
# --------------------------------------------------------------------------- #
st.sidebar.title("🛡️ SOC Controls")
upload = st.sidebar.file_uploader("Upload flow CSV", type=["csv", "txt"],
                                  help="CICIDS2017, NSL-KDD, or any CSV with numeric flow features.")
if upload is None:
    n_syn = st.sidebar.slider("Synthetic flows (demo data)", 5_000, 50_000, 20_000, step=5_000)
    df = make_synthetic(n_syn)
    source = f"synthetic demo traffic ({n_syn:,} flows)"
else:
    df = read_upload(upload.getvalue(), upload.name)
    source = upload.name

model_name = st.sidebar.selectbox("Detection model", list(MODEL_REGISTRY))
contamination = st.sidebar.slider(
    "Contamination (expected anomaly share)", 0.005, 0.20, 0.05, step=0.005,
    help="Isolation Forest / LOF flag exactly this share of flows (top scores). "
         "For DBSCAN it steers the automatic eps choice.")

# Model-specific advanced parameters (None/auto by default).
params = {}
with st.sidebar.expander("Advanced parameters"):
    if model_name == "Isolation Forest":
        params["n_estimators"] = st.slider("Trees (n_estimators)", 50, 500, 200, step=50)
    elif model_name == "Local Outlier Factor":
        if st.checkbox("Set neighbours manually", False):
            params["n_neighbors"] = st.slider("n_neighbors (k)", 5, 1000, 100)
        else:
            st.caption("k chosen automatically from the contamination rate.")
    else:
        if st.checkbox("Set min_samples manually", False):
            params["min_samples"] = st.slider("min_samples", 3, 1000, 50)
        else:
            st.caption("min_samples chosen automatically; eps from the k-distance heuristic.")
params_key = tuple(sorted(params.items()))       # hashable -> usable as cache key

# --------------------------------------------------------------------------- #
# Run the detector
# --------------------------------------------------------------------------- #
st.title("Network Anomaly Detection - SOC Dashboard")
st.caption(f"Source: {source}  |  Model: {model_name}  |  Unsupervised - labels (if any) are used only for validation")

try:
    res = run_detection(df, model_name, contamination, params_key)
except ValueError as err:
    st.error(f"Could not analyse this file: {err}")
    st.stop()

labels, scores, threshold = res["labels"], res["scores"], res["threshold"]
n_flows, n_anom = len(labels), int(labels.sum())
pct = 100.0 * n_anom / max(n_flows, 1)
severity = compute_severity(scores, threshold)

# Alert frame = original columns + model outputs (identifiers are kept for the analyst).
alerts = df.reset_index(drop=True).copy()
alerts["anomaly_score"] = scores
alerts["severity"] = severity
alerts["is_anomaly"] = labels

# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
level, colour = threat_level(pct)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total flows processed", f"{n_flows:,}")
k2.metric("Anomalies detected", f"{n_anom:,}")
k3.metric("Anomaly percentage", f"{pct:.2f}%")
k4.markdown(
    f"<div style='padding:10px 14px;border-radius:8px;background:{colour}22;border:2px solid {colour}'>"
    f"<div style='font-size:0.8rem;opacity:.75'>System threat level</div>"
    f"<div style='font-size:1.8rem;font-weight:700;color:{colour}'>{level}</div></div>",
    unsafe_allow_html=True)

tab_overview, tab_compare, tab_data = st.tabs(["🔎 Threat overview", "⚖️ Model comparison", "🗂️ Data & features"])

# --------------------------------------------------------------------------- #
# Tab 1: visualisations, alerts, validation
# --------------------------------------------------------------------------- #
with tab_overview:
    left, right = st.columns(2)

    # --- PCA scatter ------------------------------------------------------ #
    with left:
        st.subheader("PCA projection of flows")
        rng = np.random.default_rng(0)
        idx_norm = np.flatnonzero(labels == 0)
        idx_anom = np.flatnonzero(labels == 1)
        if len(idx_norm) > MAX_PLOT_POINTS:          # sub-sample normals only; never hide an alert
            idx_norm = rng.choice(idx_norm, MAX_PLOT_POINTS, replace=False)
        c = res["coords"]
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=c[idx_norm, 0], y=c[idx_norm, 1], mode="markers", name="Normal",
                                   marker=dict(size=4, color="#4C78A8", opacity=0.4)))
        fig.add_trace(go.Scattergl(x=c[idx_anom, 0], y=c[idx_anom, 1], mode="markers", name="Anomaly",
                                   text=[f"severity {s:.0f}" for s in severity[idx_anom]],
                                   marker=dict(size=6, color="#d62728", opacity=0.85)))
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title=f"PC1 ({res['explained'][0]:.0%} variance)",
                          yaxis_title=f"PC2 ({res['explained'][1]:.0%} variance)",
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig)
        st.caption("PCA is only for display - detection runs in the full scaled feature space.")

    # --- Score histogram -------------------------------------------------- #
    with right:
        st.subheader("Anomaly score distribution")
        log_y = st.checkbox("Log y-axis (reveals the tail)", True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scores[labels == 0], name="Normal", nbinsx=80,
                                   marker_color="#4C78A8", opacity=0.75))
        fig.add_trace(go.Histogram(x=scores[labels == 1], name="Anomaly", nbinsx=80,
                                   marker_color="#d62728", opacity=0.75))
        fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                      annotation_text="cut-off", annotation_position="top")
        fig.update_layout(barmode="overlay", height=430, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="anomaly score (higher = more anomalous)", yaxis_title="flows",
                          yaxis_type="log" if log_y else "linear", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig)
        detail = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in res["detail"].items())
        st.caption(f"Cut-off = {threshold:.4f}. Fit time {res['fit_seconds']:.1f}s. {detail}")

    # --- Live alert table --------------------------------------------------- #
    st.subheader("Alert queue (highest severity first)")
    f1, f2, f3 = st.columns([1, 1, 1])
    min_sev = f1.slider("Minimum severity", 0, 100, 0)
    ip_query = f2.text_input("IP contains", "", placeholder="e.g. 10.0.1.")
    attack_choices = sorted(alerts["attack_type"].astype(str).unique()) if "attack_type" in alerts else []
    chosen = f3.multiselect("Ground-truth attack type", attack_choices) if attack_choices else []

    view = alerts[alerts["is_anomaly"] == 1]
    view = view[view["severity"] >= min_sev]
    ip_cols = [c for c in ("src_ip", "dst_ip", "source_ip", "destination_ip") if c in view.columns]
    if ip_query and ip_cols:
        mask = np.zeros(len(view), dtype=bool)
        for col in ip_cols:
            mask |= view[col].astype(str).str.contains(ip_query, regex=False).to_numpy()
        view = view[mask]
    if chosen:
        view = view[view["attack_type"].astype(str).isin(chosen)]
    view = view.sort_values("anomaly_score", ascending=False)

    lead = [c for c in ("severity", "anomaly_score", "timestamp", "src_ip", "dst_ip", "dst_port",
                        "attack_type") if c in view.columns]
    ordered = lead + [c for c in view.columns if c not in lead and c not in ("is_anomaly", "label")]
    st.caption(f"{len(view):,} alerts match the filters")
    st.dataframe(view[ordered].head(2000), hide_index=True, height=420,
                 column_config={"severity": st.column_config.ProgressColumn(
                     "Severity", min_value=0, max_value=100, format="%.0f")})
    st.download_button("⬇️ Download alerts (CSV)", view[ordered].to_csv(index=False).encode(),
                       file_name="anomaly_alerts.csv", mime="text/csv")

    # --- Validation (only if the file carried labels) ---------------------- #
    with st.expander("Validation metrics", expanded=True):
        u = res["unsup"]
        st.write(f"**Silhouette (normal vs flagged):** {u['silhouette']:.3f}" if not np.isnan(u["silhouette"])
                 else "**Silhouette:** n/a (needs at least two groups)")
        if res["sup"] is None:
            st.info("No label column found - showing unsupervised metrics only.")
        else:
            s = res["sup"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{s['precision']:.3f}")
            m2.metric("Recall", f"{s['recall']:.3f}")
            m3.metric("F1-score", f"{s['f1']:.3f}")
            m4.metric("ROC-AUC", f"{s['roc_auc']:.3f}")
            cm_col, br_col = st.columns(2)
            cm = s["confusion_matrix"]
            cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               x=["Pred normal", "Pred anomaly"], y=["Actual normal", "Actual anomaly"])
            cm_fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
            cm_col.plotly_chart(cm_fig)
            if res["attack"] is not None:
                by_attack = (pd.DataFrame({"attack_type": res["attack"], "detected": labels})
                             .groupby("attack_type")["detected"].agg(flows="count", detected="sum"))
                by_attack["recall"] = (by_attack["detected"] / by_attack["flows"]).round(3)
                br_col.write("**Detection by attack type**")
                br_col.dataframe(by_attack)

# --------------------------------------------------------------------------- #
# Tab 2: run all three detectors side by side
# --------------------------------------------------------------------------- #
with tab_compare:
    st.write("All three detectors on the same data, using the contamination rate from the sidebar "
             "and automatic hyper-parameters.")
    if st.button("Run comparison", type="primary"):
        rows = []
        for name in MODEL_REGISTRY:
            r = run_detection(df, name, contamination, ())
            row = {"model": name, "flagged": r["unsup"]["n_anomalies"],
                   "flagged %": round(r["unsup"]["anomaly_pct"], 2),
                   "silhouette": r["unsup"]["silhouette"], "fit seconds": round(r["fit_seconds"], 2)}
            if r["sup"] is not None:
                row.update({k: r["sup"][k] for k in ("precision", "recall", "f1", "roc_auc")})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("model").round(3))
        st.caption("Density methods (LOF, DBSCAN) can miss attack bursts that form dense clusters; "
                   "Isolation Forest is usually the most robust default for flow data.")

# --------------------------------------------------------------------------- #
# Tab 3: data preview and feature report
# --------------------------------------------------------------------------- #
with tab_data:
    st.write("**Features used by the model** (identifiers and labels are excluded automatically)")
    st.dataframe(res["features"], hide_index=True)
    st.caption("log1p_applied = the column was heavily right-skewed, so log(1+x) was applied before scaling.")
    st.write("**Raw data preview**")
    st.dataframe(df.head(200), hide_index=True)
