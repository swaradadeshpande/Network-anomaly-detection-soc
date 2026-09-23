"""
app.py (v2) - Interactive, dark-themed SOC dashboard for Network Anomaly Detection
==================================================================================

Run with:   streamlit run app.py

What's new compared to v1
-------------------------
* Dark "cyber" theme, hero header, KPI cards and a threat-level gauge.
* PCA map in 2D or 3D, coloured by alert severity, with rich hover details.
* Anomaly timeline, top attacker IPs, targeted ports, severity mix.
* "Investigate" tab: click any alert row to see WHY it was flagged
  (which features deviate most from normal traffic).
* Model comparison with charts.

Detection logic is unchanged: NetworkPreprocessor -> unsupervised detector
(fit on the uploaded batch) -> scores/labels. Labels, if present, are only
used for validation panels.
"""

import html
import inspect
import io
import sys
from pathlib import Path

# Make `import src...` work regardless of the directory Streamlit is started from.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from src.evaluator import (compute_severity, compute_supervised_metrics,
                           compute_unsupervised_metrics, threat_level)
from src.generate_data import generate_synthetic_traffic
from src.preprocessor import (BENIGN_VALUES, NetworkPreprocessor, extract_labels,
                              load_dataset, standardize_columns)
from src.train import MODEL_REGISTRY, train_model

# set_page_config must be the first Streamlit call.
st.set_page_config(page_title="SOC | Network Anomaly Detection", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
CYAN, PURPLE, RED, AMBER, GREEN, SLATE = "#22d3ee", "#8b5cf6", "#ef4444", "#f59e0b", "#22c55e", "#64748b"
GRID = "rgba(148,163,184,0.15)"
ATTACK_COLORS = {"PortScan": AMBER, "DoS": RED, "Exfiltration": PURPLE, "BENIGN": SLATE, "Anomaly": RED}
MODEL_COLORS = {"Isolation Forest": CYAN, "DBSCAN": PURPLE, "Local Outlier Factor": AMBER}
MAX_PLOT_POINTS = 8000

# Streamlit renamed use_container_width -> width="stretch" in newer versions; support both.
WIDE = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
        else {"use_container_width": True})

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #14213d 0%, #0b1220 55%); color:#e5e7eb; }
.block-container { padding-top: 1.2rem; max-width: 1600px; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f172a 0%,#0b1220 100%);
                            border-right: 1px solid #1e2a44; }

/* hero banner */
.hero { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px;
        padding:22px 26px; border-radius:18px; margin-bottom:18px;
        background: linear-gradient(120deg, rgba(34,211,238,.16), rgba(139,92,246,.16) 55%, rgba(239,68,68,.10));
        border:1px solid rgba(148,163,184,.22); box-shadow:0 10px 40px rgba(0,0,0,.35); }
.hero-title { font-size:1.9rem; font-weight:800; letter-spacing:-.02em; color:#f8fafc; }
.hero-sub { color:#94a3b8; font-size:.95rem; margin-top:2px; }
.pill { display:inline-flex; align-items:center; gap:7px; padding:6px 13px; margin:3px 4px 3px 0;
        border-radius:999px; font-size:.78rem; font-weight:600; color:#cbd5e1;
        background:rgba(15,23,42,.75); border:1px solid rgba(148,163,184,.25); }
.pill.live { color:#86efac; border-color:rgba(34,197,94,.45); }
.dot { width:8px; height:8px; border-radius:50%; background:#22c55e; animation:pulse 1.6s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(34,197,94,.7)} 70%{box-shadow:0 0 0 9px rgba(34,197,94,0)}
                   100%{box-shadow:0 0 0 0 rgba(34,197,94,0)} }

/* KPI cards */
.kpi { position:relative; overflow:hidden; padding:18px 18px 16px; border-radius:16px; height:150px;
       background:linear-gradient(160deg, rgba(30,41,68,.85), rgba(15,23,42,.9));
       border:1px solid rgba(148,163,184,.2); transition:transform .18s, border-color .18s; }
.kpi:hover { transform:translateY(-3px); border-color:var(--accent); }
.kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent); }
.kpi::after { content:""; position:absolute; right:-30px; top:-30px; width:110px; height:110px; border-radius:50%;
              background:var(--accent); opacity:.10; filter:blur(4px); }
.kpi-icon { font-size:1.35rem; }
.kpi-label { color:#94a3b8; font-size:.74rem; font-weight:600; text-transform:uppercase; letter-spacing:.09em; margin-top:6px; }
.kpi-value { color:#f8fafc; font-size:2.1rem; font-weight:800; line-height:1.15; letter-spacing:-.02em; }
.kpi-sub { color:#64748b; font-size:.78rem; margin-top:2px; }

/* headings, badges, detail table */
.sec { font-size:1.05rem; font-weight:700; color:#f1f5f9; margin:4px 0 2px; }
.sec-sub { color:#94a3b8; font-size:.82rem; margin-bottom:6px; }
.badge { display:inline-block; padding:3px 11px; border-radius:8px; font-weight:700; font-size:.8rem; }
.kv { width:100%; border-collapse:collapse; font-size:.86rem; }
.kv td { padding:6px 4px; border-bottom:1px solid rgba(148,163,184,.14); }
.kv td:first-child { color:#94a3b8; width:46%; } .kv td:last-child { color:#f1f5f9; font-weight:600; text-align:right; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid #1e2a44; }
.stTabs [data-baseweb="tab"] { padding:10px 18px; border-radius:10px 10px 0 0; font-weight:600; color:#94a3b8; }
.stTabs [aria-selected="true"] { color:#22d3ee !important; background:rgba(34,211,238,.08); }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; }
.stButton>button, .stDownloadButton>button { border-radius:10px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def style(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the shared dark, transparent look to every Plotly figure."""
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=28, b=8),
                      font=dict(color="#cbd5e1", family="Inter, system-ui, sans-serif", size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def show(fig: go.Figure, height: int = 380, **kw) -> None:
    st.plotly_chart(style(fig, height), **WIDE, **kw)


def section(title: str, sub: str = "") -> None:
    st.markdown(f"<div class='sec'>{title}</div><div class='sec-sub'>{sub}</div>", unsafe_allow_html=True)


def kpi(icon: str, label: str, value: str, sub: str, accent: str) -> str:
    return (f"<div class='kpi' style='--accent:{accent}'><div class='kpi-icon'>{icon}</div>"
            f"<div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div>"
            f"<div class='kpi-sub'>{sub}</div></div>")


def first_col(df: pd.DataFrame, candidates) -> "str | None":
    return next((c for c in candidates if c in df.columns), None)


# --------------------------------------------------------------------------- #
# Cached computation (unchanged logic, now also returns X and a 3-D PCA)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading uploaded file...")
def read_upload(raw: bytes, name: str) -> pd.DataFrame:
    return load_dataset(io.BytesIO(raw))


@st.cache_data(show_spinner="Generating synthetic traffic...")
def make_synthetic(n_flows: int) -> pd.DataFrame:
    return standardize_columns(generate_synthetic_traffic(n_flows))


@st.cache_data(show_spinner="Training detector...")
def run_detection(df: pd.DataFrame, model_name: str, contamination: float, params: tuple) -> dict:
    features, y, attack = extract_labels(df)                 # labels never reach the model
    pre = NetworkPreprocessor().fit(features)
    X = pre.transform(features)
    trained = train_model(model_name, X, contamination, **dict(params))
    pca = PCA(n_components=min(3, X.shape[1]), random_state=42)   # 3 comps: 2-D and 3-D views
    coords = pca.fit_transform(X)
    if coords.shape[1] < 3:                                   # pad if the data has <3 features
        coords = np.hstack([coords, np.zeros((len(coords), 3 - coords.shape[1]))])
    return {
        "labels": trained.labels, "scores": trained.scores,
        "threshold": float(trained.detector.threshold_), "coords": coords,
        "explained": np.r_[pca.explained_variance_ratio_, np.zeros(3)][:3],
        "fit_seconds": trained.fit_seconds, "features": pre.get_feature_report(),
        "X": X.astype("float32"), "y": y, "attack": None if attack is None else attack.to_numpy(),
        "unsup": compute_unsupervised_metrics(X, trained.labels),
        "sup": None if y is None else compute_supervised_metrics(y, trained.labels, trained.scores),
        "detail": {k: v for k, v in vars(trained.detector).items()
                   if k in ("eps_", "min_samples_", "n_neighbors_", "n_clusters_", "n_estimators")},
    }


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("<div style='font-size:1.35rem;font-weight:800;color:#f8fafc'>🛡️ SOC Console</div>"
                "<div style='color:#64748b;font-size:.8rem;margin-bottom:10px'>Unsupervised threat hunting</div>",
                unsafe_allow_html=True)
    st.markdown("**1 · Data**")
    upload = st.file_uploader("Upload flow CSV", type=["csv", "txt"], label_visibility="collapsed",
                              help="CICIDS2017, NSL-KDD, or any CSV with numeric flow features.")
    if upload is None:
        n_syn = st.slider("Synthetic demo flows", 5_000, 50_000, 20_000, step=5_000)
        df = make_synthetic(n_syn)
        source = f"Synthetic ({n_syn:,} flows)"
    else:
        df = read_upload(upload.getvalue(), upload.name)
        source = upload.name

    st.markdown("**2 · Detection model**")
    model_name = st.radio("Model", list(MODEL_REGISTRY), label_visibility="collapsed",
                          captions=["Fast, robust default", "Density noise = anomaly", "Local density outliers"])
    contamination = st.slider("Expected anomaly share", 0.005, 0.20, 0.05, step=0.005,
                              help="Isolation Forest / LOF flag exactly this share of flows. "
                                   "For DBSCAN it steers the automatic eps.")
    params = {}
    with st.expander("⚙️ Advanced parameters"):
        if model_name == "Isolation Forest":
            params["n_estimators"] = st.slider("Trees", 50, 500, 200, step=50)
        elif model_name == "Local Outlier Factor":
            if st.checkbox("Set neighbours manually", False):
                params["n_neighbors"] = st.slider("n_neighbors (k)", 5, 1000, 100)
            else:
                st.caption("k is chosen automatically from the anomaly share.")
        else:
            if st.checkbox("Set min_samples manually", False):
                params["min_samples"] = st.slider("min_samples", 3, 1000, 50)
            else:
                st.caption("min_samples automatic; eps from the k-distance heuristic.")
    params_key = tuple(sorted(params.items()))
    if st.button("🔄 Clear cache & re-run"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption("Labels (if the file has them) are used only to validate - never to detect.")

# --------------------------------------------------------------------------- #
# Run detector and derive everything the UI needs
# --------------------------------------------------------------------------- #
try:
    res = run_detection(df, model_name, contamination, params_key)
except ValueError as err:
    st.error(f"Could not analyse this file: {err}")
    st.stop()

labels, scores, threshold, X = res["labels"], res["scores"], res["threshold"], res["X"]
n_flows, n_anom = len(labels), int(labels.sum())
pct = 100.0 * n_anom / max(n_flows, 1)
severity = compute_severity(scores, threshold)
level, level_colour = threat_level(pct)
n_critical = int((severity >= 75).sum())

alerts = df.reset_index(drop=True).copy()
alerts["anomaly_score"], alerts["severity"], alerts["is_anomaly"] = scores, severity, labels
an = alerts[alerts["is_anomaly"] == 1]

src_col = first_col(alerts, ("src_ip", "source_ip"))
dst_col = first_col(alerts, ("dst_ip", "destination_ip"))
port_col = first_col(alerts, ("dst_port", "destination_port"))
time_col = first_col(alerts, ("timestamp", "time", "date"))
has_attack = "attack_type" in alerts.columns

run_key = (source, model_name, contamination, params_key)
if st.session_state.get("last_run") != run_key:               # toast once per new analysis
    st.toast(f"{model_name}: {n_anom:,} anomalies in {n_flows:,} flows", icon="🛡️")
    st.session_state["last_run"] = run_key

# --------------------------------------------------------------------------- #
# Hero + KPI row
# --------------------------------------------------------------------------- #
st.markdown(f"""
<div class='hero'>
  <div><div class='hero-title'>Network Anomaly Detection</div>
       <div class='hero-sub'>Security Operations Center · unsupervised machine-learning threat hunting</div></div>
  <div><span class='pill live'><span class='dot'></span>ANALYSIS COMPLETE</span>
       <span class='pill'>🧠 {html.escape(model_name)}</span>
       <span class='pill'>📁 {html.escape(source)}</span>
       <span class='pill'>⏱ {res['fit_seconds']:.1f}s</span></div>
</div>""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.5])
c1.markdown(kpi("📡", "Flows processed", f"{n_flows:,}", "network flows analysed", CYAN), unsafe_allow_html=True)
c2.markdown(kpi("🚨", "Anomalies detected", f"{n_anom:,}", "flagged as suspicious", RED), unsafe_allow_html=True)
c3.markdown(kpi("📊", "Anomaly rate", f"{pct:.2f}%", "of total traffic", AMBER), unsafe_allow_html=True)
c4.markdown(kpi("🔥", "Critical alerts", f"{n_critical:,}", "severity ≥ 75 / 100", PURPLE), unsafe_allow_html=True)
with c5:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pct, number=dict(suffix="%", font=dict(size=30, color="#f8fafc")),
        gauge=dict(axis=dict(range=[0, 20], tickcolor="#64748b"), bar=dict(color=level_colour, thickness=0.28),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, 1], color="rgba(34,197,94,.30)"), dict(range=[1, 3], color="rgba(143,191,63,.30)"),
                          dict(range=[3, 8], color="rgba(245,158,11,.30)"), dict(range=[8, 15], color="rgba(232,89,12,.30)"),
                          dict(range=[15, 20], color="rgba(214,39,40,.35)")])))
    gauge.update_layout(template="plotly_dark", height=150, margin=dict(l=18, r=18, t=18, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"))
    st.markdown(f"<div style='text-align:center;margin-bottom:-6px'><span class='kpi-label'>System threat level</span> "
                f"<span class='badge' style='background:{level_colour}33;color:{level_colour};"
                f"border:1px solid {level_colour}'>{level}</span></div>", unsafe_allow_html=True)
    st.plotly_chart(gauge, **WIDE)

st.write("")
tab_over, tab_attack, tab_inv, tab_cmp, tab_data = st.tabs(
    ["🔎 Overview", "🎯 Attack analysis", "🕵️ Investigate", "⚖️ Model comparison", "🗂️ Data & features"])


# =========================================================================== #
# TAB 1 - Overview
# =========================================================================== #
def hover_text(idx: np.ndarray) -> list:
    """Rich tooltip for flagged flows: severity, endpoints, ground-truth type."""
    out = []
    for i in idx:
        row = alerts.iloc[i]
        lines = [f"<b>Severity {severity[i]:.0f}</b>"]
        if src_col and dst_col:
            port = f":{row[port_col]}" if port_col else ""
            lines.append(f"{row[src_col]} → {row[dst_col]}{port}")
        if has_attack:
            lines.append(f"type: {row['attack_type']}")
        out.append("<br>".join(lines))
    return out


with tab_over:
    left, right = st.columns(2)
    with left, st.container(border=True):
        top = st.columns([3, 1])
        with top[0]:
            section("Traffic map (PCA)", "Each dot is a flow; anomalies are coloured by severity")
        three_d = top[1].toggle("3D view", False)
        rng = np.random.default_rng(0)
        idx_n, idx_a = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
        cap = 3000 if three_d else MAX_PLOT_POINTS            # never hide alerts in 2-D; cap for 3-D speed
        if len(idx_n) > cap:
            idx_n = rng.choice(idx_n, cap, replace=False)
        if three_d and len(idx_a) > 3000:
            idx_a = rng.choice(idx_a, 3000, replace=False)
        cs = res["coords"]
        heat = [[0, "#fbbf24"], [0.5, "#f97316"], [1, "#ef4444"]]
        cbar = dict(title="Severity", thickness=10, len=0.6)
        if three_d:
            fig = go.Figure([
                go.Scatter3d(x=cs[idx_n, 0], y=cs[idx_n, 1], z=cs[idx_n, 2], mode="markers", name="Normal",
                             marker=dict(size=2, color=CYAN, opacity=0.35), hoverinfo="skip"),
                go.Scatter3d(x=cs[idx_a, 0], y=cs[idx_a, 1], z=cs[idx_a, 2], mode="markers", name="Anomaly",
                             text=hover_text(idx_a), hoverinfo="text",
                             marker=dict(size=3.5, color=severity[idx_a], colorscale=heat, cmin=0, cmax=100,
                                         colorbar=cbar, opacity=0.95))])
            ax = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, zerolinecolor=GRID, showbackground=False)
            fig.update_layout(scene=dict(xaxis=dict(title="PC1", **ax), yaxis=dict(title="PC2", **ax),
                                         zaxis=dict(title="PC3", **ax)))
        else:
            fig = go.Figure([
                go.Scattergl(x=cs[idx_n, 0], y=cs[idx_n, 1], mode="markers", name="Normal", hoverinfo="skip",
                             marker=dict(size=4, color=CYAN, opacity=0.30)),
                go.Scattergl(x=cs[idx_a, 0], y=cs[idx_a, 1], mode="markers", name="Anomaly",
                             text=hover_text(idx_a), hoverinfo="text",
                             marker=dict(size=7, color=severity[idx_a], colorscale=heat, cmin=0, cmax=100,
                                         colorbar=cbar, opacity=0.9, line=dict(width=0.5, color="#0b1220")))])
            fig.update_layout(xaxis_title=f"PC1 ({res['explained'][0]:.0%} variance)",
                              yaxis_title=f"PC2 ({res['explained'][1]:.0%} variance)")
        show(fig, 440)
        st.caption("PCA is for display only - detection runs in the full scaled feature space.")

    with right, st.container(border=True):
        top = st.columns([3, 1])
        with top[0]:
            section("Anomaly score distribution", "Flows right of the dashed cut-off are flagged")
        log_y = top[1].toggle("Log y", True)
        fig = go.Figure([
            go.Histogram(x=scores[labels == 0], name="Normal", nbinsx=80, marker_color=CYAN, opacity=0.75),
            go.Histogram(x=scores[labels == 1], name="Anomaly", nbinsx=80, marker_color=RED, opacity=0.85)])
        fig.add_vline(x=threshold, line_dash="dash", line_color=AMBER, annotation_text="cut-off",
                      annotation_font_color=AMBER)
        fig.update_layout(barmode="overlay", xaxis_title="anomaly score (higher = more anomalous)",
                          yaxis_title="flows", yaxis_type="log" if log_y else "linear")
        show(fig, 440)
        det = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in res["detail"].items())
        st.caption(f"Cut-off {threshold:.4f} · {det}")

    with st.container(border=True):
        section("Anomalies over time", "Flagged flows per time bucket (line = all traffic)")
        t = pd.to_datetime(alerts[time_col], errors="coerce") if time_col else None
        if t is not None and t.notna().mean() > 0.5 and t.max() > t.min():
            step = f"{max(int((t.max() - t.min()).total_seconds() / 48), 1)}s"
            bucket = t.dt.floor(step)
            fig = go.Figure()
            kind = an["attack_type"].astype(str) if has_attack else pd.Series("Anomaly", index=an.index)
            for name in sorted(kind.unique()):
                counts = bucket[an.index][kind == name].value_counts().sort_index()
                fig.add_trace(go.Bar(x=counts.index, y=counts.values, name=name,
                                     marker_color=ATTACK_COLORS.get(name, PURPLE)))
            total = bucket.value_counts().sort_index()
            fig.add_trace(go.Scatter(x=total.index, y=total.values, name="All flows", yaxis="y2",
                                     line=dict(color="rgba(148,163,184,.6)", width=1.5)))
            fig.update_layout(barmode="stack", yaxis_title="flagged flows",
                              yaxis2=dict(overlaying="y", side="right", showgrid=False, title="all flows"))
            show(fig, 320)
        else:
            st.info("No usable timestamp column in this file, so the timeline is unavailable.")

# =========================================================================== #
# TAB 2 - Attack analysis
# =========================================================================== #
with tab_attack:
    a1, a2 = st.columns(2)
    with a1, st.container(border=True):
        section("Severity mix", "How urgent are the flagged flows?")
        sev_an = severity[labels == 1]
        bands = [("Critical", 75, 101, RED), ("High", 50, 75, "#f97316"), ("Medium", 25, 50, AMBER), ("Low", 0, 25, GREEN)]
        counts = [int(((sev_an >= lo) & (sev_an < hi)).sum()) for _, lo, hi, _ in bands]
        fig = go.Figure(go.Pie(labels=[b[0] for b in bands], values=counts, hole=0.62, sort=False,
                               marker=dict(colors=[b[3] for b in bands], line=dict(color="#0b1220", width=2)),
                               textinfo="label+percent"))
        fig.update_layout(annotations=[dict(text=f"<b>{n_anom:,}</b><br>alerts", showarrow=False, font=dict(size=18))],
                          showlegend=False)
        show(fig, 340)

    with a2, st.container(border=True):
        if has_attack:
            section("Detection by attack type", "Ground truth vs what the model caught")
            types = alerts.assign(hit=labels).groupby(alerts["attack_type"].astype(str))["hit"].agg(["sum", "count"])
            bad = types[~types.index.str.lower().isin(BENIGN_VALUES)]
            fig = go.Figure([go.Bar(y=bad.index, x=bad["sum"], name="Detected", orientation="h", marker_color=GREEN),
                             go.Bar(y=bad.index, x=bad["count"] - bad["sum"], name="Missed", orientation="h",
                                    marker_color=RED)])
            fig.update_layout(barmode="stack", xaxis_title="flows")
            show(fig, 300)
            fp = int(types.loc[types.index.str.lower().isin(BENIGN_VALUES), "sum"].sum())
            st.caption(f"False alarms (benign flows flagged): {fp:,}")
        else:
            section("Alert severity histogram", "No ground-truth labels in this file")
            show(go.Figure(go.Histogram(x=sev_an, nbinsx=25, marker_color=PURPLE)), 340)

    b1, b2 = st.columns(2)
    with b1, st.container(border=True):
        section("Top source IPs", "Most frequent origins among alerts")
        if src_col and len(an):
            vc = an[src_col].astype(str).value_counts().head(10)
            fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h",
                                   marker=dict(color=vc.values, colorscale=[[0, PURPLE], [1, RED]])))
            fig.update_yaxes(autorange="reversed")
            show(fig, 340)
        else:
            st.info("No source-IP column available.")
    with b2, st.container(border=True):
        section("Targeted destination ports", "Where the alerts are aimed")
        if port_col and len(an):
            vc = an[port_col].astype(str).value_counts().head(10)
            fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker=dict(color=vc.values, colorscale=[[0, CYAN], [1, PURPLE]])))
            fig.update_xaxes(type="category", title="destination port")
            show(fig, 340)
        else:
            st.info("No destination-port column available.")

    with st.container(border=True):
        section("What makes anomalies different?",
                "Average standardised feature value of flagged flows minus normal flows (log-scaled space)")
        if 0 < n_anom < n_flows:
            dev = X[labels == 1].mean(axis=0) - X[labels == 0].mean(axis=0)
            names = list(res["features"]["feature"])
            order = np.argsort(dev)
            fig = go.Figure(go.Bar(x=dev[order], y=[names[i] for i in order], orientation="h",
                                   marker_color=[RED if d > 0 else CYAN for d in dev[order]]))
            fig.update_layout(xaxis_title="difference in standard deviations (red = higher in anomalies)")
            show(fig, 380)

# =========================================================================== #
# TAB 3 - Investigate
# =========================================================================== #
with tab_inv:
    section("Alert queue", "Click a row to open the flow investigation panel")
    f1, f2, f3 = st.columns(3)
    min_sev = f1.slider("Minimum severity", 0, 100, 0)
    ip_query = f2.text_input("IP contains", "", placeholder="e.g. 10.0.1.")
    chosen = f3.multiselect("Ground-truth type", sorted(alerts["attack_type"].astype(str).unique())) if has_attack else []

    view = alerts[(alerts["is_anomaly"] == 1) & (alerts["severity"] >= min_sev)]
    if ip_query:
        mask = np.zeros(len(view), dtype=bool)
        for col in (src_col, dst_col):
            if col:
                mask |= view[col].astype(str).str.contains(ip_query, regex=False).to_numpy()
        view = view[mask]
    if chosen:
        view = view[view["attack_type"].astype(str).isin(chosen)]
    view = view.sort_values("anomaly_score", ascending=False)

    lead = [c for c in ("severity", "anomaly_score", "timestamp", "src_ip", "dst_ip", "dst_port", "attack_type")
            if c in view.columns]
    cols = lead + [c for c in view.columns if c not in lead and c not in ("is_anomaly", "label")]
    shown = view[cols].head(1000)
    st.caption(f"{len(view):,} alerts match · showing top {len(shown):,}")
    event = st.dataframe(shown, hide_index=True, height=340, on_select="rerun", selection_mode="single-row",
                         **{k: v for k, v in WIDE.items()},
                         column_config={"severity": st.column_config.ProgressColumn(
                             "Severity", min_value=0, max_value=100, format="%.0f")})
    st.download_button("⬇️ Export filtered alerts (CSV)", view[cols].to_csv(index=False).encode(),
                       file_name="anomaly_alerts.csv", mime="text/csv")

    st.write("")
    with st.container(border=True):
        section("Flow investigation")
        picked = event.selection.rows if event and event.selection else []
        if not picked:
            st.info("👆 Select an alert above to see why it was flagged.")
        else:
            orig = shown.index[picked[0]]                  # index into alerts / X
            row = alerts.loc[orig]
            sev = row["severity"]
            col_sev = RED if sev >= 75 else "#f97316" if sev >= 50 else AMBER if sev >= 25 else GREEN
            d1, d2 = st.columns([1, 1.4])
            with d1:
                st.markdown(f"<span class='badge' style='background:{col_sev}33;color:{col_sev};border:1px solid {col_sev}'>"
                            f"SEVERITY {sev:.0f} / 100</span>", unsafe_allow_html=True)
                rows_html = "".join(
                    f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(row[k]))}</td></tr>"
                    for k in [c for c in ("timestamp", "src_ip", "dst_ip", "dst_port", "protocol", "attack_type")
                              if c in alerts.columns] + list(res["features"]["feature"]))
                st.markdown(f"<table class='kv'>{rows_html}</table>", unsafe_allow_html=True)
            with d2:
                z = X[orig]
                names = list(res["features"]["feature"])
                order = np.argsort(np.abs(z))[::-1]
                top = [f"**{names[i]}** ({z[i]:+.1f}σ)" for i in order[:3]]
                st.markdown("**Why flagged:** most unusual features are " + ", ".join(top) + ".")
                fig = go.Figure(go.Bar(x=z[order[::-1]], y=[names[i] for i in order[::-1]], orientation="h",
                                       marker_color=[RED if v > 0 else CYAN for v in z[order[::-1]]]))
                fig.update_layout(xaxis_title="standard deviations from typical traffic (after log transform)")
                show(fig, 380)

# =========================================================================== #
# TAB 4 - Model comparison
# =========================================================================== #
with tab_cmp:
    section("Compare all detectors", "Same data and anomaly share, automatic hyper-parameters")
    cmp_key = (source, len(df), contamination)
    if st.button("▶ Run comparison", type="primary"):
        rows = []
        with st.spinner("Training all three models (DBSCAN can take a few seconds)..."):
            for name in MODEL_REGISTRY:
                r = run_detection(df, name, contamination, ())
                row = {"model": name, "flagged": r["unsup"]["n_anomalies"], "flagged %": r["unsup"]["anomaly_pct"],
                       "silhouette": r["unsup"]["silhouette"], "fit seconds": r["fit_seconds"]}
                if r["sup"] is not None:
                    row.update({k: r["sup"][k] for k in ("precision", "recall", "f1", "roc_auc")})
                rows.append(row)
        st.session_state["cmp"] = (cmp_key, pd.DataFrame(rows))
    saved = st.session_state.get("cmp")
    if saved and saved[0] == cmp_key:
        table = saved[1]
        if "precision" in table.columns:
            long = table.melt(id_vars="model", value_vars=["precision", "recall", "f1", "roc_auc"],
                              var_name="metric", value_name="score")
            fig = px.bar(long, x="metric", y="score", color="model", barmode="group",
                         color_discrete_map=MODEL_COLORS, range_y=[0, 1.05], text_auto=".2f")
        else:
            fig = px.bar(table, x="model", y="flagged %", color="model", color_discrete_map=MODEL_COLORS)
        with st.container(border=True):
            show(fig, 380)
        st.dataframe(table.set_index("model").round(3), **WIDE)
        st.caption("Density methods (LOF, DBSCAN) can miss attack bursts that form dense clusters; "
                   "Isolation Forest is usually the most robust default for flow data.")
    else:
        st.info("Press **Run comparison** to train all three models on the current data.")

# =========================================================================== #
# TAB 5 - Data & features
# =========================================================================== #
with tab_data:
    d1, d2 = st.columns([1, 1.6])
    with d1, st.container(border=True):
        section("Features used", "Identifiers and labels are excluded automatically")
        st.dataframe(res["features"], hide_index=True, height=360, **WIDE)
        st.caption("log1p_applied = heavily right-skewed column, so log(1+x) was applied before scaling.")
    with d2, st.container(border=True):
        section("Raw data preview", f"First 200 of {n_flows:,} rows")
        st.dataframe(df.head(200), hide_index=True, height=360, **WIDE)
    with st.expander("ℹ️ How detection works"):
        st.markdown("""
1. **Preprocess** - drop IPs/ports/timestamps/labels, fix ±inf/NaN, `log(1+x)` on skewed columns, standardise.
2. **Detect** - Isolation Forest (random cuts isolate rare points), DBSCAN (noise = anomaly) or LOF (local density outliers).
3. **Score** - every flow gets an anomaly score; flows above the cut-off are alerts, ranked by a 0-100 severity.
4. **Validate** - if the file has labels, precision / recall / F1 / ROC-AUC are computed (never used for training).
""")
