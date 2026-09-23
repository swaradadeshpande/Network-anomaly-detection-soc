# Network Anomaly Detection System

Unsupervised detection of suspicious network flows (port scans, DoS bursts, data
exfiltration) with **Isolation Forest**, **DBSCAN** and **Local Outlier Factor**,
plus an interactive **Streamlit SOC dashboard**. No labels are needed to detect;
labels (when present) are used only to validate.

## 1. Overview

| Capability | Where |
|---|---|
| Realistic synthetic traffic (95 % benign / 5 % attacks) | `src/generate_data.py` |
| Cleaning, `log(1+x)` skew reduction, scaling, ID removal | `src/preprocessor.py` |
| Three unsupervised detectors + `.pkl` persistence | `src/train.py` |
| Precision / Recall / F1 / ROC-AUC / Silhouette / confusion matrix | `src/evaluator.py` |
| SOC dashboard (KPIs, PCA map, score histogram, alert queue) | `app.py` |
| One-command pipeline | `run.py` |

## 2. Architecture

```
 CSV / synthetic generator
          │
          ▼
 ┌────────────────────┐   drop IPs/ports/timestamps/labels
 │ NetworkPreprocessor│   inf→NaN → median impute → log1p (skewed) → StandardScaler
 └─────────┬──────────┘
           ▼   X (n_flows × n_features, scaled)
 ┌──────────────────────────────────────────────┐
 │ Isolation Forest │ DBSCAN │ Local Outlier Factor │   common API:
 └─────────┬────────────────────────────────────┘   score ↑ = more anomalous, anomaly ⇔ score > threshold
           ▼
 labels + scores ──► evaluator (metrics, severity 0-100, threat level)
           │
           ├──► models/*.pkl, metrics.csv/json, reports/*.png   (run.py)
           └──► Streamlit SOC dashboard                          (app.py)
```

## 3. Installation

Requires Python 3.9+.

**macOS / Linux**
```bash
cd network_anomaly_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd network_anomaly_detection
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (cmd.exe)**
```bat
cd network_anomaly_detection
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 4. Usage

```bash
python run.py                       # generate data → train → evaluate → open dashboard
python run.py --no-dashboard        # headless run, prints the comparison table
python run.py --n-flows 50000 --contamination 0.03
streamlit run app.py                # dashboard only (works without running run.py first)
```

Individual steps:
```bash
python -m src.generate_data --n-flows 20000 --output data/synthetic_traffic.csv
python -m src.train --data data/synthetic_traffic.csv --contamination 0.05
```

Outputs of `run.py`: `data/synthetic_traffic.csv`, `models/{isolation_forest,dbscan,local_outlier_factor}.pkl`
(each contains preprocessor + detector), `models/metrics.csv|json`, `models/reports/*.png`.

Reusing a saved model on new flows:
```python
from src.train import load_artifact, score_dataframe
import pandas as pd
labels, scores = score_dataframe(load_artifact("models/isolation_forest.pkl"), pd.read_csv("new_flows.csv"))
```
Run this from the project root (the pickles reference the `src` package).
Only load `.pkl` files you created yourself - pickle can execute code.

### Dashboard
* **Sidebar** – CSV upload (or built-in synthetic traffic), model selector, contamination slider, advanced parameters.
* **KPI cards** – flows processed, anomalies, anomaly %, threat level.
* **Threat overview tab** – PCA scatter (red = anomaly), score histogram with cut-off, filterable alert queue sorted by severity (CSV export), validation metrics + confusion matrix when labels exist.
* **Model comparison tab** – all three models side by side.
* **Data & features tab** – which columns were used and which were log-transformed.

The dashboard fits the chosen model on the uploaded batch (typical threat-hunting use). Persisted `.pkl` artifacts are for scoring new traffic programmatically.

### Using real datasets
* **CICIDS2017**: upload or pass any of the daily CSVs (`python run.py --data <file>.csv --max-rows 50000`). Headers such as `Flow Bytes/s` and `Total Length of Fwd Packets` are mapped automatically, `Infinity` values are handled, and `Label` (`BENIGN` vs. attack names) becomes ground truth.
* **NSL-KDD**: `KDDTrain+.txt` / `KDDTest+.txt` (header-less) is detected and named automatically; symbolic columns (`protocol_type`, `service`, `flag`) are ignored.
* **Any other CSV**: all numeric non-identifier columns are used; a `label` column (0/1 or text with `benign`/`normal` = normal) enables validation.
* DBSCAN/LOF memory grows with rows, so `run.py` samples large files (`--max-rows`).

## 5. Algorithm breakdown

**Preprocessing (why it matters).** Distances and densities are dominated by large-range columns, so features are standardised. Flow metrics are log-normal, so skewed non-negative columns (skewness > 1) get `log(1+x)` *before* scaling; otherwise one huge flow flattens all normal traffic. Identifiers are dropped so models learn behaviour, not hosts. Infinities become NaN and are imputed with medians.

**Isolation Forest.** Random axis-parallel splits isolate anomalies in fewer steps because they are few and different. Score = 2^(−E[h(x)]/c(n)). `contamination` sets the cut-off quantile; `n_estimators=200` stabilises the ranking; per-tree sub-samples (256) reduce masking. Fast, scales linearly, robust default.

**DBSCAN.** Points with ≥ `min_samples` neighbours within `eps` are core points; points not reachable from any core point are noise (label −1) = anomalies. Because DBSCAN has no `predict`, we score a flow by its distance to the nearest core point (anomaly ⇔ distance > eps), which reproduces the −1 labels and also ranks severity. `eps` is chosen from the k-distance quantile tied to `contamination`.

**Local Outlier Factor.** Compares a point's local reachability density with its k neighbours'; LOF ≈ 1 is normal, ≫ 1 is a local outlier. Run in novelty mode so unseen flows can be scored. Good at outliers relative to *local* density.

**Important caveat – attack bursts.** Density methods only flag a group if it is smaller than their neighbourhood (`k` / `min_samples`). A port scan or flood produces hundreds of near-identical flows, which looks like a legitimate dense cluster if `k` is too small (LOF ≈ 1). By default `k`/`min_samples` are therefore set to half of the expected anomaly count (`0.5 · contamination · n`, bounded to [35 or 10, 1000]). If you set `contamination` far below the true attack share, LOF and DBSCAN degrade sharply (the comparison tab shows this), while Isolation Forest still ranks attacks well (ROC-AUC stays high) but recalls fewer of them.

**Typical results** (synthetic, 20 000 flows, 5 % attacks, `contamination=0.05`):

| Model | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Isolation Forest | ~0.98 | ~0.98 | ~1.00 |
| Local Outlier Factor | ~0.84 | ~0.84 | ~1.00 |
| DBSCAN | ~1.00 | ~0.66 | ~0.96 |

Synthetic attacks are cleaner than real ones; expect lower numbers on CICIDS2017.

**Metrics.** Silhouette measures geometric separation of flagged vs. normal flows (no labels needed). Precision = alert quality, recall = attacks caught, F1 = balance, ROC-AUC = ranking quality independent of threshold.

**Threat level.** Based on the flagged share (<1 % LOW, <3 % GUARDED, <8 % ELEVATED, <15 % HIGH, else CRITICAL). For Isolation Forest and LOF the flagged share is set by `contamination`, so read it relative to that setting. **Severity** (0-100) is a log-scaled distance of the score above the cut-off.

## 6. Packaging the project as a `.zip`

Run from the folder that *contains* `network_anomaly_detection/`.

**macOS / Linux**
```bash
zip -r network_anomaly_detection.zip network_anomaly_detection \
    -x "*/__pycache__/*" "*/.venv/*" "*.pyc"
```

**Windows PowerShell**
```powershell
Compress-Archive -Path network_anomaly_detection -DestinationPath network_anomaly_detection.zip
```

**Windows cmd.exe (Windows 10+)**
```bat
tar -a -c -f network_anomaly_detection.zip network_anomaly_detection
```
(Delete `.venv` first or move it out; it is large and machine-specific.)

## 7. Limitations & ideas
* Synthetic data is only a demo; validate on CICIDS2017 before drawing conclusions.
* Flow-level features only - no payload or sequence modelling.
* Extend with additional detectors (e.g. PyOD's ECOD/COPOD, listed in `requirements.txt` for that purpose), autoencoders, or streaming scoring.
