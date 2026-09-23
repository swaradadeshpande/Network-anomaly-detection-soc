"""
run.py - One-command pipeline: generate -> preprocess/train -> evaluate -> dashboard
====================================================================================

Examples
--------
    python run.py                               # everything, then open the dashboard
    python run.py --no-dashboard                # headless (CI, servers)
    python run.py --n-flows 50000 --contamination 0.03
    python run.py --data data/Friday-WorkingHours.csv --max-rows 50000   # real CICIDS2017 file
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))            # allow `import src...` when launched from any directory

from src.evaluator import compare_models, save_report_figures      # noqa: E402
from src.generate_data import generate_synthetic_traffic            # noqa: E402
from src.preprocessor import load_dataset                           # noqa: E402
from src.train import train_all_models                              # noqa: E402


def banner(text: str) -> None:
    """Print a visible step header so the console log is easy to follow."""
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def main() -> None:
    p = argparse.ArgumentParser(description="Network anomaly detection pipeline.")
    p.add_argument("--data", help="CSV to analyse (CICIDS2017, NSL-KDD, ...). Default: synthetic data.")
    p.add_argument("--n-flows", type=int, default=20_000, help="synthetic flows to generate")
    p.add_argument("--anomaly-ratio", type=float, default=0.05, help="true attack share in synthetic data")
    p.add_argument("--contamination", type=float, default=0.05, help="expected anomaly share for the models")
    p.add_argument("--max-rows", type=int, default=50_000,
                   help="random sub-sample for large real datasets (DBSCAN/LOF memory grows with rows)")
    p.add_argument("--no-dashboard", action="store_true", help="skip launching Streamlit")
    p.add_argument("--port", type=int, default=8501)
    a = p.parse_args()

    # ---- Step 1: data ------------------------------------------------------ #
    banner("Step 1/4 - Data")
    if a.data:
        df = load_dataset(a.data)
        print(f"Loaded {len(df):,} rows from {a.data}")
        if len(df) > a.max_rows:
            df = df.sample(a.max_rows, random_state=42).reset_index(drop=True)
            print(f"Sub-sampled to {a.max_rows:,} rows (use --max-rows to change)")
    else:
        df = generate_synthetic_traffic(a.n_flows, a.anomaly_ratio)
        out = ROOT / "data" / "synthetic_traffic.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)                    # saved so it can be uploaded in the dashboard
        print(f"Generated {len(df):,} synthetic flows -> {out}")

    # ---- Step 2: preprocess + train --------------------------------------- #
    banner("Step 2/4 - Preprocessing and training (Isolation Forest, DBSCAN, LOF)")
    pre, X, results, y = train_all_models(df, a.contamination, ROOT / "models")
    print(f"Features used ({len(pre.feature_names_)}): {', '.join(pre.feature_names_)}")
    print(f"Log-transformed: {', '.join(pre.log_cols_) or 'none'}")
    print(f"Artifacts saved to {ROOT / 'models'}")

    # ---- Step 3: evaluate -------------------------------------------------- #
    banner("Step 3/4 - Evaluation")
    table = compare_models(results, X, y)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table)
    if y is None:
        print("(No label column: only unsupervised metrics are available.)")
    table.to_csv(ROOT / "models" / "metrics.csv")
    (ROOT / "models" / "metrics.json").write_text(json.dumps(table.reset_index().to_dict("records"), indent=2))
    for path in save_report_figures(results, y, ROOT / "models" / "reports"):
        print(f"Saved figure: {path}")

    # ---- Step 4: dashboard ------------------------------------------------- #
    if a.no_dashboard:
        banner("Done (dashboard skipped). Launch later with:  streamlit run app.py")
        return
    banner("Step 4/4 - Launching the SOC dashboard (Ctrl+C to stop)")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                    "--server.port", str(a.port)], check=False)


if __name__ == "__main__":
    main()
