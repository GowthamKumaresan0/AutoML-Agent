"""
export_dashboard_data.py
========================
Reads your real AutoML Agent outputs (report + model bundle)
and exports them as dashboard_data.json so the HTML dashboard
can display YOUR actual model metrics — not fake ones.

Run this from your AutoML-Agent directory:
    python ai-dashboard/export_dashboard_data.py

It will produce: ai-dashboard/dashboard_data.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent          # e:\AutoML-Agent
REPORT_MD  = BASE_DIR / "reports" / "automl_report.md"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
OUT_JSON   = Path(__file__).parent / "dashboard_data.json"

# ── 1. Parse the Markdown report ───────────────────────────────────────────
def parse_report(md_path: Path) -> dict:
    if not md_path.exists():
        print(f"[WARN] Report not found at {md_path}. Using placeholder data.")
        return {}

    text = md_path.read_text(encoding="utf-8")
    data: dict = {}

    # Problem type
    m = re.search(r"\*\*Problem Type\*\*: (.+?)(?:\n|$)", text)
    data["problem_type"] = m.group(1).strip() if m else "Unknown"

    # Target variable
    m = re.search(r"\*\*Target Variable\*\*: `(.+?)`", text)
    data["target"] = m.group(1).strip() if m else "Unknown"

    # Total rows
    m = re.search(r"\*\*Total Rows Processed\*\*: (\d+)", text)
    data["total_rows"] = int(m.group(1)) if m else 0

    # Total features
    m = re.search(r"\*\*Total Features\*\*: (\d+)", text)
    data["total_features"] = int(m.group(1)) if m else 0

    # Best model
    m = re.search(r"\*\*Best Model\*\*: \*\*(.+?)\*\*", text)
    data["best_model"] = m.group(1).strip() if m else "Unknown"

    # Primary metrics from the summary line
    m = re.search(r"\*\*Primary Performance\*\*: (.+?)(?:\n|$)", text)
    metrics_raw = {}
    if m:
        for pair in re.findall(r"\*\*(\w+)\*\*: ([0-9.]+)", m.group(1)):
            metrics_raw[pair[0].lower()] = float(pair[1])
    data["best_metrics"] = metrics_raw

    # Leaderboard: parse table rows
    leaderboard = []
    lb_section = re.search(r"## 4\. AutoML Model Leaderboard(.*?)## 5\.", text, re.DOTALL)
    if lb_section:
        for row in re.finditer(
            r"\|\s*\d+\s*\|\s*\*\*(.+?)\*\*.*?\|\s*`(.+?)`\s*\|", lb_section.group(1)
        ):
            name_raw = row.group(1).strip()
            # Strip the trophy/label
            name = re.sub(r"\s*🏆.*$", "", name_raw).strip()
            metrics_str = row.group(2)
            metrics = {}
            for pair in re.findall(r"([\w_]+):\s*([0-9.]+)", metrics_str):
                metrics[pair[0]] = float(pair[1])
            leaderboard.append({"model": name, "metrics": metrics})
    data["leaderboard"] = leaderboard

    # Feature importances from section 5
    drivers = []
    fi_section = re.search(r"## 5\. Model Interpretability.*?(## 6\.|\Z)", text, re.DOTALL)
    if fi_section:
        for d in re.finditer(
            r"Rank (\d+): \*\*(.+?)\*\* contributes \*\*([0-9.]+)%\*\*", fi_section.group(0)
        ):
            drivers.append({
                "rank": int(d.group(1)),
                "feature": d.group(2),
                "importance": float(d.group(3))
            })
    data["feature_drivers"] = drivers

    # Cleaning log
    cleaning = []
    cl_section = re.search(r"## 3\. Data Cleaning Log(.*?)## 4\.", text, re.DOTALL)
    if cl_section:
        for item in re.findall(r"- (.+)", cl_section.group(1)):
            cleaning.append(item.strip())
    data["cleaning_log"] = cleaning

    return data


# ── 2. Pull extras from the joblib model bundle ────────────────────────────
def extract_model_bundle(model_path: Path) -> dict:
    try:
        import joblib
    except ImportError:
        print("[WARN] joblib not installed — skipping model bundle extraction.")
        return {}

    if not model_path.exists():
        print(f"[WARN] model.joblib not found at {model_path}.")
        return {}

    try:
        bundle = joblib.load(model_path)
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        return {}

    result = {}

    # Confusion matrix
    result["confusion_matrix"] = bundle.get("confusion_matrix")
    result["class_labels"]     = bundle.get("class_labels")
    result["feature_columns"]  = bundle.get("feature_columns", [])

    # Test predictions vs actuals (max 200 for chart)
    preds   = bundle.get("test_predictions", [])
    actuals = bundle.get("test_actuals", [])
    if hasattr(preds, "tolist"):   preds   = preds.tolist()
    if hasattr(actuals, "tolist"): actuals = actuals.tolist()
    result["test_predictions"] = [str(p) for p in preds[:200]]
    result["test_actuals"]     = [str(a) for a in actuals[:200]]

    # Full leaderboard metrics (more precise than MD)
    raw_lb = bundle.get("leaderboard", [])
    lb_out = []
    for entry in raw_lb:
        m = entry.get("metrics", {})
        lb_out.append({
            "model": entry.get("model_name", ""),
            "metrics": {k: round(float(v), 4) for k, v in m.items()},
            "primary_score": round(float(entry.get("primary_score", 0)), 4),
        })
    if lb_out:
        result["leaderboard_full"] = lb_out

    return result


# ── 3. Merge and export ────────────────────────────────────────────────────
def main():
    print("[*] Exporting real AutoML Agent data to dashboard_data.json ...")

    report_data = parse_report(REPORT_MD)
    bundle_data = extract_model_bundle(MODEL_PATH)

    # Merge: bundle leaderboard wins (more precise)
    dashboard_data = {**report_data, **bundle_data}

    # If bundle has a more precise leaderboard, use it
    if "leaderboard_full" in dashboard_data:
        dashboard_data["leaderboard"] = dashboard_data.pop("leaderboard_full")

    # Metadata
    from datetime import datetime
    dashboard_data["generated_at"] = datetime.now().isoformat()
    dashboard_data["source"] = str(REPORT_MD)

    OUT_JSON.write_text(json.dumps(dashboard_data, indent=2), encoding="utf-8")
    print(f"[OK] Exported to: {OUT_JSON}")
    print(f"   Best Model : {dashboard_data.get('best_model', 'N/A')}")
    print(f"   Leaderboard: {len(dashboard_data.get('leaderboard', []))} models")
    print(f"   Features   : {dashboard_data.get('total_features', 'N/A')}")


if __name__ == "__main__":
    main()
