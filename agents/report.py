from __future__ import annotations

from pathlib import Path
import json


class ReportAgent:
    """Generate comprehensive Markdown and HTML AutoML Reports."""

    def __init__(
        self,
        plan_info: dict,
        eda_info: dict,
        cleaning_summary: dict,
        training_info: dict,
        explain_info: dict,
    ):
        self.plan = plan_info
        self.eda = eda_info
        self.cleaning = cleaning_summary
        self.training = training_info
        self.explain = explain_info

    def generate_markdown_report(self) -> str:
        """Construct full Markdown report."""
        lines = []

        lines.append("# 🤖 AutoML Agent Executive Report")
        lines.append("")
        lines.append("## 1. Executive Summary")
        lines.append(f"- **Problem Type**: {self.plan.get('problem_type', 'N/A')} ({self.plan.get('sub_type', 'N/A')})")
        lines.append(f"- **Target Variable**: `{self.plan.get('target', 'N/A')}`")
        lines.append(f"- **Total Rows Processed**: {self.plan.get('rows', 'N/A')}")
        lines.append(f"- **Total Features**: {self.plan.get('columns', 1) - 1}")
        lines.append(f"- **Best Model**: **{self.training.get('best_model', 'N/A')}**")
        
        metrics = self.training.get("metrics", {})
        metrics_str = ", ".join([f"**{k.upper()}**: {v}" for k, v in metrics.items()])
        lines.append(f"- **Primary Performance**: {metrics_str}")
        lines.append("")

        lines.append("## 2. Dataset Overview & Data Quality Audit")
        lines.append(f"- **Total Samples**: {self.eda.get('rows', 'N/A')}")
        lines.append(f"- **Total Columns**: {self.eda.get('columns', 'N/A')}")
        
        health_alerts = self.eda.get("health_alerts", [])
        lines.append("### Data Health Alerts")
        if health_alerts:
            for alert in health_alerts:
                sev = alert.get("severity", "INFO")
                msg = alert.get("message", "")
                cat = alert.get("category", "")
                lines.append(f"- `[{sev}]` **{cat}**: {msg}")
        else:
            lines.append("- ✅ No major data health issues detected.")
        lines.append("")

        lines.append("## 3. Data Cleaning Log")
        cleaning_logs = self.cleaning.get("cleaning_log", [])
        if cleaning_logs:
            for log_item in cleaning_logs:
                lines.append(f"- {log_item}")
        else:
            lines.append("- Dataset was cleaned automatically with default imputation strategies.")
        lines.append("")

        lines.append("## 4. AutoML Model Leaderboard")
        leaderboard = self.training.get("leaderboard", [])
        lines.append("| Rank | Model Name | Primary Metric | Details |")
        lines.append("|---|---|---|---|")
        for rank, item in enumerate(leaderboard, 1):
            name = item.get("model_name", "")
            is_best = " 🏆 (Best)" if rank == 1 else ""
            m = item.get("metrics", {})
            m_formatted = ", ".join([f"{k}: {v}" for k, v in m.items()])
            lines.append(f"| {rank} | **{name}**{is_best} | `{m_formatted}` | Evaluated on holdout set |")
        lines.append("")

        lines.append("## 5. Model Interpretability & Key Feature Drivers")
        drivers = self.explain.get("drivers_summary", [])
        lines.append("### Top Predictive Drivers")
        if drivers:
            for d in drivers:
                lines.append(f"- {d}")
        else:
            lines.append("- Feature importance values extracted from top estimator.")
        lines.append("")

        lines.append("## 6. Model Artifacts & Deployment")
        lines.append(f"- **Model Path**: `{self.training.get('model_path', 'models/model.joblib')}`")
        lines.append("- **Inference**: Load with `joblib.load('models/model.joblib')` and call `.predict(df)`.")
        lines.append("")

        return "\n".join(lines)

    def generate_html_report(self) -> str:
        """Wrap Markdown report in a beautifully styled standalone HTML page."""
        md_text = self.generate_markdown_report()

        # Convert simple markdown headers/lists to basic HTML
        html_body = []
        for line in md_text.splitlines():
            if line.startswith("# "):
                html_body.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_body.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_body.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("- "):
                html_body.append(f"<li>{line[2:]}</li>")
            elif line.startswith("|"):
                html_body.append(f"<code>{line}</code><br/>")
            elif line.strip() == "":
                html_body.append("<br/>")
            else:
                html_body.append(f"<p>{line}</p>")

        content = "\n".join(html_body)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AutoML Agent Executive Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: #1e293b;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            border: 1px solid #334155;
        }}
        h1 {{
            color: #38bdf8;
            border-bottom: 2px solid #38bdf8;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #818cf8;
            margin-top: 30px;
        }}
        h3 {{
            color: #94a3b8;
        }}
        li {{
            margin-bottom: 8px;
        }}
        code {{
            background-color: #0f172a;
            padding: 2px 6px;
            border-radius: 4px;
            color: #f43f5e;
            font-family: monospace;
        }}
        .footer {{
            margin-top: 50px;
            text-align: center;
            font-size: 0.85em;
            color: #64748b;
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
        <div class="footer">
            Generated by 🤖 AutoML Agent Dashboard
        </div>
    </div>
</body>
</html>"""

    def save_reports(self, output_dir: str = "reports") -> tuple[Path, Path]:
        """Save report files to specified directory."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        md_file = out_path / "automl_report.md"
        html_file = out_path / "automl_report.html"

        md_file.write_text(self.generate_markdown_report(), encoding="utf-8")
        html_file.write_text(self.generate_html_report(), encoding="utf-8")

        return md_file, html_file
