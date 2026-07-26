from __future__ import annotations

import numpy as np
import pandas as pd


class EDAAgent:
    """Exploratory Data Analysis Agent for profiling datasets and detecting health issues."""

    def __init__(self, dataframe: pd.DataFrame, target_col: str | None = None):
        self.df = dataframe.copy()
        self.target_col = target_col

    def get_summary_statistics(self) -> dict[str, pd.DataFrame]:
        """Compute detailed summary statistics for numeric and categorical columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        categorical_df = self.df.select_dtypes(exclude=[np.number])

        num_summary = pd.DataFrame()
        if not numeric_df.empty:
            num_summary = numeric_df.describe().T
            num_summary["skewness"] = numeric_df.skew()
            num_summary["missing_count"] = numeric_df.isnull().sum()
            num_summary["missing_pct"] = (numeric_df.isnull().mean() * 100).round(2)

        cat_summary = pd.DataFrame()
        if not categorical_df.empty:
            cat_summary = pd.DataFrame(index=categorical_df.columns)
            cat_summary["unique_count"] = categorical_df.nunique()
            cat_summary["top_category"] = categorical_df.apply(lambda col: col.mode().iloc[0] if not col.mode().empty else None)
            cat_summary["top_freq"] = categorical_df.apply(lambda col: col.value_counts().max() if not col.empty else 0)
            cat_summary["missing_count"] = categorical_df.isnull().sum()
            cat_summary["missing_pct"] = (categorical_df.isnull().mean() * 100).round(2)

        return {
            "numeric": num_summary,
            "categorical": cat_summary,
        }

    def get_missing_summary(self) -> pd.DataFrame:
        """Return a DataFrame detailing missing values per column."""
        total = self.df.isnull().sum()
        pct = (self.df.isnull().mean() * 100).round(2)
        summary = pd.DataFrame({"Missing Count": total, "Missing Percentage (%)": pct, "Data Type": self.df.dtypes})
        return summary.sort_values(by="Missing Count", ascending=False)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Compute Pearson correlation matrix for numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return pd.DataFrame()
        return numeric_df.corr().round(3)

    def get_data_health_alerts(self) -> list[dict[str, str]]:
        """Identify potential data quality issues and return formatted health alerts."""
        alerts: list[dict[str, str]] = []

        # 1. Duplicate rows check
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            dup_pct = (dup_count / len(self.df)) * 100
            alerts.append({
                "severity": "MEDIUM",
                "category": "Duplicates",
                "message": f"Found {dup_count} duplicate row(s) ({dup_pct:.1f}% of total dataset).",
            })

        # 2. Missing values check
        for col in self.df.columns:
            missing_pct = (self.df[col].isnull().mean()) * 100
            if missing_pct > 30.0:
                alerts.append({
                    "severity": "HIGH" if missing_pct > 50 else "MEDIUM",
                    "category": "Missing Data",
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values.",
                })

        # 3. Constant columns check
        for col in self.df.columns:
            if self.df[col].nunique(dropna=True) <= 1:
                alerts.append({
                    "severity": "HIGH",
                    "category": "Constant Column",
                    "message": f"Column '{col}' contains only 1 unique value.",
                })

        # 4. High cardinality categorical check
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            unique_cnt = self.df[col].nunique()
            if unique_cnt > 50 and col != self.target_col:
                alerts.append({
                    "severity": "MEDIUM",
                    "category": "High Cardinality",
                    "message": f"Categorical column '{col}' has {unique_cnt} unique values.",
                })

        # 5. Collinearity check
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr().abs()
            high_corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if val > 0.85:
                        col1, col2 = corr.columns[i], corr.columns[j]
                        if col1 != self.target_col and col2 != self.target_col:
                            high_corr_pairs.append((col1, col2, val))

            for col1, col2, val in high_corr_pairs[:3]:
                alerts.append({
                    "severity": "LOW",
                    "category": "High Collinearity",
                    "message": f"Strong correlation between '{col1}' and '{col2}' (r = {val:.2f}).",
                })

        # 6. Class imbalance check
        if self.target_col and self.target_col in self.df.columns:
            target_series = self.df[self.target_col].dropna()
            if target_series.nunique() >= 2 and target_series.nunique() <= 10:
                counts = target_series.value_counts(normalize=True)
                min_class_pct = counts.min() * 100
                if min_class_pct < 15.0:
                    alerts.append({
                        "severity": "MEDIUM",
                        "category": "Class Imbalance",
                        "message": f"Target column '{self.target_col}' is imbalanced. Smallest class is {min_class_pct:.1f}% of data.",
                    })

        return alerts

    def analyze(self) -> dict:
        """Run full EDA pipeline and return summary dict."""
        stats = self.get_summary_statistics()
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "missing_summary": self.get_missing_summary(),
            "numeric_stats": stats["numeric"],
            "categorical_stats": stats["categorical"],
            "correlation_matrix": self.get_correlation_matrix(),
            "health_alerts": self.get_data_health_alerts(),
        }
