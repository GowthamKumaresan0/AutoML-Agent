from __future__ import annotations

import pandas as pd


class PlannerAgent:
    """Analyze dataset, infer schema, profile features, detect problem type, and suggest metrics."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def profile_features(self) -> dict[str, list[str]]:
        """Categorize features into numeric, categorical, boolean, and datetime types."""
        feature_types: dict[str, list[str]] = {
            "numeric": [],
            "categorical": [],
            "boolean": [],
            "datetime": [],
        }

        for col in self.df.columns:
            dtype = self.df[col].dtype
            if pd.api.types.is_bool_dtype(dtype):
                feature_types["boolean"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                feature_types["datetime"].append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                # Check if it might be a binary integer column (0/1)
                if self.df[col].dropna().isin([0, 1]).all() and self.df[col].nunique() <= 2:
                    feature_types["boolean"].append(col)
                else:
                    feature_types["numeric"].append(col)
            else:
                feature_types["categorical"].append(col)

        return feature_types

    def guess_target(self) -> str:
        """Guess the target column based on column names or default to the last column."""
        priority = [
            "target",
            "label",
            "class",
            "output",
            "y",
            "price",
            "salary",
            "churn",
            "survived",
            "diagnosis",
            "species",
            "quality",
            "approved",
            "status",
            "is_fraud",
            "outcome",
        ]

        columns_lower = {c.lower(): c for c in self.df.columns}

        for col in priority:
            if col in columns_lower:
                return columns_lower[col]

        return str(self.df.columns[-1])

    def detect_problem(self, target_column: str) -> tuple[str, str, str]:
        """
        Determine problem type (Classification vs Regression), sub-type, and recommended metric.
        Returns (problem_type, sub_type, recommended_metric)
        """
        if target_column not in self.df.columns:
            target_column = self.guess_target()

        target_series = self.df[target_column].dropna()
        unique_count = target_series.nunique()
        dtype = target_series.dtype

        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            if unique_count == 2:
                return "Classification", "Binary Classification", "Accuracy"
            return "Classification", "Multiclass Classification", "F1 Score"

        # Numeric column
        if unique_count <= 2:
            return "Classification", "Binary Classification", "Accuracy"
        elif unique_count <= 5 and (target_series % 1 == 0).all():
            return "Classification", "Multiclass Classification", "F1 Score"
        else:
            return "Regression", "Continuous Regression", "RMSE"

    def analyze(self, target_column: str | None = None) -> dict:
        """Produce a complete problem plan and dataset metadata dict."""
        target = target_column if (target_column and target_column in self.df.columns) else self.guess_target()
        problem_type, sub_type, metric = self.detect_problem(target)
        features = self.profile_features()

        feature_cols = [c for c in self.df.columns if c != target]

        if problem_type == "Classification":
            recommended_models = [
                "Random Forest Classifier",
                "Gradient Boosting Classifier",
                "Logistic Regression",
                "Decision Tree Classifier",
            ]
        else:
            recommended_models = [
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
                "Ridge Regression",
                "Decision Tree Regressor",
            ]

        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "target": target,
            "problem_type": problem_type,
            "sub_type": sub_type,
            "metric": metric,
            "features": feature_cols,
            "feature_counts": {
                "numeric": len([c for c in features["numeric"] if c != target]),
                "categorical": len([c for c in features["categorical"] if c != target]),
                "boolean": len([c for c in features["boolean"] if c != target]),
                "datetime": len([c for c in features["datetime"] if c != target]),
            },
            "recommended_models": recommended_models,
        }