from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.metrics import compute_classification_metrics, compute_regression_metrics


class TrainerAgent:
    """Automated Model Training Agent using Scikit-Learn Pipelines."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_col: str,
        problem_type: str | None = None,
        output_dir: str = "models",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.df = dataframe.copy()
        self.target_col = target_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.random_state = random_state

        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe.")

        # Determine problem type if not specified
        if problem_type is None:
            target_series = self.df[self.target_col].dropna()
            if target_series.dtype == "object" or target_series.nunique() <= 10:
                self.problem_type = "Classification"
            else:
                self.problem_type = "Regression"
        else:
            self.problem_type = problem_type

    def _build_preprocessor(self, X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
        """Construct scikit-learn preprocessing ColumnTransformer."""
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

        return preprocessor, numeric_features, categorical_features

    def _get_candidate_models(self) -> dict[str, object]:
        """Return dict of candidate model algorithms based on problem type."""
        if self.problem_type == "Classification":
            return {
                "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=self.random_state),
            }
        else:
            return {
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
                "Ridge Regression": Ridge(random_state=self.random_state),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=self.random_state),
            }

    def run(self) -> dict:
        """Run model candidate evaluation, select best model, and persist artifact."""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Stratify split if classification and sufficient class representation
        stratify = None
        if self.problem_type == "Classification" and y.nunique() >= 2:
            class_counts = y.value_counts()
            if (class_counts >= 2).all():
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )

        preprocessor, num_cols, cat_cols = self._build_preprocessor(X)
        candidate_models = self._get_candidate_models()

        leaderboard: list[dict] = []
        trained_pipelines: dict[str, Pipeline] = {}
        predictions_map: dict[str, list] = {}

        for model_name, estimator in candidate_models.items():
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            try:
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                predictions_map[model_name] = preds

                if self.problem_type == "Classification":
                    metrics = compute_classification_metrics(y_test, preds)
                    primary_score = metrics["accuracy"]
                else:
                    metrics = compute_regression_metrics(y_test, preds)
                    primary_score = -metrics["rmse"]  # lower RMSE is better

                leaderboard.append({
                    "model_name": model_name,
                    "primary_score": primary_score,
                    "metrics": metrics,
                })
                trained_pipelines[model_name] = pipeline
            except Exception as e:
                continue

        if not leaderboard:
            raise RuntimeError("All candidate models failed to train.")

        # Sort leaderboard
        leaderboard.sort(key=lambda item: item["primary_score"], reverse=True)
        best_entry = leaderboard[0]
        best_model_name = best_entry["model_name"]
        best_pipeline = trained_pipelines[best_model_name]
        best_preds = predictions_map[best_model_name]

        # Compute additional artifact diagnostics
        cm = None
        labels = None
        if self.problem_type == "Classification":
            labels = [str(c) for c in np.unique(y_test)]
            cm = confusion_matrix([str(val) for val in y_test], [str(val) for val in best_preds], labels=labels).tolist()

        # Save model bundle artifact
        model_bundle = {
            "pipeline": best_pipeline,
            "model_name": best_model_name,
            "problem_type": self.problem_type,
            "target_col": self.target_col,
            "feature_columns": X.columns.tolist(),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "metrics": best_entry["metrics"],
            "leaderboard": leaderboard,
            "test_predictions": [float(p) if isinstance(p, (np.number, float, int)) else str(p) for p in best_preds],
            "test_actuals": [float(a) if isinstance(a, (np.number, float, int)) else str(a) for a in y_test],
            "confusion_matrix": cm,
            "class_labels": labels,
        }

        model_path = self.output_dir / "model.joblib"
        joblib.dump(model_bundle, model_path)

        return {
            "problem_type": self.problem_type,
            "best_model": best_model_name,
            "metrics": best_entry["metrics"],
            "leaderboard": leaderboard,
            "model_path": model_path,
            "feature_columns": X.columns.tolist(),
            "test_predictions": best_preds,
            "test_actuals": y_test,
            "confusion_matrix": cm,
            "class_labels": labels,
            "model_bundle": model_bundle,
        }

    @staticmethod
    def load_and_predict(model_path: str | Path, input_data: pd.DataFrame | dict) -> list | np.ndarray:
        """Load persisted model bundle and make predictions on new data."""
        bundle = joblib.load(model_path)
        pipeline: Pipeline = bundle["pipeline"]

        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        # Ensure all required features exist
        for col in bundle["feature_columns"]:
            if col not in input_df.columns:
                input_df[col] = np.nan

        input_df = input_df[bundle["feature_columns"]]
        return pipeline.predict(input_df)
