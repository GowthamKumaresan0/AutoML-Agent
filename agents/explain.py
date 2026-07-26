from __future__ import annotations

import numpy as np
import pandas as pd


class ExplainabilityAgent:
    """Explainability Agent to extract feature importances and interpret model predictions."""

    def __init__(self, model_bundle: dict):
        self.bundle = model_bundle
        self.pipeline = model_bundle["pipeline"]
        self.feature_columns = model_bundle["feature_columns"]
        self.numeric_columns = model_bundle["numeric_columns"]
        self.categorical_columns = model_bundle["categorical_columns"]
        self.model_name = model_bundle["model_name"]
        self.problem_type = model_bundle["problem_type"]

    def _get_feature_names(self) -> list[str]:
        """Extract feature names after OneHotEncoder transformation in ColumnTransformer."""
        preprocessor = self.pipeline.named_steps["preprocessor"]
        feature_names: list[str] = list(self.numeric_columns)

        if self.categorical_columns:
            try:
                cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
                encoded_cats = cat_encoder.get_feature_names_out(self.categorical_columns).tolist()
                feature_names.extend(encoded_cats)
            except Exception:
                feature_names.extend(self.categorical_columns)

        return feature_names

    def get_feature_importances(self) -> dict[str, float]:
        """Extract feature importances from trained pipeline estimator and aggregate by raw column."""
        estimator = self.pipeline.named_steps["model"]
        transformed_feature_names = self._get_feature_names()

        raw_importances = None

        if hasattr(estimator, "feature_importances_"):
            raw_importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            raw_importances = coef

        if raw_importances is None or len(raw_importances) != len(transformed_feature_names):
            # Fallback: uniform distribution if importances not directly exposed
            n = len(self.feature_columns)
            return {col: round(1.0 / n, 4) for col in self.feature_columns}

        # Map transformed feature importances back to raw input columns
        aggregated: dict[str, float] = {col: 0.0 for col in self.feature_columns}

        for fname, imp in zip(transformed_feature_names, raw_importances):
            found_raw = None
            for raw_col in self.feature_columns:
                if fname == raw_col or fname.startswith(f"{raw_col}_"):
                    found_raw = raw_col
                    break
            if found_raw:
                aggregated[found_raw] += float(imp)
            else:
                # Assign to closest matching column
                matched = False
                for col in self.feature_columns:
                    if col in fname:
                        aggregated[col] += float(imp)
                        matched = True
                        break
                if not matched and self.feature_columns:
                    aggregated[self.feature_columns[0]] += float(imp)

        total_imp = sum(aggregated.values())
        if total_imp > 0:
            aggregated = {k: round(v / total_imp, 4) for k, v in aggregated.items()}

        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))

    def get_top_drivers_summary(self, top_k: int = 5) -> list[str]:
        """Generate human-readable insights on the top feature drivers."""
        importances = self.get_feature_importances()
        top_items = list(importances.items())[:top_k]

        insights: list[str] = []
        for rank, (feature, score) in enumerate(top_items, 1):
            pct = score * 100
            insights.append(
                f"Rank {rank}: **{feature}** contributes **{pct:.1f}%** to the model's overall prediction decisions."
            )

        return insights

    def explain_instance(self, instance_dict: dict) -> dict:
        """Provide a per-feature contribution breakdown for a single input row prediction."""
        importances = self.get_feature_importances()
        input_df = pd.DataFrame([instance_dict])

        # Predict
        predicted_value = self.pipeline.predict(input_df)[0]

        contributions = {}
        for feature, imp in importances.items():
            val = instance_dict.get(feature, "N/A")
            contributions[feature] = {
                "input_value": val,
                "importance_weight": imp,
                "description": f"Feature '{feature}' (value: {val}) has weight {imp:.4f}",
            }

        return {
            "prediction": float(predicted_value) if isinstance(predicted_value, (np.number, float, int)) else str(predicted_value),
            "feature_contributions": contributions,
            "top_influencer": list(importances.keys())[0] if importances else "N/A",
        }
