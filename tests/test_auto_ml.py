from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from agents.cleaner import CleanerAgent
from agents.eda import EDAAgent
from agents.explain import ExplainabilityAgent
from agents.planner import PlannerAgent
from agents.report import ReportAgent
from agents.trainer import TrainerAgent


@pytest.fixture
def classification_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": [25, 30, 45, 35, 50, 23, 40, 60, 29, 38],
        "income": [50000.0, 70000.0, np.nan, 80000.0, 110000.0, 45000.0, 95000.0, 120000.0, 62000.0, np.nan],
        "city": ["NY", "SF", "NY", "LA", "SF", "LA", "NY", "SF", "LA", "NY"],
        "target": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    })


@pytest.fixture
def regression_df():
    np.random.seed(42)
    return pd.DataFrame({
        "sqft": [1000, 1500, 2000, 2500, 3000, 1200, 1800, 2200, 2800, 3200],
        "bedrooms": [2, 3, 3, 4, 4, 2, 3, 3, 4, 5],
        "price": [200000, 300000, 400000, 500000, 600000, 240000, 360000, 440000, 560000, 640000],
    })


def test_planner_agent(classification_df, regression_df):
    # Test classification plan
    planner = PlannerAgent(classification_df)
    info = planner.analyze()
    assert info["problem_type"] == "Classification"
    assert info["target"] == "target"
    assert "numeric" in info["feature_counts"]

    # Test regression plan
    reg_planner = PlannerAgent(regression_df)
    reg_info = reg_planner.analyze(target_column="price")
    assert reg_info["problem_type"] == "Regression"
    assert reg_info["target"] == "price"


def test_cleaner_agent(classification_df):
    cleaner = CleanerAgent(classification_df, target_col="target", numeric_impute_strategy="median")
    cleaned_df = cleaner.clean()

    assert cleaned_df["income"].isnull().sum() == 0
    assert len(cleaned_df) == len(classification_df)
    summary = cleaner.get_summary()
    assert len(summary["cleaning_log"]) > 0


def test_eda_agent(classification_df):
    eda = EDAAgent(classification_df, target_col="target")
    analysis = eda.analyze()

    assert "numeric_stats" in analysis
    assert "categorical_stats" in analysis
    assert len(analysis["health_alerts"]) >= 1  # income missing value alert


def test_trainer_agent_classification(classification_df, tmp_path):
    cleaner = CleanerAgent(classification_df, target_col="target")
    cleaned = cleaner.clean()

    trainer = TrainerAgent(cleaned, target_col="target", problem_type="Classification", output_dir=str(tmp_path))
    result = trainer.run()

    assert result["problem_type"] == "Classification"
    assert "accuracy" in result["metrics"]
    assert len(result["leaderboard"]) > 0
    assert result["model_path"].exists()

    # Test loading model and inference
    sample_input = {"age": 30, "income": 60000.0, "city": "NY"}
    preds = TrainerAgent.load_and_predict(result["model_path"], sample_input)
    assert len(preds) == 1


def test_trainer_agent_regression(regression_df, tmp_path):
    trainer = TrainerAgent(regression_df, target_col="price", problem_type="Regression", output_dir=str(tmp_path))
    result = trainer.run()

    assert result["problem_type"] == "Regression"
    assert "rmse" in result["metrics"]
    assert result["model_path"].exists()


def test_explainability_agent(classification_df, tmp_path):
    cleaner = CleanerAgent(classification_df, target_col="target")
    cleaned = cleaner.clean()

    trainer = TrainerAgent(cleaned, target_col="target", output_dir=str(tmp_path))
    result = trainer.run()

    explainer = ExplainabilityAgent(result["model_bundle"])
    importances = explainer.get_feature_importances()
    drivers = explainer.get_top_drivers_summary()

    assert len(importances) > 0
    assert len(drivers) > 0

    instance_exp = explainer.explain_instance({"age": 28, "income": 55000.0, "city": "LA"})
    assert "prediction" in instance_exp
    assert "feature_contributions" in instance_exp


def test_report_agent(classification_df, tmp_path):
    planner = PlannerAgent(classification_df)
    plan = planner.analyze()

    eda = EDAAgent(classification_df, target_col="target").analyze()

    cleaner = CleanerAgent(classification_df, target_col="target")
    cleaned = cleaner.clean()
    cleaning_summary = cleaner.get_summary()

    trainer = TrainerAgent(cleaned, target_col="target", output_dir=str(tmp_path))
    train_res = trainer.run()

    explainer = ExplainabilityAgent(train_res["model_bundle"])
    explain_info = {
        "importances": explainer.get_feature_importances(),
        "drivers_summary": explainer.get_top_drivers_summary(),
        "explainer": explainer,
    }

    reporter = ReportAgent(plan, eda, cleaning_summary, train_res, explain_info)
    md_file, html_file = reporter.save_reports(str(tmp_path))

    assert md_file.exists()
    assert html_file.exists()
    assert "# 🤖 AutoML Agent Executive Report" in md_file.read_text(encoding="utf-8")
