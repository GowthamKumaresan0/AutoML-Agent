from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    """Generate matplotlib correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)

    ticks = np.arange(len(corr_matrix.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(corr_matrix.columns, fontsize=9)

    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.7 else "white", fontsize=8)

    plt.title("Correlation Matrix", pad=20, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feature_importances(importances: dict[str, float], top_n: int = 15) -> plt.Figure:
    """Generate feature importances horizontal bar chart."""
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [x[0] for x in reversed(sorted_features)]
    scores = [x[1] for x in reversed(sorted_features)]

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.3)))
    colors = plt.cm.viridis(np.linspace(0.4, 0.8, len(features)))
    ax.barh(features, scores, color=colors)
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    ax.set_title(f"Top {len(features)} Feature Importances", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_missing_values(df: pd.DataFrame) -> plt.Figure | None:
    """Generate missing values percentage bar chart."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return None

    missing_pct = (missing / len(df)) * 100
    missing_pct = missing_pct.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3, len(missing_pct) * 0.3)))
    ax.barh(missing_pct.index, missing_pct.values, color="tomato")
    ax.set_xlabel("Missing Percentage (%)", fontsize=10)
    ax.set_title("Missing Values Breakdown", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list[str]) -> plt.Figure:
    """Generate confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center", color="white" if val > cm.max() / 2 else "black")

    plt.title("Confusion Matrix", pad=20, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_residuals(y_true: list | np.ndarray, y_pred: list | np.ndarray) -> plt.Figure:
    """Generate residual plot for regression."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    residuals = y_t - y_p

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_p, residuals, alpha=0.6, color="teal", edgecolors="k")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Values", fontsize=10)
    ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=10)
    ax.set_title("Residual Analysis Plot", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig
