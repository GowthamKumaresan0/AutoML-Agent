# 🤖 AutoML Agent

> An end-to-end automated machine learning dashboard built with Streamlit and scikit-learn. Upload any CSV dataset, and AutoML Agent will automatically plan, clean, benchmark models, explain predictions, and export a professional report — all within a single interactive UI.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Smart Problem Detection** | Automatically identifies Classification vs. Regression from target column statistics |
| **Exploratory Data Analysis** | Descriptive statistics, health alerts, missing-value breakdown, and correlation heatmap |
| **Configurable Data Cleaning** | Numeric & categorical imputation strategies, duplicate removal, constant-column dropping |
| **Automated Model Benchmarking** | Trains and compares 4 scikit-learn models in a ranked leaderboard |
| **Model Explainability** | Feature importance charts and per-instance prediction drivers |
| **Live Prediction Sandbox** | Enter custom feature values and get real-time predictions from the trained model |
| **Report Export** | One-click download of the model, cleaned CSV, and an executive report (HTML + Markdown) |
| **Built-in Sample Datasets** | Run instantly without uploading data using synthetic Classification or Regression datasets |

---

## 🖼️ App Overview

The dashboard is organized into **5 tabs**:

```
📊 Dataset & EDA  →  🧹 Data Cleaning  →  🚀 Model Leaderboard  →  🔍 Explainability & Sandbox  →  📄 Report & Export
```

---

## 🗂️ Project Structure

```
AutoML-Agent/
│
├── app.py                    # Main Streamlit application
│
├── agents/                   # Core AI Agent modules
│   ├── planner.py            # PlannerAgent  – schema inference, problem type detection
│   ├── cleaner.py            # CleanerAgent  – configurable data cleaning & imputation
│   ├── eda.py                # EDAAgent      – statistical profiling & health alerts
│   ├── trainer.py            # TrainerAgent  – multi-model training & leaderboard ranking
│   ├── explain.py            # ExplainabilityAgent – feature importances & instance explanations
│   └── report.py             # ReportAgent   – Markdown & HTML executive report generation
│
├── utils/
│   ├── metrics.py            # Classification & regression metric helpers
│   └── visualization.py     # Matplotlib/Seaborn chart builders
│
├── models/                   # Persisted model artifacts (.joblib)
├── reports/                  # Auto-generated HTML & Markdown reports
├── uploads/                  # User-uploaded datasets (runtime)
├── data/                     # Static reference data
├── tests/
│   └── test_auto_ml.py       # Pytest unit test suite
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Agent Architecture

```
                        ┌─────────────────────────────────────┐
                        │           User (Streamlit UI)        │
                        └──────────────┬──────────────────────┘
                                       │
                            ┌──────────▼──────────┐
                            │    PlannerAgent      │
                            │  (schema + problem   │
                            │   type detection)    │
                            └──────────┬──────────┘
                        ┌─────────────┼─────────────┐
               ┌────────▼──────┐  ┌──▼────────┐  ┌─▼────────────┐
               │   EDAAgent    │  │ Cleaner   │  │  Trainer     │
               │  (profiling & │  │  Agent    │  │   Agent      │
               │  health alerts│  │(imputation│  │ (4 models,   │
               └───────────────┘  │ & cleaning│  │  leaderboard)│
                                  └───────────┘  └──────┬───────┘
                                                        │
                                           ┌────────────▼────────────┐
                                           │   ExplainabilityAgent   │
                                           │  (importances, sandbox) │
                                           └────────────┬────────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │    ReportAgent     │
                                              │ (HTML + Markdown)  │
                                              └────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- `pip`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/AutoML-Agent.git
cd AutoML-Agent

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 📊 Supported Models

### Classification
| Model | Algorithm |
|---|---|
| Random Forest Classifier | Ensemble of decision trees |
| Gradient Boosting Classifier | Sequential boosted trees |
| Logistic Regression | Linear model with log-loss |
| Decision Tree Classifier | Single interpretable tree |

### Regression
| Model | Algorithm |
|---|---|
| Random Forest Regressor | Ensemble of decision trees |
| Gradient Boosting Regressor | Sequential boosted trees |
| Ridge Regression | Regularized linear regression |
| Decision Tree Regressor | Single interpretable tree |

The best model is selected automatically by **Accuracy** (Classification) or **RMSE** (Regression) and saved to `models/model.joblib`.

---

## 📈 Metrics

| Problem Type | Primary Metric | Additional Metrics |
|---|---|---|
| Classification | Accuracy | Precision, Recall, F1 Score |
| Regression | RMSE | MAE, R² Score |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

The test suite covers all six agents end-to-end:

- `test_planner_agent` – problem type detection for Classification & Regression
- `test_cleaner_agent` – imputation and cleaning log
- `test_eda_agent` – statistical profiling and health alerts
- `test_trainer_agent_classification` – model training, leaderboard, and inference
- `test_trainer_agent_regression` – regression pipeline and metrics
- `test_explainability_agent` – feature importances and instance-level explanations
- `test_report_agent` – Markdown & HTML report generation

---

## 📥 Exported Artifacts

After training, the following files can be downloaded directly from the **Report & Export** tab:

| Artifact | Format | Description |
|---|---|---|
| Trained Model | `.joblib` | Serialized sklearn pipeline (preprocessor + best model) |
| Cleaned Dataset | `.csv` | Dataset after applying all cleaning steps |
| Executive Report | `.html` | Styled, shareable report with all findings |
| Executive Report | `.md` | Markdown version of the same report |

---

## 🔧 Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Interactive web dashboard |
| [scikit-learn](https://scikit-learn.org) | ML pipelines, models, preprocessing |
| [pandas](https://pandas.pydata.org) | Data manipulation |
| [NumPy](https://numpy.org) | Numerical operations |
| [Matplotlib / Seaborn](https://seaborn.pydata.org) | Visualizations |
| [XGBoost](https://xgboost.readthedocs.io) | Gradient boosting (available as extension) |
| [joblib](https://joblib.readthedocs.io) | Model serialization |
| [ReportLab](https://www.reportlab.com) | PDF generation support |
| [pytest](https://docs.pytest.org) | Testing framework |

---

## 📝 Notes

- The project is intentionally **framework-agnostic** and uses only scikit-learn, so it runs fully offline without any cloud MLOps dependencies.
- The `PlannerAgent` auto-detects the target column by matching common column name patterns (`target`, `label`, `price`, `churn`, etc.), falling back to the last column if none match.
- All preprocessing (imputation, scaling, encoding) is encapsulated in a single sklearn `Pipeline` object, ensuring no data leakage between train and test sets.
- Saved model bundles include the full preprocessing pipeline, so inference on new data requires only raw feature values.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
