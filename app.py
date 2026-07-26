from __future__ import annotations

from pathlib import Path
import hashlib
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from agents.cleaner import CleanerAgent
from agents.eda import EDAAgent
from agents.explain import ExplainabilityAgent
from agents.planner import PlannerAgent
from agents.report import ReportAgent
from agents.trainer import TrainerAgent
from utils.visualization import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importances,
    plot_missing_values,
    plot_residuals,
)

st.set_page_config(page_title="AutoML Agent Dashboard", page_icon="🤖", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #334155;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🤖 AutoML Agent Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a dataset, perform EDA, clean data, benchmark models, and explain predictions in minutes.</div>',
    unsafe_allow_html=True,
)

# Sidebar data source selection
st.sidebar.header("📁 Data Input")
upload_option = st.sidebar.radio("Select Dataset Source:", ["Upload CSV", "Use Sample Dataset"])

df = None
dataset_id = None  # Used to detect dataset changes

if upload_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataset_id = hashlib.md5(uploaded_file.getvalue()).hexdigest()
else:
    sample_choice = st.sidebar.selectbox(
        "Choose Sample Dataset:",
        ["Classification (Iris / Synthetic)", "Regression (Housing / Synthetic)"],
    )
    dataset_id = sample_choice  # Use choice string as identifier

    if "Classification" in sample_choice:
        np.random.seed(0)
        df = pd.DataFrame({
            "age": np.random.randint(18, 65, 120),
            "income": np.random.randint(30000, 120000, 120).astype(float),
            "credit_score": np.random.randint(550, 850, 120),
            "debt_ratio": np.round(np.random.uniform(0.1, 0.9, 120), 2),
            "approved": np.random.choice(["Yes", "No"], 120, p=[0.6, 0.4]),
        })
        df.loc[::10, "income"] = np.nan
        df.loc[::15, "debt_ratio"] = np.nan
    else:
        np.random.seed(1)
        df = pd.DataFrame({
            "square_feet": np.random.randint(600, 3500, 120).astype(float),
            "bedrooms": np.random.randint(1, 5, 120),
            "bathrooms": np.random.randint(1, 4, 120),
            "year_built": np.random.randint(1980, 2022, 120),
            "price": np.random.randint(150000, 750000, 120),
        })
        df.loc[::8, "square_feet"] = np.nan

if df is not None:
    # ----------------------------------------------------------------
    # Detect dataset / target changes and reset session state
    # ----------------------------------------------------------------
    planner = PlannerAgent(df)
    plan = planner.analyze()

    st.sidebar.divider()
    st.sidebar.header("🎯 Target Selection")
    target_col = st.sidebar.selectbox(
        "Select Target Column:",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(plan["target"]) if plan["target"] in df.columns else len(df.columns) - 1,
    )

    # Re-run planner with user-selected target
    plan = planner.analyze(target_column=target_col)

    # Build a fingerprint combining dataset identity + target column
    state_key = f"{dataset_id}__{target_col}"
    if st.session_state.get("_state_key") != state_key:
        # Dataset or target changed — wipe all stale session artifacts
        for k in ["cleaned_df", "cleaning_summary", "train_result", "explain_info", "md_report", "html_report"]:
            st.session_state.pop(k, None)
        st.session_state["_state_key"] = state_key

    # ----------------------------------------------------------------
    # Pre-compute EDA and initial cleaning at top level (always fresh)
    # ----------------------------------------------------------------
    eda_agent = EDAAgent(df, target_col=target_col)
    eda_summary = eda_agent.analyze()

    # Auto-clean with defaults on first load (so cleaned_df is always available)
    if "cleaned_df" not in st.session_state:
        _default_cleaner = CleanerAgent(df, target_col=target_col)
        st.session_state["cleaned_df"] = _default_cleaner.clean()
        st.session_state["cleaning_summary"] = _default_cleaner.get_summary()

    # Header Metrics Bar
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Rows", plan["rows"])
    with col2:
        st.metric("Columns", plan["columns"])
    with col3:
        st.metric("Target Variable", plan["target"])
    with col4:
        st.metric("Problem Type", plan["problem_type"])
    with col5:
        st.metric("Suggested Metric", plan["metric"])

    st.divider()

    # Create Tabs
    tab_eda, tab_clean, tab_train, tab_explain, tab_report = st.tabs([
        "📊 Dataset & EDA",
        "🧹 Data Cleaning",
        "🚀 Model Leaderboard",
        "🔍 Explainability & Sandbox",
        "📄 Report & Export",
    ])

    # ----------------------------------------------------------------
    # TAB 1: Dataset & EDA
    # ----------------------------------------------------------------
    with tab_eda:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), width="stretch")

        st.subheader("⚠️ Data Health & Quality Alerts")
        alerts = eda_summary["health_alerts"]
        if alerts:
            for alert in alerts:
                sev = alert["severity"]
                msg = f"**[{alert['category']}]** {alert['message']}"
                if sev == "HIGH":
                    st.error(msg)
                elif sev == "MEDIUM":
                    st.warning(msg)
                else:
                    st.info(msg)
        else:
            st.success("No critical data quality issues detected.")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Numeric Features Summary")
            num_stats = eda_summary["numeric_stats"]
            if not num_stats.empty:
                st.dataframe(num_stats, width="stretch")
            else:
                st.info("No numeric features found.")

        with col_right:
            st.subheader("Categorical Features Summary")
            cat_stats = eda_summary["categorical_stats"]
            if not cat_stats.empty:
                st.dataframe(cat_stats, width="stretch")
            else:
                st.info("No categorical features found.")

        st.subheader("Visual Analysis")
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.markdown("##### Missing Values Breakdown")
            fig_missing = plot_missing_values(df)
            if fig_missing:
                st.pyplot(fig_missing)
            else:
                st.success("No missing values in dataset!")

        with viz_col2:
            st.markdown("##### Correlation Matrix")
            corr_matrix = eda_summary["correlation_matrix"]
            if not corr_matrix.empty:
                fig_corr = plot_correlation_heatmap(corr_matrix)
                st.pyplot(fig_corr)
            else:
                st.info("Insufficient numeric columns for correlation matrix.")

    # ----------------------------------------------------------------
    # TAB 2: Data Cleaning
    # ----------------------------------------------------------------
    with tab_clean:
        st.subheader("Data Cleaning Options")

        clean_col1, clean_col2 = st.columns(2)
        with clean_col1:
            num_strat = st.selectbox("Numeric Imputation Strategy:", ["median", "mean", "zero"])
            drop_dups = st.checkbox("Drop Duplicate Rows", value=True)
        with clean_col2:
            cat_strat = st.selectbox("Categorical Imputation Strategy:", ["missing", "mode"])
            drop_const = st.checkbox("Drop Constant Feature Columns", value=True)

        cleaner = CleanerAgent(
            df,
            target_col=target_col,
            numeric_impute_strategy=num_strat,
            categorical_impute_strategy=cat_strat,
            drop_duplicates=drop_dups,
            drop_constant_cols=drop_const,
        )

        if st.button("Apply Data Cleaning"):
            cleaned_df = cleaner.clean()
            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["cleaning_summary"] = cleaner.get_summary()
            # Wipe training artifacts so they are re-run against the new cleaned data
            for k in ["train_result", "explain_info", "md_report", "html_report"]:
                st.session_state.pop(k, None)

        st.success("Cleaned dataset ready for training.")
        st.markdown(f"**Cleaned Dataset Shape:** `{st.session_state['cleaned_df'].shape}`")

        with st.expander("View Cleaning Log", expanded=True):
            for item in st.session_state.get("cleaning_summary", {}).get("cleaning_log", []):
                st.write(f"- {item}")

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(st.session_state["cleaned_df"].head(10), width="stretch")

    # ----------------------------------------------------------------
    # TAB 3: Model Leaderboard
    # ----------------------------------------------------------------
    with tab_train:
        st.subheader("Automated Model Benchmarking & Training")
        current_cleaned_df = st.session_state["cleaned_df"]

        # Safety guard: ensure target_col exists in the cleaned dataframe
        if target_col not in current_cleaned_df.columns:
            st.error(
                f"Target column **'{target_col}'** is missing from the cleaned dataset. "
                "Please re-apply cleaning on the Data Cleaning tab."
            )
        else:
            if st.button("🚀 Train AutoML Models", type="primary"):
                with st.spinner("Training scikit-learn models, optimizing pipelines, and evaluating metrics..."):
                    trainer = TrainerAgent(
                        current_cleaned_df,
                        target_col=target_col,
                        problem_type=plan["problem_type"],
                        output_dir="models",
                    )
                    train_result = trainer.run()
                    st.session_state["train_result"] = train_result

                    # Explainability
                    explainer = ExplainabilityAgent(train_result["model_bundle"])
                    importances = explainer.get_feature_importances()
                    drivers = explainer.get_top_drivers_summary()
                    st.session_state["explain_info"] = {
                        "importances": importances,
                        "drivers_summary": drivers,
                        "explainer": explainer,
                    }

                    # Report
                    reporter = ReportAgent(
                        plan_info=plan,
                        eda_info=eda_summary,
                        cleaning_summary=st.session_state.get("cleaning_summary", {}),
                        training_info=train_result,
                        explain_info=st.session_state["explain_info"],
                    )
                    md_path, html_path = reporter.save_reports("reports")
                    st.session_state["md_report"] = md_path.read_text(encoding="utf-8")
                    st.session_state["html_report"] = html_path.read_text(encoding="utf-8")

                st.success(f"Best Model Trained: **{train_result['best_model']}**")

            if "train_result" in st.session_state:
                result = st.session_state["train_result"]
                st.subheader("🏆 Model Leaderboard")
                leaderboard_df = pd.DataFrame([
                    {
                        "Rank": idx + 1,
                        "Model": item["model_name"],
                        **item["metrics"],
                    }
                    for idx, item in enumerate(result["leaderboard"])
                ])
                st.dataframe(leaderboard_df, width="stretch")

                st.subheader("Performance Visualizations")
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    if result["problem_type"] == "Classification" and result["confusion_matrix"]:
                        cm_fig = plot_confusion_matrix(
                            np.array(result["confusion_matrix"]),
                            labels=result["class_labels"] or [],
                        )
                        st.pyplot(cm_fig)
                    else:
                        res_fig = plot_residuals(result["test_actuals"], result["test_predictions"])
                        st.pyplot(res_fig)

                with m_col2:
                    st.json(result["metrics"])
            else:
                st.info("Click '🚀 Train AutoML Models' to run model evaluation.")

    # ----------------------------------------------------------------
    # TAB 4: Explainability & Live Prediction Sandbox
    # ----------------------------------------------------------------
    with tab_explain:
        st.subheader("Model Interpretability & Feature Importances")

        if "explain_info" in st.session_state:
            ex_info = st.session_state["explain_info"]
            fig_imp = plot_feature_importances(ex_info["importances"])
            st.pyplot(fig_imp)

            st.markdown("#### Key Feature Drivers")
            for driver_txt in ex_info["drivers_summary"]:
                st.markdown(f"- {driver_txt}")

            st.divider()
            st.subheader("🔮 Live Prediction Sandbox")
            st.markdown("Enter custom feature values below to get a real-time model prediction:")

            current_cleaned_df = st.session_state["cleaned_df"]
            feature_cols = [c for c in current_cleaned_df.columns if c != target_col]

            sandbox_inputs = {}
            grid_cols = st.columns(3)

            for i, col_name in enumerate(feature_cols):
                with grid_cols[i % 3]:
                    col_series = current_cleaned_df[col_name]
                    if pd.api.types.is_numeric_dtype(col_series):
                        min_val = float(col_series.min())
                        max_val = float(col_series.max())
                        default_val = float(col_series.median())
                        if min_val == max_val:
                            sandbox_inputs[col_name] = default_val
                        else:
                            sandbox_inputs[col_name] = st.number_input(
                                f"{col_name}",
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val,
                                key=f"sandbox_{col_name}",
                            )
                    else:
                        unique_options = col_series.dropna().unique().tolist()
                        sandbox_inputs[col_name] = st.selectbox(
                            f"{col_name}", options=unique_options, key=f"sandbox_{col_name}"
                        )

            if st.button("Predict Target Output"):
                try:
                    prediction = TrainerAgent.load_and_predict("models/model.joblib", sandbox_inputs)[0]
                    explainer = ex_info["explainer"]
                    instance_exp = explainer.explain_instance(sandbox_inputs)
                    st.success(f"**Predicted {target_col}:** `{prediction}`")
                    st.caption(f"Top Influential Feature for this sample: **{instance_exp['top_influencer']}**")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.info("Train AutoML models first to unlock feature explanations and interactive sandbox.")

    # ----------------------------------------------------------------
    # TAB 5: Report & Export
    # ----------------------------------------------------------------
    with tab_report:
        st.subheader("📄 Automated Executive Report")

        if "md_report" in st.session_state:
            st.markdown(st.session_state["md_report"])

            st.divider()
            st.subheader("📥 Download Artifacts")
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)

            with d_col1:
                model_path = Path("models/model.joblib")
                if model_path.exists():
                    st.download_button(
                        label="💾 Model (.joblib)",
                        data=model_path.read_bytes(),
                        file_name="automl_model.joblib",
                        mime="application/octet-stream",
                    )

            with d_col2:
                cleaned_csv = st.session_state["cleaned_df"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📊 Cleaned Data (CSV)",
                    data=cleaned_csv,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                )

            with d_col3:
                st.download_button(
                    label="📄 Report (HTML)",
                    data=st.session_state["html_report"].encode("utf-8"),
                    file_name="automl_report.html",
                    mime="text/html",
                )

            with d_col4:
                st.download_button(
                    label="📝 Report (Markdown)",
                    data=st.session_state["md_report"].encode("utf-8"),
                    file_name="automl_report.md",
                    mime="text/markdown",
                )
        else:
            st.info("Train AutoML models to generate downloadable executive reports.")
else:
    st.info("Select a dataset source from the sidebar to begin.")