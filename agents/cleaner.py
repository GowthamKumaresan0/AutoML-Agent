from __future__ import annotations

import pandas as pd


class CleanerAgent:
    """Clean dataset by handling duplicates, missing values, and column typing."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_col: str | None = None,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "missing",
        drop_duplicates: bool = True,
        drop_constant_cols: bool = True,
    ):
        self.df = dataframe.copy()
        self.target_col = target_col
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.drop_duplicates_flag = drop_duplicates
        self.drop_constant_cols_flag = drop_constant_cols
        self.cleaning_log: list[str] = []

    def clean(self) -> pd.DataFrame:
        """Execute data cleaning steps and return cleaned DataFrame."""
        df = self.df.copy()
        self.cleaning_log.clear()

        # 1. Drop duplicate rows
        if self.drop_duplicates_flag:
            initial_rows = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            dropped = initial_rows - len(df)
            if dropped > 0:
                self.cleaning_log.append(f"Removed {dropped} exact duplicate rows.")

        # 2. Drop constant columns (excluding target column)
        if self.drop_constant_cols_flag:
            cols_to_drop = []
            for col in df.columns:
                if self.target_col and col == self.target_col:
                    continue
                if df[col].nunique(dropna=True) <= 1:
                    cols_to_drop.append(col)

            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                self.cleaning_log.append(f"Dropped {len(cols_to_drop)} constant feature columns: {cols_to_drop}")

        # 3. Handle missing values and type normalization
        for col in df.columns:
            if self.target_col and col == self.target_col:
                continue

            missing_count = df[col].isnull().sum()

            if pd.api.types.is_numeric_dtype(df[col]):
                if missing_count > 0:
                    if self.numeric_impute_strategy == "median":
                        val = df[col].median()
                    elif self.numeric_impute_strategy == "mean":
                        val = df[col].mean()
                    else:
                        val = 0.0
                    df[col] = df[col].fillna(val)
                    self.cleaning_log.append(f"Imputed {missing_count} missing values in numeric feature '{col}' with {self.numeric_impute_strategy}={val:.4f}.")
            else:
                if missing_count > 0:
                    if self.categorical_impute_strategy == "mode" and not df[col].mode().empty:
                        val = str(df[col].mode().iloc[0])
                    else:
                        val = "missing"
                    df[col] = df[col].fillna(val)
                    self.cleaning_log.append(f"Imputed {missing_count} missing values in categorical feature '{col}' with '{val}'.")
                df[col] = df[col].astype(str)

        # 4. Handle target column specifically
        if self.target_col and self.target_col in df.columns:
            target_missing = df[self.target_col].isnull().sum()
            if target_missing > 0:
                df = df.dropna(subset=[self.target_col]).reset_index(drop=True)
                self.cleaning_log.append(f"Dropped {target_missing} rows missing the target variable '{self.target_col}'.")

        if not self.cleaning_log:
            self.cleaning_log.append("Dataset already clean; no missing values or duplicate rows detected.")

        return df

    def get_summary(self) -> dict:
        """Return cleaning summary metadata."""
        return {
            "initial_shape": self.df.shape,
            "cleaning_log": self.cleaning_log,
        }
