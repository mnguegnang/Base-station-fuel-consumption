"""
src/features/build_features.py
-------------------------------
Feature engineering: column selection, categorical encoding, scaling, and
feature-importance analysis extracted from the original notebook.

Public API
----------
    select_modelling_columns(df)       -> pd.DataFrame   (data_all_var)
    encode_features(data_all_var)      -> tuple[pd.DataFrame, pd.DataFrame]
    select_top_features(features, y)   -> pd.DataFrame
    scale_features(X)                  -> tuple[pd.DataFrame, StandardScaler]
    build_features(df)                 -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from typing import Tuple

from config import (
    CATEGORICAL_COLUMNS,
    FIGURES_DIR,
    NUMERIC_COLUMNS_TO_DROP,
    SELECTED_FEATURES,
    TARGET_COLUMN,
)


# ── 1. Column selection for modelling ─────────────────────────────────────────

def select_modelling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns relevant to modelling (numeric + categoricals).
    Maps directly to the ``data_all_var`` variable in the original notebook.
    """
    desired = [
        "Cluster",
        "NUMBER OF DAYS",
        "TYPE OF GENERATOR",
        "GENERATOR 1 CAPACITY (KVA)",
        "CURRENT HOUR METER GE1",
        "PREVIOUS HOUR METER G1",
        "PREVIOUS FUEL QTE",
        "QTE FUEL FOUND",
        "QTE FUEL ADDED",
        "TOTALE QTE LEFT",
        TARGET_COLUMN,
        "RUNNING TIME",
        "CONSUMPTION RATE",
        "Total DC (Amps)",
        "Ph2 (Amps)",
        "Ph3 (Amps)",
        "Fuel_per_period",
    ]
    available = [c for c in desired if c in df.columns]
    return df[available].copy()


# ── 2. Categorical encoding + numeric subset ──────────────────────────────────

def encode_features(data_all_var: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categorical columns and concatenate with numeric columns.

    Returns
    -------
    features  : full feature matrix (numeric + dummies), including target
    num       : numeric-only subset (used for correlation analysis)
    """
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in data_all_var.columns]
    categorical_subset = pd.get_dummies(data_all_var[cat_cols])

    drop_from_num = cat_cols + [c for c in NUMERIC_COLUMNS_TO_DROP if c in data_all_var.columns]
    num = data_all_var.drop(columns=drop_from_num, errors="ignore")

    features = pd.concat([num, categorical_subset], axis=1)
    print(
        f"Feature matrix: {features.shape[1]} columns "
        f"({num.shape[1]} numeric + {categorical_subset.shape[1]} dummies)."
    )
    return features, num


# ── 3. Feature importance ─────────────────────────────────────────────────────

def plot_feature_importance(
    features: pd.DataFrame,
    y: pd.Series,
    model_name: str = "RandomForest",
    top_n: int = 19,
    save: bool = True,
) -> pd.Series:
    """
    Fit a tree-based model and plot/return feature importances.

    Parameters
    ----------
    features   : full feature matrix (including target column)
    y          : target series
    model_name : 'RandomForest' or 'ExtraTrees'
    top_n      : number of top features to display
    save       : persist the figure to FIGURES_DIR

    Returns
    -------
    pd.Series of feature importances (all features, not just top N)
    """
    X = features.drop(columns=[TARGET_COLUMN], errors="ignore")

    if model_name == "ExtraTrees":
        clf = ExtraTreesRegressor()
    else:
        clf = RandomForestRegressor()

    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values()

    top_importances = importances.iloc[-top_n:]
    plt.figure(figsize=(20, 12))
    plt.barh(range(len(top_importances)), top_importances)
    plt.yticks(range(len(top_importances)), top_importances.index, size=15)
    plt.xticks(size=25)
    plt.xlabel("Feature Importance", size=20)
    plt.ylabel("Features", size=20)
    plt.title(f"Top {top_n} Features – {model_name}", size=22)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"feature_importance_{model_name}.png"
        plt.savefig(path)
        print(f"Feature importance plot saved to '{path}'.")

    plt.show()
    return importances


# ── 4. Select top features ────────────────────────────────────────────────────

def select_top_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Return the pre-selected top feature columns (from notebook analysis).
    Missing columns are silently skipped so the function remains robust
    across different versions of the processed dataset.
    """
    available = [c for c in SELECTED_FEATURES if c in features.columns]
    missing = set(SELECTED_FEATURES) - set(available)
    if missing:
        print(f"Warning: these selected features were not found and will be skipped: {missing}")
    data = features[available].copy()
    null_counts = data.isnull().sum()
    if null_counts.any():
        print("Missing values in selected features:\n", null_counts[null_counts > 0])
    return data


# ── 5. Feature scaling ────────────────────────────────────────────────────────

def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standard-scale features (required for MLP).

    Returns
    -------
    X_scaled : pd.DataFrame with the same columns
    scaler   : fitted StandardScaler (use to transform new data)
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, scaler


# ── Full feature pipeline ─────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    plot_importance: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    End-to-end feature pipeline.

    Returns
    -------
    X       : top-selected feature matrix (unscaled)
    features: full encoded feature matrix
    y       : target series (CONSUMPTION HIS)
    """
    data_all_var = select_modelling_columns(df)
    features, _num = encode_features(data_all_var)
    y = features[TARGET_COLUMN]

    if plot_importance:
        plot_feature_importance(features, y, model_name="RandomForest")
        plot_feature_importance(features, y, model_name="ExtraTrees")

    X = select_top_features(features)
    print(f"\nFinal feature matrix X: {X.shape}, target y: {y.shape}")
    return X, features, y
