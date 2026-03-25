"""
src/models/predict_model.py
----------------------------
Prediction, per-cluster aggregation, and model evaluation reports extracted
from the original Fuel Consumption Jupyter Notebook.

Public API
----------
    predict(model, X_test)                           -> np.ndarray
    build_prediction_dataframe(model, X_test, y_test, df_orig) -> pd.DataFrame
    aggregate_by_cluster(pred_df)                    -> pd.DataFrame
    compare_models(results, X, y, X_scaled)          -> pd.DataFrame
    run_inference(model_name, X, y, df_orig, scaler) -> pd.DataFrame
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from config import KFOLD_SPLITS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.models.train_model import compute_metrics, load_model

warnings.filterwarnings("ignore")


# ── Single-model prediction ───────────────────────────────────────────────────

def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Return predictions for *X* using an already-fitted *model*."""
    return model.predict(X)


# ── Rich prediction DataFrame ─────────────────────────────────────────────────

def build_prediction_dataframe(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_orig: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a DataFrame with columns Cluster, Date, Observed, Predicted.

    Parameters
    ----------
    model    : fitted estimator
    X_test   : test feature matrix (index aligned with y_test and df_orig)
    y_test   : ground-truth target values
    df_orig  : cleaned DataFrame that still holds 'Cluster' and date columns
    """
    preds = model.predict(X_test)
    idx = y_test.index

    date_col = next(
        (c for c in ["EFFECTIVE DATE OF VISIT", "Date"] if c in df_orig.columns),
        None,
    )
    dates = df_orig[date_col][idx].tolist() if date_col else [None] * len(idx)
    clusters = df_orig["Cluster"][idx].values if "Cluster" in df_orig.columns else [None] * len(idx)

    pred_df = pd.DataFrame(
        {
            "Cluster": clusters,
            "Date": dates,
            "Observed": list(y_test),
            "Predicted": list(preds),
        }
    ).sort_index()

    return pred_df


# ── Cluster-level aggregation ─────────────────────────────────────────────────

def aggregate_by_cluster(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Sum Observed and Predicted consumption per Cluster."""
    return pred_df.groupby("Cluster")[["Observed", "Predicted"]].sum()


# ── Cross-model R² comparison ─────────────────────────────────────────────────

def compare_models(
    model_map: dict,
    X: pd.DataFrame,
    y: pd.Series,
    X_scaled: pd.DataFrame,
    n_splits: int = KFOLD_SPLITS,
) -> pd.DataFrame:
    """
    Evaluate each model with 10-fold cross-validation (R²) and return a
    summary DataFrame.

    Parameters
    ----------
    model_map : dict  {name: estimator}  (unfitted models)
    X         : unscaled feature matrix
    y         : target
    X_scaled  : scaled feature matrix (used for MLP)
    """
    kfold = KFold(n_splits=n_splits, random_state=RANDOM_STATE, shuffle=True)
    rows = []
    for name, model in model_map.items():
        Xi = X_scaled if name == "MLP" else X
        scores = cross_val_score(model, Xi, y, cv=kfold, scoring="r2")
        rows.append(
            {
                "Model": name,
                "R2_mean": scores.mean(),
                "R2_std": scores.std(),
                "R2_scores": list(scores),
            }
        )
    summary = (
        pd.DataFrame(rows)
        .sort_values("R2_mean", ascending=False)
        .reset_index(drop=True)
    )
    print(summary[["Model", "R2_mean", "R2_std"]].to_string(index=False))
    return summary


# ── Full inference runner ─────────────────────────────────────────────────────

def run_inference(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    df_orig: pd.DataFrame,
    scaler=None,
) -> pd.DataFrame:
    """
    Load a saved model, run train/test split, predict, and return a
    prediction DataFrame.

    Parameters
    ----------
    model_name : one of 'RF', 'GB', 'MLP', 'Lasso'
    X          : feature matrix (unscaled)
    y          : target series
    df_orig    : cleaned DataFrame (for cluster/date lookup)
    scaler     : fitted StandardScaler (required only for MLP)
    """
    model = load_model(model_name)

    if model_name == "MLP" and scaler is not None:
        Xi = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    else:
        Xi = X

    X_train, X_test, y_train, y_test = train_test_split(
        Xi, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    metrics = compute_metrics(y_test, model.predict(X_test))
    print(f"\n=== {model_name} hold-out test metrics ===")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    pred_df = build_prediction_dataframe(model, X_test, y_test, df_orig)
    return pred_df


if __name__ == "__main__":
    from src.data.make_dataset import run_pipeline
    from src.features.build_features import build_features, scale_features

    df = run_pipeline(save=False)
    X, features, y = build_features(df)
    X_scaled, scaler = scale_features(X)

    for name in ["RF", "GB", "MLP", "Lasso"]:
        pred_df = run_inference(name, X, y, df, scaler=scaler)
        cluster_summary = aggregate_by_cluster(pred_df)
        print(f"\nCluster-level summary for {name}:")
        print(cluster_summary.head())
