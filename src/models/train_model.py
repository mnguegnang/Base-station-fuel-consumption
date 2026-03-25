"""
src/models/train_model.py
--------------------------
Model training, k-fold evaluation, hyperparameter tuning, and serialisation
extracted from the original Fuel Consumption Jupyter Notebook.

Public API
----------
    compute_metrics(y_true, y_pred)          -> dict
    evaluate_model_kfold(model, X, y, ...)   -> dict
    tune_random_forest(X, y)                 -> RandomizedSearchCV
    tune_gradient_boosting(X, y)             -> RandomizedSearchCV
    tune_lasso(X, y)                         -> RandomizedSearchCV
    tune_mlp(X_scaled, y)                    -> RandomizedSearchCV
    train_final_models(X, y, X_scaled)       -> dict[str, estimator]
    save_model(model, name)                  -> Path
    run_training(X, y, X_scaled)             -> dict
"""

import pickle
import time
import warnings
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.neural_network import MLPRegressor

from config import (
    CV_FOLDS,
    GB_PARAMS,
    KFOLD_SPLITS,
    LASSO_PARAMS,
    MLP_PARAMS,
    MODELS_DIR,
    N_ITER_SEARCH,
    RANDOM_STATE,
    RF_PARAMS,
    TEST_SIZE,
)

warnings.filterwarnings("ignore")


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Compute hydrology/ML performance metrics used in the paper:
      Bias, MAE, RMSE, RSR, PBIAS, NSE (Nash-Sutcliffe Efficiency).
    """
    n = len(y_true)
    errors = np.abs(y_true - y_pred)
    residuals = y_true - y_pred

    bias = residuals.sum() / n
    mae = errors.mean()
    pbias = (residuals.sum() * 100) / y_true.sum()

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    nse = 1 - (ss_res / ss_tot)

    rmse = np.sqrt(ss_res / n)

    stdev = np.sqrt(ss_tot)
    rsr = np.sqrt(ss_res) / stdev if stdev != 0 else np.nan

    return {
        "Bias": bias,
        "MAE": mae,
        "RMSE": rmse,
        "RSR": rsr,
        "PBIAS": pbias,
        "NSE": nse,
    }


def _print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"Metrics – {name}")
    print(f"  Bias          : {metrics['Bias']:.4f}")
    print(f"  MAE           : {metrics['MAE']:.4f}")
    print(f"  RMSE          : {metrics['RMSE']:.4f}")
    print(f"  RSR           : {metrics['RSR']:.4f}")
    print(f"  PBIAS (%)     : {metrics['PBIAS']:.4f}")
    print(f"  NSE           : {metrics['NSE']:.4f}")


# ── K-Fold cross-validated evaluation ────────────────────────────────────────

def evaluate_model_kfold(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = KFOLD_SPLITS,
    model_name: str = "Model",
) -> dict:
    """
    Run k-fold cross-validation and return mean metrics across folds.

    Returns
    -------
    dict with keys: Bias, MAE, RMSE, RSR, PBIAS, NSE  (mean over folds)
    plus 'NSE_per_fold' list for boxplot comparisons.
    """
    kfold = KFold(n_splits=n_splits)
    fold_metrics: list[dict] = []

    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fold_metrics.append(compute_metrics(y_test, preds))

    mean_metrics = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
    mean_metrics["NSE_per_fold"] = [f["NSE"] for f in fold_metrics]
    _print_metrics(f"{model_name} (10-fold CV)", mean_metrics)
    return mean_metrics


# ── Hyperparameter search spaces ──────────────────────────────────────────────

def _rf_param_grid() -> dict:
    return {
        "n_estimators": [int(x) for x in np.linspace(10, 1000, 10)],
        "max_features": ["sqrt", None],
        "max_depth": [int(x) for x in np.linspace(10, 110, 11)] + [None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }


def _gb_param_grid() -> dict:
    return {
        "n_estimators": list(range(100, 1000, 100)),
        "max_depth": list(range(2, 12)),
        "loss": ["ls", "lad", "huber"],   # sklearn 0.21 names
        "alpha": list(np.linspace(0.00001, 0.99, 50)),
        "learning_rate": list(
            np.logspace(np.log(0.005), np.log(0.2), base=np.e, num=30)
        ),
        "min_samples_leaf": list(range(1, 8)),
    }


def _lasso_param_grid() -> dict:
    return {
        "tol": [0.01, 0.001, 0.0001, 0.00001],
        "max_iter": [500, 600, 800, 1000, 1300, 1500],
        "alpha": list(np.linspace(0.00001, 0.99, 50)),
    }


def _mlp_param_grid() -> dict:
    return {
        "hidden_layer_sizes": [1, 2, 3, 4],
        "alpha": list(np.linspace(0.00001, 0.99, 50)),
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "max_iter": [int(x) for x in np.linspace(100, 1000, 7)],
        "early_stopping": [True, False],
        "learning_rate_init": list(
            np.logspace(np.log(0.005), np.log(0.2), base=np.e, num=30)
        ),
        "activation": ["identity", "logistic", "tanh", "relu"],
    }


# ── Tuning functions ──────────────────────────────────────────────────────────

def _run_random_search(estimator, param_grid: dict, X, y, n_iter: int = N_ITER_SEARCH) -> RandomizedSearchCV:
    start = time.time()
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=CV_FOLDS,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X, y)
    elapsed = (time.time() - start) / 60
    print(f"Search completed in {elapsed:.2f} min. Best params:")
    pprint(search.best_params_)
    return search


def tune_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomizedSearchCV:
    """Run RandomizedSearchCV for Random Forest and return the fitted search."""
    print("\nTuning Random Forest …")
    return _run_random_search(
        RandomForestRegressor(random_state=RANDOM_STATE),
        _rf_param_grid(), X, y
    )


def tune_gradient_boosting(X: pd.DataFrame, y: pd.Series) -> RandomizedSearchCV:
    """Run RandomizedSearchCV for Gradient Boosting and return the fitted search."""
    print("\nTuning Gradient Boosting …")
    return _run_random_search(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        _gb_param_grid(), X, y
    )


def tune_lasso(X: pd.DataFrame, y: pd.Series) -> RandomizedSearchCV:
    """Run RandomizedSearchCV for Lasso and return the fitted search."""
    print("\nTuning Lasso …")
    return _run_random_search(Lasso(), _lasso_param_grid(), X, y)


def tune_mlp(X_scaled: pd.DataFrame, y: pd.Series) -> RandomizedSearchCV:
    """Run RandomizedSearchCV for MLP (requires scaled data) and return the search."""
    print("\nTuning MLP …")
    return _run_random_search(
        MLPRegressor(random_state=RANDOM_STATE),
        _mlp_param_grid(), X_scaled, y, n_iter=700
    )


# ── Train final (best-params) models ─────────────────────────────────────────

def train_final_models(
    X: pd.DataFrame,
    y: pd.Series,
    X_scaled: pd.DataFrame,
) -> dict:
    """
    Instantiate and k-fold evaluate all four models with the best
    hyperparameters found during the tuning phase.

    Returns
    -------
    dict  {'RF': fitted_model, 'GB': fitted_model, 'MLP': fitted_model, 'Lasso': fitted_model}
    """
    models = {
        "RF": RandomForestRegressor(**RF_PARAMS),
        "GB": GradientBoostingRegressor(**GB_PARAMS),
        "MLP": MLPRegressor(**MLP_PARAMS),
        "Lasso": Lasso(**LASSO_PARAMS),
    }
    data = {"RF": (X, y), "GB": (X, y), "MLP": (X_scaled, y), "Lasso": (X, y)}

    results: dict = {}
    for name, model in models.items():
        Xi, yi = data[name]
        metrics = evaluate_model_kfold(model, Xi, yi, model_name=name)
        results[name] = {"model": model, "metrics": metrics}
        # Refit on the full dataset for serialisation
        model.fit(Xi, yi)

    return results


# ── Model serialisation ───────────────────────────────────────────────────────

def save_model(model, name: str) -> Path:
    """Pickle *model* to MODELS_DIR/<name>.pkl and return the path."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model '{name}' saved to '{path}'.")
    return path


def load_model(name: str):
    """Load and return a pickled model from MODELS_DIR/<name>.pkl."""
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Convenience runner ────────────────────────────────────────────────────────

def run_training(
    X: pd.DataFrame,
    y: pd.Series,
    X_scaled: pd.DataFrame,
    tune: bool = False,
) -> dict:
    """
    Full training workflow.

    Parameters
    ----------
    X        : unscaled feature matrix
    y        : target series
    X_scaled : StandardScaler-transformed X (for MLP)
    tune     : if True, run RandomizedSearchCV before final evaluation
               (expensive – set False to use pre-tuned params from config)

    Returns
    -------
    dict of {model_name: {'model': estimator, 'metrics': dict}}
    """
    if tune:
        print("=== Hyperparameter Tuning ===")
        tune_random_forest(X, y)
        tune_gradient_boosting(X, y)
        tune_lasso(X, y)
        tune_mlp(X_scaled, y)

    print("\n=== Final Model Evaluation (10-fold CV) ===")
    results = train_final_models(X, y, X_scaled)

    print("\n=== Saving Models ===")
    for name, info in results.items():
        save_model(info["model"], name)

    print("\n=== Saving Evaluation Metrics ===")
    from config import EVAL_METRICS_DIR
    EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    nse_rows = [
        {"Models": name, "NSE": nse_val}
        for name, info in results.items()
        for nse_val in info["metrics"]["NSE_per_fold"]
    ]
    pd.DataFrame(nse_rows).to_csv(EVAL_METRICS_DIR / "models-nse.csv", index=False)

    stats_rows = [
        {"Models": name, **{k: v for k, v in info["metrics"].items() if k != "NSE_per_fold"}}
        for name, info in results.items()
    ]
    pd.DataFrame(stats_rows).to_csv(EVAL_METRICS_DIR / "models-statistic-metrics.csv", index=False)
    print(f"Evaluation metrics saved to '{EVAL_METRICS_DIR}'.")

    return results


if __name__ == "__main__":
    from src.data.make_dataset import run_pipeline
    from src.features.build_features import build_features, scale_features

    df = run_pipeline(save=False)
    X, features, y = build_features(df)
    X_scaled, _ = scale_features(X)
    run_training(X, y, X_scaled, tune=False)
