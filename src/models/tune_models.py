"""
src/models/tune_models.py
--------------------------
Standalone hyperparameter tuning script.

Run from the project root::

    python -m src.models.tune_models

For each of the four models (Random Forest, Gradient Boosting, Lasso, MLP)
the script performs RandomizedSearchCV, prints the best parameters, and
writes a JSON summary to ``models/tune_results.json`` so users can copy the
winning hyperparameters back into ``config.py``.

The search spaces and CV / n_iter settings are defined inline rather than
imported from ``train_model.py`` so this script is fully self-contained.
"""

import json
import time
import warnings
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from config import (
    CV_FOLDS,
    MODELS_DIR,
    N_ITER_SEARCH,
    PROCESSED_DATA_FILE,
    RANDOM_STATE,
    SELECTED_FEATURES,
    TARGET_COLUMN,
    TEST_SIZE,
)

warnings.filterwarnings("ignore")

# ── Search spaces ─────────────────────────────────────────────────────────────

RF_PARAM_GRID = {
    "n_estimators": [int(x) for x in np.linspace(10, 1000, 10)],
    "max_features": ["sqrt", None],
    "max_depth": [int(x) for x in np.linspace(10, 110, 11)] + [None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

GB_PARAM_GRID = {
    "n_estimators": list(range(100, 1000, 100)),
    "max_depth": list(range(2, 12)),
    "loss": ["ls", "lad", "huber"],   # sklearn 0.21 names
    "alpha": list(np.linspace(0.00001, 0.99, 50)),
    "learning_rate": list(
        np.logspace(np.log(0.005), np.log(0.2), base=np.e, num=30)
    ),
    "min_samples_leaf": list(range(1, 8)),
}

LASSO_PARAM_GRID = {
    "tol": [0.01, 0.001, 0.0001, 0.00001],
    "max_iter": [500, 600, 800, 1000, 1300, 1500],
    "alpha": list(np.linspace(0.00001, 0.99, 50)),
}

MLP_PARAM_GRID = {
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

# MLP tuning is run with more iterations because the search space is larger
MLP_N_ITER = 700


# ── Core search runner ────────────────────────────────────────────────────────

def run_search(
    name: str,
    estimator,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = N_ITER_SEARCH,
) -> dict:
    """
    Fit RandomizedSearchCV for *estimator* and return a result dict with:
    ``best_params``, ``best_score`` (negative MSE → converted to RMSE),
    and ``elapsed_min``.
    """
    print(f"\n{'═' * 60}")
    print(f"Tuning {name}  (n_iter={n_iter}, cv={CV_FOLDS})")
    print(f"{'═' * 60}")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    t0 = time.time()
    search.fit(X, y)
    elapsed = (time.time() - t0) / 60

    best_rmse = np.sqrt(-search.best_score_)
    print(f"\nBest CV RMSE : {best_rmse:.4f}")
    print(f"Elapsed      : {elapsed:.2f} min")
    print("Best params  :")
    pprint(search.best_params_)

    return {
        "best_params": search.best_params_,
        "best_cv_rmse": float(best_rmse),
        "elapsed_min": round(elapsed, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load processed data
    df = pd.read_csv(PROCESSED_DATA_FILE)
    feature_cols = [c for c in SELECTED_FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scaled features for Lasso / MLP
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    results: dict = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    results["RandomForest"] = run_search(
        "Random Forest",
        RandomForestRegressor(random_state=RANDOM_STATE),
        RF_PARAM_GRID,
        X_train,
        y_train,
    )

    # ── Gradient Boosting ─────────────────────────────────────────────────────
    results["GradientBoosting"] = run_search(
        "Gradient Boosting",
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        GB_PARAM_GRID,
        X_train,
        y_train,
    )

    # ── Lasso (uses scaled features) ─────────────────────────────────────────
    results["Lasso"] = run_search(
        "Lasso",
        Lasso(),
        LASSO_PARAM_GRID,
        X_train_sc,
        y_train,
    )

    # ── MLP (uses scaled features, more iterations) ───────────────────────────
    results["MLP"] = run_search(
        "MLP",
        MLPRegressor(random_state=RANDOM_STATE),
        MLP_PARAM_GRID,
        X_train_sc,
        y_train,
        n_iter=MLP_N_ITER,
    )

    # ── Persist results ───────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "tune_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n{'═' * 60}")
    print(f"Tuning complete.  Results saved to '{out_path}'.")
    print("Copy the best_params values into config.py to update RF_PARAMS,")
    print("GB_PARAMS, LASSO_PARAMS, and MLP_PARAMS.")


if __name__ == "__main__":
    main()
