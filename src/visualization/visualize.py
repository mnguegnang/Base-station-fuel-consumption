"""
src/visualization/visualize.py
-------------------------------
All plotting routines extracted from the original Fuel Consumption Notebook.

Public API
----------
    plot_consumption_time_series(df)
    plot_rolling_mean(df)
    plot_consumption_vs_rate(df, goodrate_only)
    plot_distribution(series, label, title)
    plot_boxplot_by_cluster(df)
    plot_correlation_matrix(num_df)
    plot_scatter_matrix(df)
    plot_pred_vs_obs(pred_df, model_name)
    plot_cluster_bar(cluster_df, model_name)
    plot_nse_boxplot(nse_table)
    plot_r2_boxplot(score_table)
"""

import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm

from config import FIGURES_DIR, EDA_FIGURES_DIR, TRAINING_FIGURES_DIR, TARGET_COLUMN

warnings.filterwarnings("ignore")
sns.set()


# ── Internal helper ───────────────────────────────────────────────────────────

def _save(filename: str, dirpath=None) -> None:
    """Save the current figure to dirpath (defaults to FIGURES_DIR)."""
    if dirpath is None:
        dirpath = FIGURES_DIR
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / filename
    plt.savefig(path, bbox_inches="tight")
    print(f"Figure saved: '{path}'.")


# ── Time series ───────────────────────────────────────────────────────────────

def plot_consumption_time_series(df: pd.DataFrame, save: bool = True) -> None:
    """Plot CONSUMPTION HIS and CONSUMPTION RATE side by side over visit date."""
    date_col = "EFFECTIVE DATE OF VISIT"
    t1 = df.pivot_table(TARGET_COLUMN, index=date_col, aggfunc="mean").squeeze()
    t2 = df.pivot_table("CONSUMPTION RATE", index=date_col, aggfunc="mean").squeeze()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))

    t1.plot(ax=ax1)
    ax1.set_ylabel("Consumption HIS mean", size=20)
    ax1.set_title("Consumption Trend", size=23)
    ax1.tick_params(axis="x", labelsize=16, rotation=45)

    t2.plot(ax=ax2, color="r")
    ax2.set_ylabel("Consumption rate mean", size=20)
    ax2.set_title("Rate of Consumption Trend", size=23)
    ax2.tick_params(axis="x", labelsize=16, rotation=45)

    plt.tight_layout()
    if save:
        _save("consumption_time_series.png", EDA_FIGURES_DIR)
    plt.show()


def plot_rolling_mean(df: pd.DataFrame, window: int = 12, save: bool = True) -> None:
    """Plot a rolling mean of CONSUMPTION RATE."""
    df["CONSUMPTION RATE"].rolling(window).mean().plot(figsize=(20, 10), linewidth=5, fontsize=20)
    plt.title("Rolling Mean – Consumption Rate", size=22)
    plt.tight_layout()
    if save:
        _save("rolling_mean_consumption_rate.png", EDA_FIGURES_DIR)
    plt.show()


# ── Scatter / pivot plots ─────────────────────────────────────────────────────

def plot_consumption_vs_rate(
    df: pd.DataFrame,
    goodrate_only: bool = True,
    save: bool = True,
) -> None:
    """Plot mean CONSUMPTION HIS at each CONSUMPTION RATE."""
    data = df
    if goodrate_only:
        wrong_idx = df[(df["CONSUMPTION RATE"] > 1.75) & (df["CONSUMPTION RATE"] < 2.75)].index
        data = df.drop(wrong_idx, axis=0)

    data.pivot_table(TARGET_COLUMN, index="CONSUMPTION RATE", aggfunc="mean").plot(
        figsize=(11, 8)
    )
    plt.ylabel("Consumption HIS mean", size=23)
    plt.xlabel("Consumption rate", size=23)
    plt.title("Consumption at each rate", size=25)
    plt.xticks(fontsize=18, rotation=45)
    plt.tight_layout()
    if save:
        _save("consumption_at_each_rate.png", EDA_FIGURES_DIR)
    plt.show()


def plot_running_time_vs_consumption(df: pd.DataFrame, save: bool = True) -> None:
    """Plot mean CONSUMPTION HIS versus RUNNING TIME."""
    df.pivot_table(TARGET_COLUMN, index="RUNNING TIME", aggfunc="mean").plot(
        figsize=(11, 8)
    )
    plt.ylabel("Consumption HIS mean", size=27)
    plt.xlabel("Running Time (h)", size=27)
    plt.title("Fuel Consumption with Running Time", size=30)
    plt.xticks(fontsize=22, rotation=45)
    plt.tight_layout()
    if save:
        _save("consumption_vs_running_time.png", EDA_FIGURES_DIR)
    plt.show()


# ── Distribution plots ────────────────────────────────────────────────────────

def plot_distribution(
    series: pd.Series,
    xlabel: str,
    title: str = "",
    save: bool = True,
    filename: str = "distribution.png",
    annot_x: float = 250.0,
    annot_y: float = 2.5e-3,
) -> None:
    """
    Side-by-side KDE+norm-fit histogram + probability plot, reproducing the
    notebook style.  Default annotation position (x=250, y=2.5e-3) matches
    the Fuel_per_period distribution in the paper.
    """
    clean = series.dropna()
    mu, sigma = norm.fit(clean)

    fig = plt.figure(figsize=(17, 8))

    ax1 = fig.add_subplot(121)
    sns.distplot(clean, fit=norm, kde=True, color="#fb5ffc", norm_hist=True, ax=ax1)
    ax1.set_title(
        r"$\mu = %0.2f, \sigma = %0.2f$" % (mu, sigma), fontsize=27
    )
    ax1.set_ylabel(r"Frequency Distribution", fontsize=22)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2e"))
    ax1.set_xlabel(xlabel, fontsize=22)
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20, rotation=None)
    ax1.text(
        x=annot_x, y=annot_y,
        s=r"$\rm skew = %0.2f, kurt = %0.2f$" % (clean.skew(), clean.kurt()),
        fontsize=24,
    )

    ax2 = fig.add_subplot(122)
    stats.probplot(clean, plot=ax2, rvalue=True)
    ax2.set_title(r"Probplot of Gaussian Distribution", fontsize=22)
    ax2.set_ylabel(r"Ordered Values", fontsize=25)
    ax2.set_xlabel(r"Quantiles", fontsize=22)
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20, rotation=None)
    # enlarge the R² text produced by probplot (index 2, with safe fallback)
    try:
        ax2.get_children()[2].set_fontsize(26.0)
    except (IndexError, AttributeError):
        for child in ax2.get_children():
            if hasattr(child, "get_fontsize") and child.get_fontsize() < 20:
                child.set_fontsize(26.0)

    if title:
        fig.suptitle(title, size=24)

    plt.tight_layout()
    if save:
        _save(filename, EDA_FIGURES_DIR)
    plt.show()


# ── Box plots ─────────────────────────────────────────────────────────────────

def plot_boxplot_by_cluster(df: pd.DataFrame, save: bool = True) -> None:
    """Box plot of CONSUMPTION HIS broken down by Cluster."""
    plt.figure(figsize=(22, 14))
    sns.boxplot(y=TARGET_COLUMN, x="Cluster", data=df)
    plt.xticks(fontsize=26, rotation=80)
    plt.yticks(fontsize=28)
    plt.title("Box plot of CONSUMPTION HIS", size=40)
    plt.ylabel("CONSUMPTION HIS", size=35)
    plt.xlabel("Cluster", size=35)
    plt.tight_layout()
    if save:
        _save("boxplot_consumption_by_cluster.png", EDA_FIGURES_DIR)
    plt.show()


# ── Correlation matrix ────────────────────────────────────────────────────────

def plot_correlation_matrix(num_df: pd.DataFrame, save: bool = True) -> None:
    """Annotated heat-map of Pearson correlations for *num_df*."""
    sns.set(font_scale=1.4)
    corr = num_df.corr()
    plt.figure(figsize=(14, 14))
    sns.heatmap(corr, vmax=1, square=True, linewidths=2, annot=True)
    plt.title("Correlation Matrix", fontsize=22)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.tight_layout()
    if save:
        _save("correlation_matrix.png", EDA_FIGURES_DIR)
    plt.show()


# ── Scatter / pair plot ───────────────────────────────────────────────────────

def plot_scatter_matrix(df: pd.DataFrame, save: bool = True) -> None:
    """Pair plot for the key numeric columns."""
    cols = [TARGET_COLUMN, "RUNNING TIME", "CONSUMPTION RATE", "Fuel_per_period"]
    available = [c for c in cols if c in df.columns]
    sns.set(style="ticks")
    sns.pairplot(df[available])
    plt.tight_layout()
    if save:
        _save("scatter_matrix.png", EDA_FIGURES_DIR)
    plt.show()


# ── Prediction vs Observed ────────────────────────────────────────────────────

def plot_pred_vs_obs(
    pred_df: pd.DataFrame,
    model_name: str = "Model",
    save: bool = True,
) -> None:
    """Line plot of Predicted vs Observed fuel consumption."""
    plt.figure(figsize=(14, 7))
    plt.plot(pred_df["Observed"].values, label="Observed", linewidth=1.5)
    plt.plot(pred_df["Predicted"].values, label="Predicted", linewidth=1.5, alpha=0.8)
    plt.ylabel("Fuel Consumption", size=20)
    plt.xlabel("Observations", size=20)
    plt.title(f"Consumption – Predicted vs Observed ({model_name})", size=20)
    plt.xticks(fontsize=14, rotation=45)
    plt.legend(fontsize=14)
    plt.tight_layout()
    if save:
        _save(f"pred_vs_obs_{model_name}.png")
    plt.show()


# ── Cluster bar chart ─────────────────────────────────────────────────────────

def plot_cluster_bar(
    cluster_df: pd.DataFrame,
    model_name: str = "Model",
    save: bool = True,
) -> None:
    """Grouped bar chart comparing observed vs predicted consumption per cluster."""
    n = len(cluster_df)
    index = np.arange(n)
    bar_width = 0.40

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.bar(index, cluster_df["Observed"], bar_width, alpha=0.9, color="chocolate", label="Observed")
    ax.bar(index + bar_width, cluster_df["Predicted"], bar_width, alpha=0.9, color="darkblue", label="Predicted")

    ax.set_xlabel("Cluster", fontsize=18)
    ax.set_ylabel("Fuel Consumption", fontsize=18)
    ax.set_title(f"Fuel Consumption – Predicted vs Observed by Cluster ({model_name})", size=18)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(cluster_df.index, rotation=60, fontsize=12)
    ax.legend(fontsize=14)
    plt.tight_layout()
    if save:
        _save(f"cluster_bar_{model_name}.png", TRAINING_FIGURES_DIR)
    plt.show()


# ── Model comparison box plots ────────────────────────────────────────────────

def plot_nse_boxplot(nse_table: pd.DataFrame, save: bool = True) -> None:
    """Box plot comparing Nash-Sutcliffe Efficiency distributions across models."""
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(y="NSE", x="Models", data=nse_table, ax=ax)
    ax.set_ylabel("Nash Efficiency Scores", fontsize=25)
    ax.set_xlabel("Models", fontsize=25)
    ax.tick_params(labelsize=23)
    fig.suptitle("Nash Efficiency – Model Comparison", size=28)
    plt.tight_layout()
    if save:
        _save("nse_boxplot.png", TRAINING_FIGURES_DIR)
    plt.show()


def plot_r2_boxplot(score_table: pd.DataFrame, save: bool = True) -> None:
    """Box plot comparing R² scores across models."""
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(y="R Squared Scores", x="Models", data=score_table, ax=ax)
    ax.set_ylabel("R² Scores", fontsize=23)
    ax.set_xlabel("Models", fontsize=23)
    ax.tick_params(labelsize=22)
    fig.suptitle("R² Score – Algorithm Comparison", size=25)
    plt.tight_layout()
    if save:
        _save("r2_boxplot.png", TRAINING_FIGURES_DIR)
    plt.show()


# ── Feature importance (Extra-Trees ranking) ─────────────────────────────────

def plot_feature_importance(
    features: pd.DataFrame,
    save: bool = True,
) -> None:
    """
    Rank feature importances using ExtraTreesRegressor (top-19 features).
    Uses the exact code from the original notebook.
    Saved to EDA-figures/feature_importance.png.
    """
    from sklearn.ensemble import ExtraTreesRegressor

    Y = features[TARGET_COLUMN]
    XX = features.drop(columns=[TARGET_COLUMN])

    clf = ExtraTreesRegressor()
    clf.fit(XX, Y)

    imp_feat, names0 = zip(*sorted(zip(clf.feature_importances_, XX.columns)))
    topN = 19
    imp_feat, names = imp_feat[len(names0) - topN:], names0[len(names0) - topN:]

    plt.figure(figsize=(20, 15))
    plt.barh(range(len(names)), imp_feat)
    plt.xticks(size=30)
    plt.yticks(range(len(names)), names, size=27)
    plt.xlabel("Important features", size=33)
    plt.ylabel("Features", size=30)
    plt.tight_layout()
    if save:
        _save("feature_importance.png", EDA_FIGURES_DIR)
    plt.show()


# ── Learning curve ────────────────────────────────────────────────────────────

def plot_learning_curve(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Model",
    cv: int = 5,
    save: bool = True,
    train_sizes=None,
) -> None:
    """
    Plot training and cross-validation error as a function of training-set
    size (sklearn ``learning_curve``).
    Default train_sizes match the original notebook: np.linspace(0.25, 0.9, 10).
    Pass np.linspace(0.20, 0.9, 10) for Lasso to match its original notebook cell.
    """
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = np.linspace(0.25, 0.9, 10)

    # Pass numpy arrays to avoid pandas-index bugs in sklearn 0.21 / pandas 0.25
    Xv = X.values if hasattr(X, 'values') else X
    yv = y.values if hasattr(y, 'values') else y
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, Xv, yv, cv=cv, scoring="neg_mean_squared_error",
        train_sizes=train_sizes, n_jobs=1,
    )

    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    val_rmse = np.sqrt(-val_scores.mean(axis=1))
    train_std = np.sqrt(train_scores.std(axis=1))
    val_std = np.sqrt(val_scores.std(axis=1))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(train_sizes_abs, train_rmse, "o-", color="royalblue", label="Training RMSE")
    ax.plot(train_sizes_abs, val_rmse, "o-", color="tomato", label="Validation RMSE")
    ax.fill_between(train_sizes_abs, train_rmse - train_std, train_rmse + train_std,
                    alpha=0.15, color="royalblue")
    ax.fill_between(train_sizes_abs, val_rmse - val_std, val_rmse + val_std,
                    alpha=0.15, color="tomato")
    ax.set_xlabel("Training samples", fontsize=18)
    ax.set_ylabel("RMSE", fontsize=18)
    ax.set_title(f"Learning Curve – {model_name}", fontsize=20)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    if save:
        _save(f"learning_curve_{model_name}.png", TRAINING_FIGURES_DIR)
    plt.show()


# ── Predicted vs Observed + Prediction Error (yellowbrick) ───────────────────

def plot_pred_vs_obs_and_error(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    save: bool = True,
) -> None:
    """
    Two-panel figure (notebook style):
      Left  – line plot of Predicted vs Observed.
      Right – yellowbrick PredictionError scatter.
    Saved as pred_vs_obs_{model_name}.png
    """
    from yellowbrick.regressor import PredictionError

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))

    # Left: line overlay (notebook: plt.plot)
    ax1.plot(y_test.values, label="Observed", linewidth=1.5)
    ax1.plot(y_pred, label="Predicted", linewidth=1.5, alpha=0.8)
    ax1.set_ylabel("Consumption", size=20)
    ax1.set_xlabel("Observations", size=20)
    ax1.set_title(f"Consumption Predicted and Observed", size=20)
    ax1.tick_params(axis="x", labelsize=14, rotation=45)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.legend(fontsize=13)

    # Right: yellowbrick PredictionError scatter
    viz = PredictionError(model, ax=ax2)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    ax2.set_ylabel("Predicted", fontsize=20)
    ax2.set_xlabel("Actual", fontsize=20)
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["legend.fontsize"] = 20
    ax2.tick_params(axis="x", labelsize=14, rotation=45)
    ax2.tick_params(axis="y", labelsize=14)

    plt.tight_layout()
    if save:
        _save(f"pred_vs_obs_{model_name}.png", TRAINING_FIGURES_DIR)
    plt.show()


# ── Residual plot (yellowbrick ResidualsPlot) ─────────────────────────────────

def plot_residuals(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    save: bool = True,
) -> None:
    """
    Yellowbrick ResidualsPlot – residuals vs predicted, exactly as in the
    original notebook.  Saved as residuals_{model_name}.png.
    """
    from yellowbrick.regressor import ResidualsPlot

    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["axes.labelsize"] = 23
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["axes.titlesize"] = 25
    plt.rcParams["legend.fontsize"] = 20

    visualizer = ResidualsPlot(model)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)

    for label in visualizer.ax.texts:
        label.set_size(40)
    plt.title(f"Residuals for {model_name} model")
    if save:
        _save(f"residuals_{model_name}.png", TRAINING_FIGURES_DIR)
    visualizer.poof()


# ── Entry point ───────────────────────────────────────────────────────────────

# Registry of available figure names for help text and validation
_PRE_FIGURES = ["distribution", "rate", "running_time", "correlation", "boxplot", "scatter", "feature_importance"]
_POST_FIGURES = ["pred_vs_obs", "residuals", "learning_curve", "nse_boxplot", "r2_boxplot"]


if __name__ == "__main__":
    import argparse
    import pickle
    import sys
    import textwrap
    from pathlib import Path

    # Allow running as  python -m src.visualization.visualize  from project root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import pandas as pd
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    from config import (
        MODELS_DIR,
        PROCESSED_DATA_FILE,
        SELECTED_FEATURES,
        TARGET_COLUMN,
        TEST_SIZE,
        RANDOM_STATE,
    )

    # ── Argument parsing ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="python -m src.visualization.visualize",
        description="Generate fuel-consumption figures on demand.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            Pre-training figures  : {', '.join(_PRE_FIGURES)}
            Post-training figures : {', '.join(_POST_FIGURES)}

            Examples
            --------
              python -m src.visualization.visualize pre all
              python -m src.visualization.visualize pre distribution
              python -m src.visualization.visualize pre correlation
              python -m src.visualization.visualize post all
              python -m src.visualization.visualize post nse_boxplot
              python -m src.visualization.visualize post learning_curve
        """),
    )
    parser.add_argument(
        "stage",
        choices=["pre", "post"],
        help="'pre' for pre-training figures, 'post' for post-training figures.",
    )
    parser.add_argument(
        "figure",
        nargs="?",
        default="all",
        help="Figure name or 'all' (default: all).",
    )
    args = parser.parse_args()

    # Validate figure name early
    valid = _PRE_FIGURES if args.stage == "pre" else _POST_FIGURES
    if args.figure != "all" and args.figure not in valid:
        parser.error(
            f"Unknown figure '{args.figure}' for stage '{args.stage}'.\n"
            f"Choose from: {', '.join(valid)}  or  'all'."
        )

    figures_to_run = valid if args.figure == "all" else [args.figure]

    # ── Load processed data (always needed) ──────────────────────────────────
    df = pd.read_csv(PROCESSED_DATA_FILE)
    print(f"Loaded processed data: {df.shape[0]} rows × {df.shape[1]} columns.")

    feature_cols = [c for c in SELECTED_FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    # ══════════════════════════════════════════════════════════════════════════
    # PRE-TRAINING FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    if args.stage == "pre":
        print(f"\n── Pre-training figures: {args.figure} ──")

        for fig in figures_to_run:
            print(f"  → {fig}")
            if fig == "distribution":
                plot_distribution(
                    df["Fuel_per_period"],
                    xlabel="Fuel per period",
                    title="Distribution of Fuel per period",
                    filename="distribution_Fuel_per_period.png",
                    annot_x=250.0,
                    annot_y=2.5e-3,
                )
            elif fig == "rate":
                plot_consumption_vs_rate(df, goodrate_only=True)
            elif fig == "running_time":
                plot_running_time_vs_consumption(df)
            elif fig == "correlation":
                num_df = df.select_dtypes(include=[np.number])
                plot_correlation_matrix(num_df)
            elif fig == "boxplot":
                if "Cluster" in df.columns:
                    plot_boxplot_by_cluster(df)
                else:
                    print("    'Cluster' column not found, skipping boxplot.")
            elif fig == "scatter":
                plot_scatter_matrix(df)
            elif fig == "feature_importance":
                from src.features.build_features import build_features as _build_feat
                _, _features, _ = _build_feat(df)
                plot_feature_importance(_features)

    # ══════════════════════════════════════════════════════════════════════════
    # POST-TRAINING FIGURES  (requires fitted model .pkl files in models/)
    # ══════════════════════════════════════════════════════════════════════════
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        scaler = StandardScaler()
        X_train_sc = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_sc = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # Names match what train_model.py saves: RF.pkl, GB.pkl, MLP.pkl, Lasso.pkl
        model_files = {
            "RF":    MODELS_DIR / "RF.pkl",
            "GB":    MODELS_DIR / "GB.pkl",
            "MLP":   MODELS_DIR / "MLP.pkl",
            "Lasso": MODELS_DIR / "Lasso.pkl",
        }

        # nse_boxplot / r2_boxplot are cross-model comparisons collected here
        nse_rows: list = []
        r2_rows: list = []
        need_folds = any(f in figures_to_run for f in ("nse_boxplot", "r2_boxplot"))

        for model_name, model_path in model_files.items():
            if not model_path.exists():
                print(f"\nModel file not found, skipping: {model_name}")
                continue

            with open(model_path, "rb") as fh:
                model = pickle.load(fh)

            uses_scaling = model_name in ("MLP", "Lasso")
            Xtr = X_train_sc if uses_scaling else X_train
            Xte = X_test_sc if uses_scaling else X_test

            print(f"\n── Post-training figures [{model_name}]: {args.figure} ──")

            if "pred_vs_obs" in figures_to_run:
                print("  → pred_vs_obs")
                plot_pred_vs_obs_and_error(model, Xtr, y_train, Xte, y_test, model_name)

            if "residuals" in figures_to_run:
                print("  → residuals")
                plot_residuals(model, Xtr, y_train, Xte, y_test, model_name)

            if "learning_curve" in figures_to_run:
                print("  → learning_curve")
                lc_sizes = (
                    np.linspace(0.20, 0.9, 10) if model_name == "Lasso"
                    else np.linspace(0.25, 0.9, 10)
                )
                plot_learning_curve(model, Xtr, y_train, model_name, cv=5,
                                    train_sizes=lc_sizes)

            if need_folds:
                # Always split from the full X so indices are always in-bounds.
                # For scaled models, fit the scaler on each training fold.
                kf = KFold(n_splits=10)
                for tr_idx, te_idx in kf.split(X):
                    Xf_raw_tr = X.iloc[tr_idx]
                    Xf_raw_te = X.iloc[te_idx]
                    yf_tr = y.values[tr_idx]
                    yf_te = y.values[te_idx]
                    if uses_scaling:
                        _fold_sc = StandardScaler()
                        Xf_tr = _fold_sc.fit_transform(Xf_raw_tr)
                        Xf_te = _fold_sc.transform(Xf_raw_te)
                    else:
                        Xf_tr = Xf_raw_tr.values
                        Xf_te = Xf_raw_te.values
                    model.fit(Xf_tr, yf_tr)
                    preds = model.predict(Xf_te)
                    ss_res = ((yf_te - preds) ** 2).sum()
                    ss_tot = ((yf_te - yf_te.mean()) ** 2).sum()
                    nse_rows.append({"Models": model_name, "NSE": 1 - ss_res / ss_tot})
                    r2_rows.append({"Models": model_name, "R Squared Scores": r2_score(yf_te, preds)})

        # Cross-model comparison plots (rendered once, after all models)
        if "nse_boxplot" in figures_to_run:
            print("\n  → nse_boxplot")
            if nse_rows:
                plot_nse_boxplot(pd.DataFrame(nse_rows))
            else:
                print("    No model files found – cannot draw NSE boxplot.")

        if "r2_boxplot" in figures_to_run:
            print("\n  → r2_boxplot")
            if r2_rows:
                plot_r2_boxplot(pd.DataFrame(r2_rows))
            else:
                print("    No model files found – cannot draw R² boxplot.")

    print(f"\nDone. Figures saved to '{FIGURES_DIR}'.")
