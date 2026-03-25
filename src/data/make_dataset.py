"""
src/data/make_dataset.py
------------------------
Data loading, cleaning, and feature-engineering pipeline extracted from the
original Fuel Consumption Jupyter Notebook.

Public API
----------
    load_raw_data(filepath, sheet_name) -> pd.DataFrame
    missing_values_table(df)            -> pd.DataFrame
    drop_high_missing_columns(df, threshold) -> pd.DataFrame
    normalize_cluster_names(df)         -> pd.DataFrame
    engineer_running_time(df)           -> pd.DataFrame
    engineer_fuel_features(df)          -> pd.DataFrame
    remove_abnormal_observations(df)    -> pd.DataFrame
    remove_outliers(df)                 -> pd.DataFrame
    fill_missing_values(df)             -> pd.DataFrame
    run_pipeline(filepath, sheet_name)  -> pd.DataFrame
"""

import time
import warnings

import numpy as np
import pandas as pd

from config import (
    CLUSTERS_TO_DROP,
    CLUSTER_NAME_MAP,
    COLUMNS_TO_DROP,
    MAX_CONSUMPTION_HIS,
    MAX_DAILY_RUNNING_HOURS,
    MAX_FUEL_PER_PERIOD,
    MAX_RUNNING_TIME,
    MIN_CONSUMPTION_HIS,
    MIN_FUEL_PER_PERIOD_LOWER,
    MISSING_VALUE_THRESHOLD,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    SHEET_NAME,
)

warnings.filterwarnings("ignore")


# ── 1. Loading ────────────────────────────────────────────────────────────────

def load_raw_data(filepath=RAW_DATA_FILE, sheet_name=SHEET_NAME) -> pd.DataFrame:
    """Read the raw Excel file and return a DataFrame."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns from '{filepath}'.")
    return df


# ── 2. Missing‑value analysis ─────────────────────────────────────────────────

def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising the number and percentage of missing values
    per column (only columns that have at least one missing value).
    """
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)
    table = pd.concat([mis_val, mis_val_percent], axis=1).rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )
    table = (
        table[table.iloc[:, 1] != 0]
        .sort_values("% of Total Values", ascending=False)
        .round(1)
    )
    print(
        f"Your selected dataframe has {df.shape[1]} columns.\n"
        f"There are {table.shape[0]} columns that have missing values."
    )
    return table


def drop_high_missing_columns(
    df: pd.DataFrame, threshold: float = MISSING_VALUE_THRESHOLD
) -> pd.DataFrame:
    """Drop columns whose missing-value percentage exceeds *threshold*."""
    missing_df = missing_values_table(df)
    cols_to_drop = list(missing_df[missing_df["% of Total Values"] > threshold].index)
    print(f"Removing {len(cols_to_drop)} columns with >{threshold}% missing values.")
    return df.drop(columns=cols_to_drop)


# ── 3. Cluster name normalisation ─────────────────────────────────────────────

def normalize_cluster_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise cluster names and remove irrelevant clusters."""
    df = df.copy()
    df = df.replace(CLUSTER_NAME_MAP)
    for cluster in CLUSTERS_TO_DROP:
        idx = df[df["Cluster"] == cluster].index
        df = df.drop(idx, axis=0)
        print(f"Removed cluster '{cluster}' ({len(idx)} rows).")
    return df


# ── 4. Running‑time feature engineering ──────────────────────────────────────

def engineer_running_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ``Normal_running_time``, capping per-day running time at 24 hours.
    Rows whose normalised running time exceeds the cap are collected and later
    removed by :func:`remove_abnormal_observations`.

    Implementation note
    -------------------
    The original notebook initialised ``Normal_running_time`` as an *int64*
    column (``Clusters["Normal_running_time"] = 0``) and used chained-label
    indexing (``NRT = Clusters["Normal_running_time"]; NRT[x] = value``).
    That causes two interacting effects in pandas 0.25.3:

    1. Float → int truncation for every assignment while the column is still
       int64 (e.g. ``24.7 h/day`` becomes ``24``, so ``24 > 24 == False`` →
       treated as normal).
    2. When a NaN ``RUNNING TIME`` row is encountered, storing ``NaN`` into the
       int64 column via chained indexing triggers an automatic upcast to
       float64; all *subsequent* comparisons use true float values.

    Replicating these effects exactly—rather than "cleaning them up"—is the
    only way to reproduce the paper's reported values
    (``extra_running_time.sum() = 10496.0 h`` and
    ``Total_extra_HIS = 19814.89 L``).  The chained-indexing pattern below is
    an intentional reproduction of that behaviour; the SettingWithCopyWarning
    it raises in pandas 0.25.3 is expected and harmless here.
    """
    df = df.copy()
    start = time.time()
    abnormal_idx = []

    # int64 initial value is critical: it triggers truncation + NaN-upcast
    # behaviour that matches the original notebook exactly.
    df["Normal_running_time"] = 0  # int64
    NRT = df["Normal_running_time"]  # Series reference – chained write (intentional)

    for x in df.index:
        if df["NUMBER OF DAYS"][x] != 0:  # NaN DAYS → NaN != 0 == True
            NRT[x] = df["RUNNING TIME"][x] / df["NUMBER OF DAYS"][x]
            if NRT[x] > MAX_DAILY_RUNNING_HOURS:
                abnormal_idx.append(x)
                NRT[x] = MAX_DAILY_RUNNING_HOURS * df["NUMBER OF DAYS"][x]
            else:
                NRT[x] = df["RUNNING TIME"][x]

    df["extra_running_time"] = df["RUNNING TIME"] - df["Normal_running_time"]
    total_extra = df["extra_running_time"].sum()
    print(
        f"Engineer running time done in {(time.time() - start) / 60:.2f} min. "
        f"Extra running time: {total_extra:.2f} h across {len(abnormal_idx)} rows."
    )
    # Return abnormal_idx alongside df (replaces the removed df.attrs approach)
    return df, abnormal_idx


# ── 5. Fuel feature engineering ───────────────────────────────────────────────

def engineer_fuel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive fuel-related columns:
      * ``extra_cons_HIS``   – extra consumption from abnormal running time
      * ``Normal_cons_HIS``  – consumption corrected for abnormal running time
      * ``Fuel_per_period``  – fuel difference between visits (proxy for actual use)
    """
    df = df.copy()
    df["extra_cons_HIS"] = df["extra_running_time"] * df["CONSUMPTION RATE"]
    df["Normal_cons_HIS"] = df["CONSUMPTION RATE"] * df["Normal_running_time"]
    df["Fuel_per_period"] = df["PREVIOUS FUEL QTE"] - df["QTE FUEL FOUND"]

    total_extra_his = df["extra_cons_HIS"].sum()
    print(f"Total extra CONSUMPTION HIS from abnormal running time: {total_extra_his:.2f} L")

    # Detect rows where generator was off but fuel disappeared (possible theft)
    gen_off = df[df["RUNNING TIME"] == 0].copy()
    fuel_steal = gen_off[gen_off["Fuel_per_period"] != 0]["Fuel_per_period"].sum()
    print(f"Fuel unaccounted for (generator off, fuel missing): {fuel_steal:.2f} L")
    print(f"Total anomalous fuel: {total_extra_his + fuel_steal:.2f} L")

    return df


# ── 6. Remove abnormal observations ──────────────────────────────────────────

def remove_abnormal_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that correspond to:
      1. Abnormal running time (>24 h/day) captured in ``df.attrs``
      2. Generator-off periods where fuel nonetheless disappeared
    """
    df = df.copy()

    # Now abnormal_idx must be passed as an argument
    # (function signature will be updated below)
    raise NotImplementedError("remove_abnormal_observations must be updated to accept abnormal_idx as argument.")

def remove_abnormal_observations(df: pd.DataFrame, abnormal_idx) -> pd.DataFrame:
        """
        Drop rows that correspond to:
            1. Abnormal running time (>24 h/day) captured in abnormal_idx
            2. Generator-off periods where fuel nonetheless disappeared

        Parameters
        ----------
        df : pd.DataFrame
                Input DataFrame
        abnormal_idx : list
                List of row indices with abnormal running time (from engineer_running_time)
        """
        df = df.copy()

        # 1. Rows with abnormal running time
        df = df.drop(abnormal_idx, axis=0, errors="ignore")
        print(f"Dropped {len(abnormal_idx)} rows with abnormal running time.")

        # 2. Generator-off rows with non-zero fuel change
        gen_off = df[df["RUNNING TIME"] == 0]
        theft_idx = gen_off[gen_off["Fuel_per_period"] != 0].index.tolist()
        df = df.drop(theft_idx, axis=0, errors="ignore")
        print(f"Dropped {len(theft_idx)} rows where generator was off but fuel disappeared.")

        return df

# ── 7. Outlier removal ────────────────────────────────────────────────────────

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-knowledge thresholds to remove outlier rows.

    Rules (from notebook analysis):
      * CONSUMPTION HIS  > 1000 or < 60     → remove
      * Fuel_per_period  > 900 or < 0 or 0–60 → remove
      * RUNNING TIME     > 450               → remove
      * Negative RUNNING TIME                → remove
    """
    df = df.copy()
    n_before = len(df)

    # CONSUMPTION HIS outliers
    df = df[df["CONSUMPTION HIS"] <= MAX_CONSUMPTION_HIS]
    df = df[df["CONSUMPTION HIS"] >= MIN_CONSUMPTION_HIS]

    # Fuel per period outliers
    df = df[df["Fuel_per_period"] <= MAX_FUEL_PER_PERIOD]
    df = df[df["Fuel_per_period"] >= 0]
    df = df[df["Fuel_per_period"] >= MIN_FUEL_PER_PERIOD_LOWER]

    # Running time outliers
    df = df[df["RUNNING TIME"] <= MAX_RUNNING_TIME]
    df = df[df["RUNNING TIME"] >= 0]

    print(f"Outlier removal: {n_before} → {len(df)} rows ({n_before - len(df)} removed).")
    return df


# ── 8. Fill remaining missing values ─────────────────────────────────────────

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining NaN values with column means (numeric columns only)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    print("Filled remaining NaN values with column means.")
    return df


# ── 9. Drop non-modelling columns ─────────────────────────────────────────────

def drop_non_modelling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier, redundant, and leaky columns defined in config."""
    cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols)
    print(f"Dropped {len(cols)} non-modelling columns.")
    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    filepath=RAW_DATA_FILE,
    sheet_name=SHEET_NAME,
    save: bool = True,
) -> pd.DataFrame:
    """
    Execute the complete data-preparation pipeline and return the clean
    DataFrame ready for feature engineering.

    Parameters
    ----------
    filepath   : path to the raw Excel file
    sheet_name : worksheet name
    save       : if True, persist the processed data to ``PROCESSED_DATA_FILE``
    """
    df = load_raw_data(filepath, sheet_name)
    df = drop_high_missing_columns(df)
    # Cluster normalisation is intentionally placed AFTER the running-time and
    # fuel-feature steps to match the original notebook order (cell 39 runs
    # before cell 62).  Moving it earlier would exclude BONAPRISO/AGIP/Kotto
    # rows from the intermediate diagnostic sums printed by engineer_*.
    df, abnormal_idx = engineer_running_time(df)
    df = engineer_fuel_features(df)
    df = remove_abnormal_observations(df, abnormal_idx)
    df = normalize_cluster_names(df)
    df = remove_outliers(df)
    df = fill_missing_values(df)
    df = drop_non_modelling_columns(df)

    if save:
        PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_FILE, index=False)
        print(f"Processed data saved to '{PROCESSED_DATA_FILE}'.")

    print(f"\nFinal clean dataset: {df.shape[0]} rows × {df.shape[1]} columns.")
    return df


if __name__ == "__main__":
    run_pipeline()
