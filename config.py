"""
Central configuration for the Fuel Consumption ML project.
All hardcoded paths, thresholds, and hyperparameters live here.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_FIGURES_DIR = FIGURES_DIR / "EDA-figures"
TRAINING_FIGURES_DIR = FIGURES_DIR / "training-figures"
EVAL_METRICS_DIR = REPORTS_DIR / "evaluation-metrics"

RAW_DATA_FILE = DATA_RAW_DIR / "Full_Data_Gen_only.xlsx"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "cleaned_generator_data.csv"

# ── Data Loading ─────────────────────────────────────────────────────────────
SHEET_NAME = "Generator_only"

# ── Cleaning Thresholds ───────────────────────────────────────────────────────
MISSING_VALUE_THRESHOLD = 50          # Drop columns with > 50 % missing values
MAX_DAILY_RUNNING_HOURS = 24          # A generator cannot run more than 24 h/day
MAX_RUNNING_TIME = 450                # Hard cap on running time (hours)
MAX_CONSUMPTION_HIS = 1000            # Outlier ceiling for CONSUMPTION HIS
MIN_CONSUMPTION_HIS = 60              # Outlier floor  for CONSUMPTION HIS
MAX_FUEL_PER_PERIOD = 900             # Outlier ceiling for Fuel_per_period
MIN_FUEL_PER_PERIOD_LOWER = 60        # Remove fuel-per-period < 60 (near-zero)

# Cluster name normalisation map  (raw → canonical)
CLUSTER_NAME_MAP = {
    "Ngaoundal": "NGAOUNDAL",
    "Garoua 1": "GAROUA 1",
    "Kousseri": "KOUSSERI",
    "Garoua 2": "GAROUA 2",
    "Kaele": "KAELE",
    "Maroua 1": "MAROUA 1",
    "Maroua 2": "MAROUA 2",
    "Maroua 3": "MAROUA 3",
    "Meiganga": "Meiganga 1",
    "Ngaoundere 1": "NGAOUNDERE 1",
    "Ngaoundere 2": "NGAOUNDERE 2",
    "Tibati": "TIBATI",
    "Rey Bouba": "REY BOUBA",
    "Waza": "WAZA",
    "Yagoua": "YAGOUA",
    "Meingaga 2": "Meiganga 2",
    "Bonapriso": "BONAPRISO",
    "MAKARI": "MAKARY",
    "Guider": "GUIDER",
}

# Clusters to drop entirely (noise / too few samples)
CLUSTERS_TO_DROP = ["BONAPRISO", "AGIP", "Kotto"]

# Columns to drop before modelling (IDs, redundant, leaky)
COLUMNS_TO_DROP = [
    "ACCESS TICKET NUMBER",
    "PREVIOUS DATE OF VISIT",
    "GE N°",
    "SITE Name",
    "DEPARTURE TIME ON THE SITE",
    "TX Indoor / Outdoor",
    "Ph1 (Amps)",
    "extra_cons_HIS",
    "extra_running_time",
    "Normal_cons_HIS",
]

# Categorical columns to one-hot encode
CATEGORICAL_COLUMNS = ["Cluster", "TYPE OF GENERATOR", "GENERATOR 1 CAPACITY (KVA)"]

# Numeric columns to drop from the feature matrix (highly correlated / leaky)
NUMERIC_COLUMNS_TO_DROP = [
    "CURRENT HOUR METER GE1",
    "PREVIOUS HOUR METER G1",
    "Ph3 (Amps)",
]

# ── Feature Selection ─────────────────────────────────────────────────────────
SELECTED_FEATURES = [
    "Fuel_per_period",
    "RUNNING TIME",
    "CONSUMPTION RATE",
    "NUMBER OF DAYS",
    "GENERATOR 1 CAPACITY (KVA)_6,5 x 2",
]
TARGET_COLUMN = "CONSUMPTION HIS"

# ── Model Hyperparameters (best found via RandomizedSearchCV) ─────────────────
RF_PARAMS = {
    "bootstrap": True,
    "max_depth": 60,
    "max_features": "auto",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 890,
    "random_state": 7,
}

GB_PARAMS = {
    "max_depth": 3,
    "min_samples_leaf": 1,
    "n_estimators": 900,
    "loss": "ls",   # sklearn 0.21 name; renamed to 'squared_error' in sklearn >= 1.1
    "alpha": 0.4242914285714286,
    "learning_rate": 0.13655249878370609,
    "random_state": 45,
}

MLP_PARAMS = {
    "activation": "relu",
    "alpha": 0.9697961224489796,
    "early_stopping": False,
    "hidden_layer_sizes": 3,
    "learning_rate": "adaptive",
    "learning_rate_init": 0.015709452018676538,
    "max_iter": 400,
    "random_state": 7,
}

LASSO_PARAMS = {
    "alpha": 1e-05,
    "max_iter": 1500,
    "tol": 1e-05,
}

# ── Training ──────────────────────────────────────────────────────────────────
KFOLD_SPLITS = 10
TEST_SIZE = 0.25
RANDOM_STATE = 42

# RandomizedSearchCV settings
N_ITER_SEARCH = 200
CV_FOLDS = 10
