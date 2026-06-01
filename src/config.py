"""All the paths, constants, and hyperparameter defaults live here.

If you want to change something globally (a path, a hyperparameter, the
list of features) this is the file to edit. The other modules just read
from here.
"""
from pathlib import Path


# Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR  = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
DATA_PROC_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
METRICS_DIR   = RESULTS_DIR / "metrics"
CKPT_DIR      = PROJECT_ROOT / "checkpoints"
LOG_DIR       = PROJECT_ROOT / "logs"

# Make sure the output directories exist (it's cheap and avoids errors later)
for d in (DATA_PROC_DIR, FIGURES_DIR, METRICS_DIR, CKPT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# Dataset constants -----------------------------------------------------
STATIONS = [
    "Aotizhongxin", "Changping", "Dingling",    "Dongsi",
    "Guanyuan",     "Gucheng",   "Huairou",     "Nongzhanguan",
    "Shunyi",       "Tiantan",   "Wanliu",      "Wanshouxigong",
]
N_STATIONS = len(STATIONS)
STATION_TO_ID = {s: i for i, s in enumerate(STATIONS)}

POLLUTANTS      = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEO_NUMERIC   = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
WIND_COMPONENTS = ["wd_sin", "wd_cos"]
TIME_FEATURES   = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                   "month_sin", "month_cos"]

# These are the columns that go into the model
NUMERIC_FEATURES = POLLUTANTS + METEO_NUMERIC + WIND_COMPONENTS
ALL_FEATURES     = NUMERIC_FEATURES + TIME_FEATURES   # length F = 19

TARGET = "PM2.5"


# Train / val / test split (chronological, NOT random) ------------------
# ~4 years of hourly data per station. Picked so each split has at
# least one full season represented.
TRAIN_END = "2015-12-31 23:00:00"
VAL_END   = "2016-06-30 23:00:00"


# Sliding-window setup --------------------------------------------------
WINDOW   = 24                # how many past hours we feed the model
HORIZONS = [1, 6]            # how many hours ahead to predict


# Random seed -----------------------------------------------------------
SEED = 42


# Training hyperparameters ----------------------------------------------
BATCH_SIZE          = 512
LR                  = 1e-3
WEIGHT_DECAY        = 1e-5
MAX_EPOCHS          = 15
EARLY_STOP_PATIENCE = 4
GRAD_CLIP           = 1.0
