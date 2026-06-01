"""Load the Beijing Multi-Site Air Quality CSVs and do all the cleaning.

There are 12 station CSVs in data/raw/. This file:
  1. reads all of them
  2. handles the small amount of missing data (forward + backward fill)
  3. encodes wind direction as (sin, cos) of the bearing
  4. adds cyclic hour/day-of-week/month features
  5. splits by time (train / val / test)
  6. standardizes numeric columns using only the training set
  7. clips standardized values to [-8, 8] -- this was important, see note below

Run this file directly to print split sizes and save the scaler stats:
    python -m src.preprocessing
"""
import json

import numpy as np
import pandas as pd

from . import config


# bearing in degrees for each of the 16 compass labels in the dataset
WIND_DIR_DEG = {
    "N":  0.0,   "NNE": 22.5, "NE":  45.0,  "ENE": 67.5,
    "E":  90.0,  "ESE": 112.5,"SE":  135.0, "SSE": 157.5,
    "S":  180.0, "SSW": 202.5,"SW":  225.0, "WSW": 247.5,
    "W":  270.0, "WNW": 292.5,"NW":  315.0, "NNW": 337.5,
}


def load_all_stations():
    """Read every PRSA_Data_*.csv into one big DataFrame."""
    frames = []
    for station in config.STATIONS:
        path = config.DATA_RAW_DIR / f"PRSA_Data_{station}_20130301-20170228.csv"
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)

    # Build a real datetime column from the year/month/day/hour columns
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["No"])
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)
    return df


def encode_wind(df):
    """Turn the 16 compass labels into (sin, cos) of the bearing so 359 and 1
    are close, not opposite."""
    df = df.copy()
    df["wd_deg"] = df["wd"].map(WIND_DIR_DEG)
    df["wd_sin"] = np.sin(np.deg2rad(df["wd_deg"]))
    df["wd_cos"] = np.cos(np.deg2rad(df["wd_deg"]))
    return df


def add_time_features(df):
    """Cyclic encodings for hour-of-day, day-of-week, month-of-year."""
    df = df.copy()
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    dow = df["datetime"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def fill_missing(df):
    """Within each station, forward-fill then backward-fill. Missing values
    are mostly isolated hours so this is fine; anything still missing
    afterwards (very rare) we just set to 0."""
    df = df.copy()
    cols = config.NUMERIC_FEATURES
    df[cols] = (df.groupby("station", group_keys=False)[cols]
                  .apply(lambda g: g.ffill().bfill()))
    df[cols] = df[cols].fillna(0.0)
    return df


def add_station_id(df):
    df = df.copy()
    df["station_id"] = df["station"].map(config.STATION_TO_ID)
    return df


def build_processed():
    """Full preprocessing chain, returns the tidy DataFrame."""
    df = load_all_stations()
    df = encode_wind(df)
    df = add_time_features(df)
    df = fill_missing(df)
    df = add_station_id(df)
    keep_cols = (["station", "station_id", "datetime"]
                 + config.NUMERIC_FEATURES + config.TIME_FEATURES)
    return df[keep_cols]


def split_by_time(df):
    """Train / val / test split based on calendar dates. We never train on
    a hour that's later than a val or test hour."""
    train_end = pd.Timestamp(config.TRAIN_END)
    val_end   = pd.Timestamp(config.VAL_END)
    train = df[df["datetime"] <= train_end]
    val   = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)]
    test  = df[df["datetime"] > val_end]
    return train, val, test


def fit_scaler(train_df):
    """Compute mean/std for each numeric feature using TRAIN data only.
    Time features are already in [-1, 1] so we leave them alone."""
    stats = {}
    for col in config.NUMERIC_FEATURES:
        mu = float(train_df[col].mean())
        sigma = float(train_df[col].std())
        if sigma == 0 or not np.isfinite(sigma):
            sigma = 1.0
        stats[col] = {"mean": mu, "std": sigma}
    for col in config.TIME_FEATURES:
        stats[col] = {"mean": 0.0, "std": 1.0}
    return stats


def apply_scaler(df, stats, clip=8.0):
    """Standardize each feature and clip to [-clip, clip].

    NOTE: the clipping is important! Without it, the rainfall column has
    values up to ~90 sigma (because most hours are 0 mm but downpours are
    a few mm). Those huge inputs make the GRU produce NaN losses on the
    very first batches. Clipping at +/- 8 sigma keeps real outliers
    "large" but bounded, and training is much more stable.
    """
    df = df.copy()
    for col, s in stats.items():
        if col in df.columns:
            z = (df[col] - s["mean"]) / s["std"]
            df[col] = z.clip(lower=-clip, upper=clip)
    return df


def save_scaler(stats, path):
    path.write_text(json.dumps(stats, indent=2))


def load_scaler(path):
    return json.loads(path.read_text())


def prepare():
    """One-call entry: returns (train, val, test, scaler_stats), where all
    three DataFrames are already standardized + clipped."""
    df = build_processed()
    train_raw, val_raw, test_raw = split_by_time(df)
    stats = fit_scaler(train_raw)
    train = apply_scaler(train_raw, stats)
    val   = apply_scaler(val_raw,   stats)
    test  = apply_scaler(test_raw,  stats)
    return train, val, test, stats


if __name__ == "__main__":
    train, val, test, stats = prepare()
    print(f"train rows: {len(train)}")
    print(f"val   rows: {len(val)}")
    print(f"test  rows: {len(test)}")
    print(f"(per station roughly "
          f"train={len(train)//config.N_STATIONS}, "
          f"val={len(val)//config.N_STATIONS}, "
          f"test={len(test)//config.N_STATIONS})")
    save_scaler(stats, config.DATA_PROC_DIR / "scaler.json")
    print(f"saved scaler stats to {config.DATA_PROC_DIR / 'scaler.json'}")
