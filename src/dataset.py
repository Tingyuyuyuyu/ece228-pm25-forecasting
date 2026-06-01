"""Build the (X, y) examples that the models actually train on.

We use a sliding 24-hour window:
    input  = last 24 hours of features for some station
    target = PM2.5 at that station, `horizon` hours after the window ends

Two input settings:

  - "global"  : the window contains only the target station's own features.
                Each example also carries the station id so a single global
                model can condition on which station it's predicting for.

  - "multi"   : same as global but we also append, at every timestep, the
                channel-wise MEAN of the other 11 stations. That's our
                lightweight "spatial context" -- it doubles the feature
                count from F to 2F. We exclude the target station from
                the mean so there's no label leakage.

The output target is in original ug/m^3 units (not standardized) so MAE /
RMSE are easy to read. The model trains against the standardized target
internally (see train.py) and we de-standardize for metrics.
"""
import numpy as np
import pandas as pd

from . import config


class WindowedDataset:
    """Tiny container for a built dataset. Just holds numpy arrays.

    X           : (N, W, F)   float32
    y           : (N,)        float32, in original ug/m^3
    station_id  : (N,)        int64
    timestamps  : (N,)        numpy datetime64, the timestamp being predicted
    """
    def __init__(self, X, y, station_id, timestamps):
        self.X = X
        self.y = y
        self.station_id = station_id
        self.timestamps = timestamps


def _make_windows(feats, target_norm, ts, window, horizon, t_mean, t_std):
    """Slide a window of length `window` across `feats`.
    Target is the standardized PM2.5 at index t+window+horizon-1.
    Returns numpy arrays, with y de-standardized to ug/m^3.
    """
    n = len(feats)
    n_windows = n - window - horizon + 1
    if n_windows <= 0:
        return None

    # Build all windows at once (stack of 24-hour slices)
    idx = np.arange(n_windows)
    X = np.stack([feats[i:i + window] for i in idx], axis=0)
    target_idx = idx + window + horizon - 1
    y_norm = target_norm[target_idx]
    y = (y_norm * t_std + t_mean).astype(np.float32)
    target_ts = ts[target_idx]
    return X, y, target_ts


def _station_arrays(df, station, feat_cols):
    sub = df[df["station"] == station].sort_values("datetime")
    feats = sub[feat_cols].to_numpy(dtype=np.float32)
    ts = sub["datetime"].to_numpy()
    sid = sub["station_id"].iloc[0]
    return feats, ts, sid


def build_single(df_norm, station, horizon, target_stats, window=config.WINDOW):
    """Build the sliding-window dataset for ONE station (its own features only)."""
    feat_cols = config.ALL_FEATURES
    feats, ts, sid = _station_arrays(df_norm, station, feat_cols)
    pm_idx = feat_cols.index(config.TARGET)
    target_norm = feats[:, pm_idx]
    out = _make_windows(feats, target_norm, ts, window, horizon,
                        target_stats["mean"], target_stats["std"])
    if out is None:
        return None
    X, y, target_ts = out
    return WindowedDataset(
        X=X, y=y,
        station_id=np.full(len(X), sid, dtype=np.int64),
        timestamps=target_ts,
    )


def build_global(df_norm, horizon, target_stats, window=config.WINDOW):
    """Concatenate per-station single-station datasets. One global model
    will train on all 12 of them together."""
    parts = []
    for station in config.STATIONS:
        ds = build_single(df_norm, station, horizon, target_stats, window)
        if ds is not None:
            parts.append(ds)
    return WindowedDataset(
        X=np.concatenate([p.X for p in parts], axis=0),
        y=np.concatenate([p.y for p in parts], axis=0),
        station_id=np.concatenate([p.station_id for p in parts], axis=0),
        timestamps=np.concatenate([p.timestamps for p in parts], axis=0),
    )


def build_multi(df_norm, horizon, target_stats, window=config.WINDOW):
    """Multi-site version.

    At every timestamp we compute the mean over all 12 stations of every
    feature. For each station, we then concatenate
      [own_features, mean_of_OTHER_11_stations]
    at each timestep. Excluding self is important so we don't leak the
    label.
    """
    feat_cols = config.ALL_FEATURES
    pm_idx = feat_cols.index(config.TARGET)

    # Pivot wide: rows=timestamp, columns=(feature, station)
    pivot = df_norm.pivot_table(index="datetime", columns="station",
                                 values=feat_cols)
    pivot = pivot.sort_index()
    # Reindex on a uniform hourly grid so all stations line up cleanly
    full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq="h")
    pivot = pivot.reindex(full_idx).ffill().bfill()

    # shape (T, F, S)
    arr = np.stack([pivot[c].to_numpy(dtype=np.float32) for c in feat_cols], axis=1)
    sum_all = arr.sum(axis=2)         # (T, F)
    n_stations = arr.shape[2]

    all_X, all_y, all_sid, all_ts = [], [], [], []
    for station in config.STATIONS:
        sid = config.STATION_TO_ID[station]
        own = arr[:, :, sid]                                   # (T, F)
        mean_others = (sum_all - own) / (n_stations - 1)       # (T, F)
        feats = np.concatenate([own, mean_others], axis=1)     # (T, 2F)
        target_norm = own[:, pm_idx]
        ts = pivot.index.to_numpy()
        out = _make_windows(feats, target_norm, ts, window, horizon,
                            target_stats["mean"], target_stats["std"])
        if out is None:
            continue
        X, y, target_ts = out
        all_X.append(X)
        all_y.append(y)
        all_sid.append(np.full(len(X), sid, dtype=np.int64))
        all_ts.append(target_ts)

    return WindowedDataset(
        X=np.concatenate(all_X, axis=0),
        y=np.concatenate(all_y, axis=0),
        station_id=np.concatenate(all_sid, axis=0),
        timestamps=np.concatenate(all_ts, axis=0),
    )


def build_for_setting(setting, train_norm, val_norm, test_norm,
                      horizon, target_stats, window=config.WINDOW):
    """Return (train, val, test) WindowedDataset for the chosen setting."""
    if setting == "global":
        return (build_global(train_norm, horizon, target_stats, window),
                build_global(val_norm,   horizon, target_stats, window),
                build_global(test_norm,  horizon, target_stats, window))
    if setting == "multi":
        return (build_multi(train_norm, horizon, target_stats, window),
                build_multi(val_norm,   horizon, target_stats, window),
                build_multi(test_norm,  horizon, target_stats, window))
    raise ValueError(f"unknown setting: {setting!r}")


if __name__ == "__main__":
    from . import preprocessing
    print("Quick sanity check: build the multi-site 1h dataset")
    train, val, test, stats = preprocessing.prepare()
    ds = build_multi(train, horizon=1, target_stats=stats[config.TARGET])
    print(f"X shape: {ds.X.shape}")
    print(f"y range: {ds.y.min():.1f} .. {ds.y.max():.1f}")
    print(f"y mean : {ds.y.mean():.1f}")
