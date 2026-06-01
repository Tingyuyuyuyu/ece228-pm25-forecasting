"""Feature-based baselines: ridge regression, random forest, gradient boosting.

These models can't handle a 3D (N, W, F) tensor directly, so we flatten the
24-hour window into one long (W*F)-dim vector and tack on a one-hot station
id at the end. Same inputs and same targets as the sequence models, just
without the temporal structure.

We subsample the training set for the tree models (RF: 50k rows, GBM:
80k rows) because the full 298k rows make training slow on CPU. The deep
models see everything.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from . import config


def flatten(X, station_id=None, one_hot=True):
    """Turn an (N, W, F) array into (N, W*F), optionally appending a one-hot
    station id."""
    n = X.shape[0]
    flat = X.reshape(n, -1)
    if station_id is None or not one_hot:
        return flat
    onehot = np.zeros((n, config.N_STATIONS), dtype=np.float32)
    onehot[np.arange(n), station_id] = 1.0
    return np.concatenate([flat, onehot], axis=1)


def get_model(name, seed=config.SEED):
    name = name.lower()
    if name in ("linear", "ridge"):
        # Ridge is the linear baseline; plain LinearRegression is unstable
        # on 468+ features so a small ridge penalty keeps it sane.
        return Ridge(alpha=1.0, random_state=seed)
    if name in ("rf", "random_forest"):
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        )
    if name in ("gbm", "gradient_boosting"):
        # sklearn's GBR is single-threaded, so we keep the tree count modest.
        return GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            random_state=seed,
        )
    raise ValueError(f"unknown baseline: {name!r}")


BASELINE_NAMES = ("linear", "rf", "gbm")


def subsample(X, y, max_n, seed=config.SEED):
    """Random subsample for tractable RF / GBM training."""
    if len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]
