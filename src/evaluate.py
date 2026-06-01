"""Metrics for the PM2.5 forecasting task.

We report:
  MAE     mean absolute error           (lower = better)
  RMSE    root-mean-squared error        (lower = better)
  R2      coefficient of determination   (higher = better)
  MAPE    mean abs percentage error on rows with y >= 10 ug/m^3
  spike-* same metrics computed only on rows where y >= 75 ug/m^3,
          which is the WHO / Chinese GB "unhealthy" threshold.
  spike_recall: fraction of those high-pollution hours where the
          model also predicted >= 75 ug/m^3 -- i.e. would the model
          have flagged the bad hour as bad?
"""
import numpy as np


# 75 ug/m^3 is the WHO + Chinese GB ambient air quality threshold for
# "unhealthy" 24-hour mean PM2.5. We use it as the spike threshold.
SPIKE_THRESHOLD = 75.0


def regression_metrics(y_true, y_pred):
    """Compute a dict of regression metrics + spike metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true

    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Skip very low y for MAPE (divide-by-zero-ish)
    mape_mask = y_true >= 10.0
    if mape_mask.any():
        mape = float(np.mean(np.abs(err[mape_mask]) / y_true[mape_mask]))
    else:
        mape = float("nan")

    spike_mask = y_true >= SPIKE_THRESHOLD
    if spike_mask.any():
        spike_mae    = float(np.mean(np.abs(err[spike_mask])))
        spike_rmse   = float(np.sqrt(np.mean(err[spike_mask] ** 2)))
        spike_recall = float(np.mean(y_pred[spike_mask] >= SPIKE_THRESHOLD))
    else:
        spike_mae    = float("nan")
        spike_rmse   = float("nan")
        spike_recall = float("nan")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "spike_MAE": spike_mae,
        "spike_RMSE": spike_rmse,
        "spike_recall": spike_recall,
        "n": int(len(y_true)),
        "n_spike": int(spike_mask.sum()),
    }


def persistence_baseline(X_window, pm25_norm_idx, target_stats):
    """Predict y_{t+h} = y_t. This is the "do nothing" baseline.

    X_window has shape (N, W, F) where the PM2.5 column is at
    pm25_norm_idx. The last timestep of the window is "y_t" in
    standardized space; we de-standardize it.
    """
    last = X_window[:, -1, pm25_norm_idx]
    return last * target_stats["std"] + target_stats["mean"]
