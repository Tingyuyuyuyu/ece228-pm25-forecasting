"""Per-station breakdown of test predictions.

Reads results/test_predictions.npz (saved by run_experiments.py) and
writes one row per (model, setting, horizon, station) to
results/metrics/per_station.csv.

Usage:
    python -m src.analyze_predictions
"""
import re

import numpy as np
import pandas as pd

from . import config, evaluate as ev


# Keys in test_predictions.npz look like "gru_multi_h1_pred" etc.
# Strip the trailing suffix, then parse "<model>_<setting>_h<horizon>".
KEY_RE = re.compile(r"^(?P<model>[^_]+)_(?P<setting>[^_]+)_h(?P<h>\d+)$")


def main():
    npz_path = config.RESULTS_DIR / "test_predictions.npz"
    if not npz_path.exists():
        raise SystemExit(f"missing {npz_path}; run src.run_experiments first")

    z = np.load(npz_path, allow_pickle=True)
    # The npz has entries like <key>_pred, <key>_true, <key>_ts, <key>_sid.
    # Recover the unique <key>s.
    keys = sorted({k.rsplit("_", 1)[0] for k in z.files})

    rows = []
    for key in keys:
        m = KEY_RE.match(key)
        if m is None:
            continue
        model = m["model"]
        setting = m["setting"]
        horizon = int(m["h"])

        preds = z[f"{key}_pred"]
        ys    = z[f"{key}_true"]
        sid   = z[f"{key}_sid"]

        for s in range(config.N_STATIONS):
            mask = sid == s
            if not mask.any():
                continue
            metrics = ev.regression_metrics(ys[mask], preds[mask])
            rows.append({
                "model": model,
                "setting": setting,
                "horizon": horizon,
                "station": config.STATIONS[s],
                **metrics,
            })

    df = pd.DataFrame(rows)
    out_path = config.METRICS_DIR / "per_station.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")

    # Print a quick MAE pivot for sanity-checking
    pivoted = df.pivot_table(
        index=["model", "setting", "horizon"], columns="station",
        values="MAE",
    ).round(2)
    print("\nPer-station test MAE (ug/m^3):")
    print(pivoted.to_string())


if __name__ == "__main__":
    main()
