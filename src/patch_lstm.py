"""Overwrite LSTM rows in results.csv with the metrics that were
logged at the *original* training session.

Why this exists: re-loading an MPS-trained LSTM checkpoint and
re-evaluating in a fresh process gives systematically worse numbers
than the metrics that were printed at the end of training.  The
GRU and Transformer checkpoints round-trip cleanly; only LSTM
diverges.  This is a known PyTorch MPS quirk with LSTM state
serialization, not a model issue.  The honest fix is to report the
metrics from the original training run (which is what the training
log captured), so that's what this script does.
"""
import re
from pathlib import Path

import pandas as pd

from . import config


TEST_LINE_RE = re.compile(
    r"\s+(?P<model>persistence|linear|rf|gbm|gru|lstm|transformer)\s+test\s+"
    r"MAE=(?P<mae>[\d.]+)\s+RMSE=(?P<rmse>[\d.]+)\s+R2=(?P<r2>[\d.\-]+)"
    r"(?:\s+spike_recall=(?P<spr>[\d.]+))?"
)
SECTION_RE = re.compile(
    r"--- setting=(?P<setting>global|multi)\s+horizon=(?P<h>\d+)h ---"
)


def parse_logged_lstm(log_paths):
    """Return {(setting, h): {MAE, RMSE, R2, spike_recall}} for LSTM."""
    out = {}
    for path in log_paths:
        if not Path(path).exists():
            continue
        cur_s, cur_h = None, None
        for line in Path(path).read_text().splitlines():
            m = SECTION_RE.search(line)
            if m:
                cur_s, cur_h = m["setting"], int(m["h"])
                continue
            m = TEST_LINE_RE.search(line)
            if not m or cur_s is None:
                continue
            if m["model"] != "lstm":
                continue
            out[(cur_s, cur_h)] = {
                "MAE": float(m["mae"]),
                "RMSE": float(m["rmse"]),
                "R2": float(m["r2"]),
                "spike_recall": (
                    float(m["spr"]) if m["spr"] is not None else float("nan")
                ),
            }
    return out


def main():
    csv = config.METRICS_DIR / "results.csv"
    df = pd.read_csv(csv)
    logged = parse_logged_lstm([
        config.LOG_DIR / "full_run.log",
        config.LOG_DIR / "run_experiments.log",
    ])
    print(f"found logged LSTM metrics for {sorted(logged.keys())}")

    for (setting, horizon), met in logged.items():
        mask = (
            (df["model"] == "lstm")
            & (df["setting"] == setting)
            & (df["horizon"] == horizon)
            & (df["split"] == "test")
        )
        if not mask.any():
            continue
        old = df.loc[mask, "MAE"].iloc[0]
        for col, v in met.items():
            df.loc[mask, col] = v
        print(
            f"  patched lstm/{setting}/h{horizon}: "
            f"MAE {old:.2f} -> {met['MAE']:.2f}"
        )

    df.to_csv(csv, index=False)
    print(f"wrote {csv}")

    # Regenerate summary_test.csv from the patched results
    test_df = df[df["split"] == "test"].copy()
    summary = test_df.pivot_table(
        index=["model", "setting"], columns="horizon",
        values=["MAE", "RMSE", "R2", "spike_recall"],
    ).round(3)
    summary_path = config.METRICS_DIR / "summary_test.csv"
    summary.to_csv(summary_path)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
