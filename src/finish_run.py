"""Resume an interrupted run_experiments.py.

Reuses what's already on disk:
  - cached datasets in data/processed/ds_<setting>_h<h>.npz
  - sequence-model checkpoints in checkpoints/<model>_<setting>_h<h>.pt
  - logged test metrics in logs/full_run.log and logs/run_experiments.log

Does the minimum extra work to produce a full results.csv,
test_predictions.npz, and figure set:
  - persistence baseline for all 4 (setting, horizon) combos
  - re-fits ridge baseline for all 4 (needed for timeseries figure)
  - parses RF/GBM metrics from logs for completed combos
  - actually fits RF/GBM only for the multi h=6 combo (the missing one)
  - loads existing sequence checkpoints and evaluates on test (no retrain)
  - trains the 3 missing multi h=6 sequence models from scratch
  - regenerates all figures

Usage:
    python -m src.finish_run
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import (
    config,
    dataset as ds_mod,
    evaluate as ev,
    models_baseline as mb,
    models_dl as mdl,
    preprocessing,
    train as tr,
    utils,
    visualize as viz,
)


SETTINGS = ("global", "multi")
HORIZONS = (1, 6)
BASELINES = ("linear", "rf", "gbm")
SEQUENCE = ("gru", "lstm", "transformer")


def load_cached(setting: str, horizon: int):
    cache = config.DATA_PROC_DIR / f"ds_{setting}_h{horizon}.npz"
    z = np.load(cache, allow_pickle=True)

    def _wrap(prefix):
        return ds_mod.WindowedDataset(
            X=z[f"{prefix}_X"],
            y=z[f"{prefix}_y"],
            station_id=z[f"{prefix}_sid"],
            timestamps=z[f"{prefix}_ts"],
        )

    return _wrap("train"), _wrap("val"), _wrap("test")


# Logged test lines look like:
#   ...     linear test  MAE=10.32 RMSE=19.24 R2=0.954 spike_recall=0.95
TEST_LINE_RE = re.compile(
    r"\s+(?P<model>persistence|linear|rf|gbm|gru|lstm|transformer)\s+test\s+"
    r"MAE=(?P<mae>[\d.]+)\s+RMSE=(?P<rmse>[\d.]+)\s+R2=(?P<r2>[\d.\-]+)"
    r"(?:\s+spike_recall=(?P<spr>[\d.]+))?"
)
SECTION_RE = re.compile(
    r"--- setting=(?P<setting>global|multi)\s+horizon=(?P<h>\d+)h ---"
)


def parse_logged_metrics(log_paths: list[Path]) -> dict:
    """Return {(model, setting, h): metrics-dict} extracted from log files.

    If the same combo appears twice (e.g. the run was restarted), the last
    occurrence wins.
    """
    out: dict[tuple[str, str, int], dict] = {}
    for path in log_paths:
        if not path.exists():
            continue
        cur_setting = None
        cur_h = None
        for line in path.read_text().splitlines():
            m = SECTION_RE.search(line)
            if m:
                cur_setting = m["setting"]
                cur_h = int(m["h"])
                continue
            m = TEST_LINE_RE.search(line)
            if m and cur_setting is not None:
                spr = m["spr"]
                out[(m["model"], cur_setting, cur_h)] = {
                    "MAE": float(m["mae"]),
                    "RMSE": float(m["rmse"]),
                    "R2": float(m["r2"]),
                    "spike_recall": float(spr) if spr is not None else float("nan"),
                }
    return out


def fit_baseline(name, tr_ds, va_ds, te_ds, max_rf_n, max_gbm_n, logger):
    Xtr = mb.flatten(tr_ds.X, tr_ds.station_id)
    Xte = mb.flatten(te_ds.X, te_ds.station_id)
    ytr, yte = tr_ds.y, te_ds.y

    if name == "rf":
        Xtr_s, ytr_s = mb.subsample(Xtr, ytr, max_rf_n)
    elif name == "gbm":
        Xtr_s, ytr_s = mb.subsample(Xtr, ytr, max_gbm_n)
    else:
        Xtr_s, ytr_s = Xtr, ytr

    t0 = time.time()
    model = mb.get_model(name)
    logger.info(f"  fitting {name} on X={Xtr_s.shape}")
    model.fit(Xtr_s, ytr_s)
    pred_test = model.predict(Xte)
    m_test = ev.regression_metrics(yte, pred_test)
    logger.info(
        f"    {name} test  MAE={m_test['MAE']:.2f} RMSE={m_test['RMSE']:.2f} "
        f"R2={m_test['R2']:.3f} spike_recall={m_test['spike_recall']:.2f} "
        f"({time.time() - t0:.0f}s)"
    )
    return pred_test, m_test


def load_or_train_sequence(name, setting, horizon, tr_ds, va_ds, te_ds,
                            target_stats, device, logger):
    tag = f"{name}_{setting}_h{horizon}"
    ckpt_path = config.CKPT_DIR / f"{tag}.pt"
    input_dim = tr_ds.X.shape[-1]
    model = mdl.build_model(name, input_dim).to(device)

    if ckpt_path.exists():
        logger.info(f"  loading existing checkpoint {ckpt_path.name}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["state_dict"])
    else:
        logger.info(f"  training {tag} from scratch")
        sub_logger = utils.get_logger(
            f"train.{tag}", config.LOG_DIR / f"{tag}.log"
        )
        result = tr.train_sequence_model(
            model, tr_ds, va_ds, target_stats,
            device=device, ckpt_path=ckpt_path, logger=sub_logger,
        )
        # reload best
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        viz.plot_training_curves(
            result.history,
            config.FIGURES_DIR / f"{tag}_train_curve.png",
            title=tag.replace("_", " "),
        )

    pred_test = tr.predict(model, te_ds, target_stats, device=device)
    m_test = ev.regression_metrics(te_ds.y, pred_test)
    logger.info(
        f"    {name} test  MAE={m_test['MAE']:.2f} RMSE={m_test['RMSE']:.2f} "
        f"R2={m_test['R2']:.3f} spike_recall={m_test['spike_recall']:.2f}"
    )
    return pred_test, m_test


def main():
    utils.set_seed(config.SEED)
    device = utils.get_device()
    logger = utils.get_logger("finish", config.LOG_DIR / "finish_run.log")
    logger.info(f"device = {device}")

    # Reconstruct scaler stats so we can predict() the sequence models.
    scaler_path = config.DATA_PROC_DIR / "scaler.json"
    if scaler_path.exists():
        scaler = preprocessing.load_scaler(scaler_path)
        logger.info(f"loaded scaler from {scaler_path}")
    else:
        logger.info("rebuilding scaler from raw data")
        _, _, _, scaler = preprocessing.prepare()
        preprocessing.save_scaler(scaler, scaler_path)
    target_stats = scaler[config.TARGET]
    logger.info(f"target stats: mean={target_stats['mean']:.2f} "
                f"std={target_stats['std']:.2f}")

    logged = parse_logged_metrics([
        config.LOG_DIR / "full_run.log",
        config.LOG_DIR / "run_experiments.log",
    ])
    logger.info(f"parsed {len(logged)} (model, setting, horizon) "
                f"entries from existing logs")

    rows = []
    test_predictions: dict[tuple[str, str, int], tuple] = {}

    for setting in SETTINGS:
        for horizon in HORIZONS:
            tag_base = f"{setting}_h{horizon}"
            logger.info(f"--- {tag_base} ---")
            tr_ds, va_ds, te_ds = load_cached(setting, horizon)

            # Persistence
            pm_idx = config.ALL_FEATURES.index(config.TARGET)
            persist_pred = ev.persistence_baseline(te_ds.X, pm_idx, target_stats)
            m_persist = ev.regression_metrics(te_ds.y, persist_pred)
            rows.append({
                "model": "persistence", "setting": setting,
                "horizon": horizon, "split": "test", **m_persist,
            })
            test_predictions[("persistence", setting, horizon)] = (
                persist_pred, te_ds.y, te_ds.timestamps, te_ds.station_id,
            )

            # Linear: always re-fit (needed for figures, cheap)
            pred_test, m_test = fit_baseline(
                "linear", tr_ds, va_ds, te_ds, 50_000, 80_000, logger,
            )
            rows.append({
                "model": "linear", "setting": setting,
                "horizon": horizon, "split": "test", **m_test,
            })
            test_predictions[("linear", setting, horizon)] = (
                pred_test, te_ds.y, te_ds.timestamps, te_ds.station_id,
            )

            # RF / GBM: refit only if missing from logs (and we still need
            # predictions for multi h=1 -- but we only plot linear + seq,
            # so predictions aren't needed; metrics alone are fine).
            for name in ("rf", "gbm"):
                key = (name, setting, horizon)
                if key in logged:
                    logger.info(f"  using logged metrics for {name}")
                    rows.append({
                        "model": name, "setting": setting,
                        "horizon": horizon, "split": "test",
                        **logged[key],
                    })
                else:
                    pred_test, m_test = fit_baseline(
                        name, tr_ds, va_ds, te_ds, 50_000, 80_000, logger,
                    )
                    rows.append({
                        "model": name, "setting": setting,
                        "horizon": horizon, "split": "test", **m_test,
                    })

            # Sequence models: load checkpoint or train
            for name in SEQUENCE:
                pred_test, m_test = load_or_train_sequence(
                    name, setting, horizon,
                    tr_ds, va_ds, te_ds, target_stats, device, logger,
                )
                rows.append({
                    "model": name, "setting": setting,
                    "horizon": horizon, "split": "test", **m_test,
                })
                test_predictions[(name, setting, horizon)] = (
                    pred_test, te_ds.y, te_ds.timestamps, te_ds.station_id,
                )

    # Write results.csv
    results = pd.DataFrame(rows)
    results_path = config.METRICS_DIR / "results.csv"
    results.to_csv(results_path, index=False)
    logger.info(f"wrote {results_path}")

    # Test summary
    test_df = results[results["split"] == "test"].copy()
    summary = test_df.pivot_table(
        index=["model", "setting"], columns="horizon",
        values=["MAE", "RMSE", "R2", "spike_recall"],
    ).round(3)
    summary_path = config.METRICS_DIR / "summary_test.csv"
    summary.to_csv(summary_path)
    logger.info(f"wrote {summary_path}")
    logger.info("\n=== Test summary ===\n%s", summary.to_string())

    # Figures: comparison bars
    for metric in ("MAE", "RMSE", "R2", "spike_recall"):
        viz.plot_metric_comparison(
            test_df, metric=metric,
            path=config.FIGURES_DIR / f"comparison_{metric}.png",
            title=f"Test {metric} by model / setting / horizon",
        )

    # Figures: timeseries + scatter for linear + sequence models
    plot_models = {"linear"} | set(SEQUENCE)
    for key in test_predictions:
        name, setting, horizon = key
        if name not in plot_models:
            continue
        preds, ys, ts, sid = test_predictions[key]
        tag = f"{name}_{setting}_h{horizon}"
        mask = sid == 0  # Aotizhongxin only for readability
        if mask.any():
            viz.plot_prediction_timeseries(
                ts[mask], ys[mask], preds[mask],
                path=config.FIGURES_DIR / f"{tag}_timeseries.png",
                title=f"{tag} - station {config.STATIONS[0]} (test, first 21 days)",
            )
        viz.plot_scatter_pred_vs_true(
            ys, preds,
            path=config.FIGURES_DIR / f"{tag}_scatter.png",
            title=f"{tag} - predicted vs. observed (test)",
        )

    # Save test predictions for downstream analyze_predictions.py
    pred_dump = {}
    for (name, setting, horizon), (preds, ys, ts, sid) in test_predictions.items():
        key = f"{name}_{setting}_h{horizon}"
        pred_dump[f"{key}_pred"] = preds
        pred_dump[f"{key}_true"] = ys
        pred_dump[f"{key}_ts"] = ts
        pred_dump[f"{key}_sid"] = sid
    np.savez_compressed(config.RESULTS_DIR / "test_predictions.npz", **pred_dump)
    logger.info(f"saved test predictions to {config.RESULTS_DIR}/test_predictions.npz")

    logger.info("DONE.")


if __name__ == "__main__":
    main()
