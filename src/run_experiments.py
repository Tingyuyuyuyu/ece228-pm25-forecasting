"""Main driver. Trains every (model x setting x horizon) combination and
writes the results out.

Outputs:
  results/metrics/results.csv      one row per (model, setting, horizon, split)
  results/metrics/summary_test.csv pivoted test-only summary
  results/test_predictions.npz     raw test predictions for downstream analysis
  results/figures/*.png            comparison bar charts, training curves, etc.
  checkpoints/*.pt                 best sequence-model weights
  logs/*.log                       per-run training logs

How to run:
  python -m src.run_experiments                # everything
  python -m src.run_experiments --skip-sequence  # only baselines
  python -m src.run_experiments --horizons 1     # only 1-hour ahead
"""
import argparse
import time

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


SETTINGS  = ("global", "multi")
BASELINES = ("linear", "rf", "gbm")
SEQUENCE  = ("gru", "lstm", "transformer")


def build_or_load_dataset(setting, horizon, target_stats,
                           train_n, val_n, test_n, logger):
    """Build the (train, val, test) WindowedDataset for one (setting, horizon),
    caching the result to disk so we don't redo it every time."""
    cache = config.DATA_PROC_DIR / f"ds_{setting}_h{horizon}.npz"
    if cache.exists():
        logger.info(f"loading cached dataset {cache.name}")
        z = np.load(cache, allow_pickle=True)
        def _wrap(prefix):
            return ds_mod.WindowedDataset(
                X=z[f"{prefix}_X"],
                y=z[f"{prefix}_y"],
                station_id=z[f"{prefix}_sid"],
                timestamps=z[f"{prefix}_ts"],
            )
        return _wrap("train"), _wrap("val"), _wrap("test")

    logger.info(f"building dataset setting={setting} horizon={horizon}h")
    tr_ds, va_ds, te_ds = ds_mod.build_for_setting(
        setting, train_n, val_n, test_n, horizon, target_stats,
    )
    np.savez_compressed(
        cache,
        train_X=tr_ds.X, train_y=tr_ds.y,
        train_sid=tr_ds.station_id, train_ts=tr_ds.timestamps,
        val_X=va_ds.X, val_y=va_ds.y,
        val_sid=va_ds.station_id, val_ts=va_ds.timestamps,
        test_X=te_ds.X, test_y=te_ds.y,
        test_sid=te_ds.station_id, test_ts=te_ds.timestamps,
    )
    logger.info(f"cached -> {cache.name} "
                f"(train {tr_ds.X.shape}, val {va_ds.X.shape}, test {te_ds.X.shape})")
    return tr_ds, va_ds, te_ds


def run_baseline(model_name, tr_ds, va_ds, te_ds, target_stats,
                 max_rf_n, max_gbm_n, logger):
    """Train a feature-based baseline and evaluate on val + test."""
    Xtr = mb.flatten(tr_ds.X, tr_ds.station_id)
    Xva = mb.flatten(va_ds.X, va_ds.station_id)
    Xte = mb.flatten(te_ds.X, te_ds.station_id)
    ytr, yva, yte = tr_ds.y, va_ds.y, te_ds.y

    # Tree models get a subsample of train so they fit in reasonable time
    if model_name == "rf":
        Xtr_s, ytr_s = mb.subsample(Xtr, ytr, max_rf_n)
    elif model_name == "gbm":
        Xtr_s, ytr_s = mb.subsample(Xtr, ytr, max_gbm_n)
    else:
        Xtr_s, ytr_s = Xtr, ytr

    logger.info(f"  fitting {model_name} on X={Xtr_s.shape}")
    t0 = time.time()
    model = mb.get_model(model_name)
    model.fit(Xtr_s, ytr_s)
    elapsed = time.time() - t0

    pred_val  = model.predict(Xva)
    pred_test = model.predict(Xte)
    m_val  = ev.regression_metrics(yva, pred_val)
    m_test = ev.regression_metrics(yte, pred_test)
    m_val["fit_secs"]  = elapsed
    m_test["fit_secs"] = elapsed
    return model, pred_test, m_val, m_test


def run_sequence(model_name, tr_ds, va_ds, te_ds, target_stats, device,
                 ckpt_dir, log_dir, tag, logger):
    """Train a sequence model and evaluate on val + test."""
    input_dim = tr_ds.X.shape[-1]
    model = mdl.build_model(model_name, input_dim)

    ckpt_path = ckpt_dir / f"{tag}.pt"
    sub_logger = utils.get_logger(f"train.{tag}", log_dir / f"{tag}.log")
    result = tr.train_sequence_model(
        model, tr_ds, va_ds, target_stats,
        device=device, ckpt_path=ckpt_path, logger=sub_logger,
    )

    # Reload the best-val checkpoint before evaluating on val + test
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)["state_dict"]
    )
    pred_val  = tr.predict(model, va_ds, target_stats, device=device)
    pred_test = tr.predict(model, te_ds, target_stats, device=device)
    m_val  = ev.regression_metrics(va_ds.y, pred_val)
    m_test = ev.regression_metrics(te_ds.y, pred_test)
    m_val["best_epoch"]  = result.best_epoch
    m_test["best_epoch"] = result.best_epoch

    # Save the training curve figure
    viz.plot_training_curves(
        result.history,
        config.FIGURES_DIR / f"{tag}_train_curve.png",
        title=tag.replace("_", " "),
    )
    return model, pred_test, result.history, m_val, m_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS),
                        choices=list(SETTINGS))
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=config.HORIZONS)
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES),
                        choices=list(BASELINES))
    parser.add_argument("--sequence", nargs="+", default=list(SEQUENCE),
                        choices=list(SEQUENCE))
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-sequence", action="store_true")
    parser.add_argument("--max-rf-n", type=int, default=50_000,
                        help="cap RF training rows for tractability")
    parser.add_argument("--max-gbm-n", type=int, default=80_000,
                        help="cap GBM training rows for tractability")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    args = parser.parse_args()

    utils.set_seed(config.SEED)
    device = utils.get_device()
    logger = utils.get_logger("run", config.LOG_DIR / "run_experiments.log")
    logger.info(f"device = {device}")
    config.MAX_EPOCHS = args.epochs

    logger.info("preprocessing dataset...")
    train_n, val_n, test_n, scaler = preprocessing.prepare()
    preprocessing.save_scaler(scaler, config.DATA_PROC_DIR / "scaler.json")
    target_stats = scaler[config.TARGET]
    logger.info(f"target stats: mean={target_stats['mean']:.2f} "
                f"std={target_stats['std']:.2f}")

    rows = []
    test_predictions = {}   # (model, setting, horizon) -> (preds, true, ts, sid)

    for setting in args.settings:
        for horizon in args.horizons:
            tag_base = f"{setting}_h{horizon}"
            logger.info(f"--- setting={setting} horizon={horizon}h ---")
            tr_ds, va_ds, te_ds = build_or_load_dataset(
                setting, horizon, target_stats, train_n, val_n, test_n, logger,
            )

            # Persistence baseline (just to anchor the comparison)
            pm_idx = config.ALL_FEATURES.index(config.TARGET)
            persist_pred = ev.persistence_baseline(te_ds.X, pm_idx, target_stats)
            m_persist = ev.regression_metrics(te_ds.y, persist_pred)
            rows.append({"model": "persistence", "setting": setting,
                          "horizon": horizon, "split": "test", **m_persist})
            logger.info(f"  persistence  test MAE={m_persist['MAE']:.2f} "
                        f"RMSE={m_persist['RMSE']:.2f} R2={m_persist['R2']:.3f}")

            if not args.skip_baselines:
                for name in args.baselines:
                    logger.info(f"  >>> baseline: {name}")
                    _, pred_test, m_val, m_test = run_baseline(
                        name, tr_ds, va_ds, te_ds, target_stats,
                        args.max_rf_n, args.max_gbm_n, logger,
                    )
                    rows.append({"model": name, "setting": setting,
                                  "horizon": horizon, "split": "val", **m_val})
                    rows.append({"model": name, "setting": setting,
                                  "horizon": horizon, "split": "test", **m_test})
                    test_predictions[(name, setting, horizon)] = (
                        pred_test, te_ds.y, te_ds.timestamps, te_ds.station_id,
                    )
                    logger.info(f"    {name} test  MAE={m_test['MAE']:.2f} "
                                f"RMSE={m_test['RMSE']:.2f} R2={m_test['R2']:.3f} "
                                f"spike_recall={m_test['spike_recall']:.2f}")

            if not args.skip_sequence:
                for name in args.sequence:
                    logger.info(f"  >>> sequence: {name}")
                    tag = f"{name}_{tag_base}"
                    _, pred_test, hist, m_val, m_test = run_sequence(
                        name, tr_ds, va_ds, te_ds, target_stats, device,
                        config.CKPT_DIR, config.LOG_DIR, tag, logger,
                    )
                    rows.append({"model": name, "setting": setting,
                                  "horizon": horizon, "split": "val", **m_val})
                    rows.append({"model": name, "setting": setting,
                                  "horizon": horizon, "split": "test", **m_test})
                    test_predictions[(name, setting, horizon)] = (
                        pred_test, te_ds.y, te_ds.timestamps, te_ds.station_id,
                    )
                    logger.info(f"    {name} test  MAE={m_test['MAE']:.2f} "
                                f"RMSE={m_test['RMSE']:.2f} R2={m_test['R2']:.3f} "
                                f"spike_recall={m_test['spike_recall']:.2f}")

    # --- Save out everything --------------------------------------------
    results = pd.DataFrame(rows)
    results_path = config.METRICS_DIR / "results.csv"
    results.to_csv(results_path, index=False)
    logger.info(f"\nwrote {results_path}")

    test_df = results[results["split"] == "test"].copy()
    summary = test_df.pivot_table(
        index=["model", "setting"], columns="horizon",
        values=["MAE", "RMSE", "R2", "spike_recall"],
    ).round(3)
    summary_path = config.METRICS_DIR / "summary_test.csv"
    summary.to_csv(summary_path)
    logger.info("\n=== Test summary ===\n%s", summary.to_string())

    # --- Figures --------------------------------------------------------
    for metric in ("MAE", "RMSE", "R2", "spike_recall"):
        viz.plot_metric_comparison(
            test_df, metric=metric,
            path=config.FIGURES_DIR / f"comparison_{metric}.png",
            title=f"Test {metric} by model / setting / horizon",
        )

    # Per-model timeseries + scatter, but only for the linear baseline and
    # the sequence models (otherwise we'd have 50+ plots)
    plot_models = {"linear"} | set(SEQUENCE)
    for key in test_predictions:
        name, setting, horizon = key
        if name not in plot_models:
            continue
        preds, ys, ts, sid = test_predictions[key]
        tag = f"{name}_{setting}_h{horizon}"
        mask = sid == 0   # plot just one station (Aotizhongxin) for readability
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

    # Save raw test predictions for any post-hoc analysis
    pred_dump = {}
    for (name, setting, horizon), (preds, ys, ts, sid) in test_predictions.items():
        key = f"{name}_{setting}_h{horizon}"
        pred_dump[f"{key}_pred"] = preds
        pred_dump[f"{key}_true"] = ys
        pred_dump[f"{key}_ts"]   = ts
        pred_dump[f"{key}_sid"]  = sid
    if pred_dump:
        np.savez_compressed(config.RESULTS_DIR / "test_predictions.npz", **pred_dump)
        logger.info(f"saved test predictions to {config.RESULTS_DIR}/test_predictions.npz")

    logger.info("DONE.")


if __name__ == "__main__":
    main()
