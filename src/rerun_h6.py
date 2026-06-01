"""Re-train the sequence models at the 6-hour horizon with hyperparameters
that work better for the longer-horizon problem.

Why this exists: when we ran the main sweep with our default lr=1e-3 +
cosine LR + patience 4, every sequence model early-stopped after only
1 or 2 epochs at h=6 (the val MAE got worse before the model had a chance
to learn the harder mapping). That left them below the GBM baseline,
which felt wrong. Here we use a lower lr, more patience, and a plateau
LR scheduler so the model gets more time and the LR drops automatically
once we stall.

Settings:
  lr           = 2e-4
  weight_decay = 1e-4   (slightly more regularization)
  patience     = 6      (let val MAE wander up a bit before quitting)
  max_epochs   = 25
  scheduler    = ReduceLROnPlateau(factor 0.5, patience 2)

New rows are appended to results/metrics/results.csv with model names
suffixed by `_tuned`. Original h=6 rows are kept for comparison.
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import (
    config, dataset as ds_mod, evaluate as ev, models_dl as mdl,
    preprocessing, train as tr, utils, visualize as viz,
)


def make_loader(X, y, sid, mu, sigma, shuffle, batch_size=512):
    y_n = (y - mu) / sigma
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(sid).long(),
        torch.from_numpy(y_n).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_tuned(model, train_data, val_data, target_stats, device,
                ckpt_path, logger=None):
    """Same idea as src.train.train_sequence_model but with the tuned
    hyperparameters above."""
    mu = target_stats["mean"]
    sigma = target_stats["std"]
    epochs = 25
    patience = 6
    lr = 2e-4
    weight_decay = 1e-4

    tr_loader = make_loader(train_data.X, train_data.y, train_data.station_id,
                            mu, sigma, shuffle=True)
    va_loader = make_loader(val_data.X, val_data.y, val_data.station_id,
                            mu, sigma, shuffle=False)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2, min_lr=1e-6,
    )
    loss_fn = nn.SmoothL1Loss()

    best_mae = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tr_losses = []

        for x, sid, y in tr_loader:
            x = x.to(device); sid = sid.to(device); y = y.to(device)
            opt.zero_grad()
            p = model(x, sid)
            loss = loss_fn(p, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        # validation
        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for x, sid, y in va_loader:
                x = x.to(device); sid = sid.to(device)
                p = model(x, sid).cpu().numpy() * sigma + mu
                preds.append(p)
                ys.append(y.numpy() * sigma + mu)
            preds = np.concatenate(preds)
            ys = np.concatenate(ys)
            mae = float(np.mean(np.abs(preds - ys)))

        sched.step(mae)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(tr_losses)),
            "val_mae": mae,
            "lr": opt.param_groups[0]["lr"],
            "secs": elapsed,
        })

        if logger is not None:
            logger.info(f"epoch {epoch:02d} | val_MAE={mae:.2f} | "
                        f"lr={opt.param_groups[0]['lr']:.1e} | {elapsed:.1f}s")

        if mae < best_mae - 1e-4:
            best_mae = mae
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_mae": mae},
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if logger is not None:
                    logger.info(f"early stop at epoch {epoch} (best={best_epoch})")
                break

    return best_mae, best_epoch, history


def main():
    utils.set_seed(config.SEED)
    device = utils.get_device()
    logger = utils.get_logger("rerun_h6", config.LOG_DIR / "rerun_h6.log")
    logger.info(f"device = {device}")

    train_n, val_n, test_n, scaler = preprocessing.prepare()
    target_stats = scaler[config.TARGET]

    # Append to whatever's already in results.csv
    existing = pd.read_csv(config.METRICS_DIR / "results.csv")
    new_rows = []

    for setting in ("global", "multi"):
        cache = config.DATA_PROC_DIR / f"ds_{setting}_h6.npz"
        if not cache.exists():
            logger.info(f"building dataset {setting}/h6")
            t, v, te = ds_mod.build_for_setting(
                setting, train_n, val_n, test_n, 6, target_stats,
            )
            np.savez_compressed(
                cache,
                train_X=t.X, train_y=t.y,
                train_sid=t.station_id, train_ts=t.timestamps,
                val_X=v.X, val_y=v.y,
                val_sid=v.station_id, val_ts=v.timestamps,
                test_X=te.X, test_y=te.y,
                test_sid=te.station_id, test_ts=te.timestamps,
            )

        z = np.load(cache, allow_pickle=True)
        def wrap(p):
            return ds_mod.WindowedDataset(
                X=z[f"{p}_X"], y=z[f"{p}_y"],
                station_id=z[f"{p}_sid"], timestamps=z[f"{p}_ts"],
            )
        tr_ds, va_ds, te_ds = wrap("train"), wrap("val"), wrap("test")

        for name in ("gru", "lstm", "transformer"):
            logger.info(f"--- tuned {name} {setting} h=6 ---")
            utils.set_seed(config.SEED)
            input_dim = tr_ds.X.shape[-1]
            model = mdl.build_model(name, input_dim)

            ckpt = config.CKPT_DIR / f"{name}_{setting}_h6_tuned.pt"
            sub_logger = utils.get_logger(
                f"rerun.{name}_{setting}",
                config.LOG_DIR / f"{name}_{setting}_h6_tuned.log",
            )
            best_mae, best_ep, hist = train_tuned(
                model, tr_ds, va_ds, target_stats,
                device=device, ckpt_path=ckpt, logger=sub_logger,
            )

            # Evaluate the best-val checkpoint on test
            model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
            preds = tr.predict(model, te_ds, target_stats, device=device)
            m_test = ev.regression_metrics(te_ds.y, preds)
            m_test["best_epoch"] = best_ep

            logger.info(
                f"    {name}_tuned test MAE={m_test['MAE']:.2f} "
                f"RMSE={m_test['RMSE']:.2f} R2={m_test['R2']:.3f} "
                f"spike_recall={m_test['spike_recall']:.2f}"
            )
            new_rows.append({
                "model": f"{name}_tuned", "setting": setting,
                "horizon": 6, "split": "test", **m_test,
            })

            viz.plot_training_curves(
                [{"epoch": h["epoch"], "train_loss": h["train_loss"],
                  "val_mae": h["val_mae"]} for h in hist],
                config.FIGURES_DIR / f"{name}_{setting}_h6_tuned_train_curve.png",
                title=f"{name}_{setting}_h6_tuned",
            )

    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    updated.to_csv(config.METRICS_DIR / "results.csv", index=False)
    logger.info(f"appended {len(new_rows)} tuned rows to results.csv")


if __name__ == "__main__":
    main()
