"""Training loop for the sequence models (LSTM / GRU / Transformer).

Why we made some of these choices:
  - Loss is smooth-L1 (Huber-ish): MSE was sensitive to PM2.5 outliers
    early in training. Smooth-L1 keeps the gradient bounded for large
    errors but is still differentiable.
  - Optimizer is AdamW + cosine LR. Cosine helps because we don't tune
    LR schedules per model.
  - We train against the STANDARDIZED target for a better-conditioned
    loss surface, then de-standardize predictions for metrics so they
    show up in ug/m^3.
  - Early stopping uses validation MAE (not val loss) -- MAE is what we
    actually report.
  - Non-finite-loss guard: occasionally MPS produces a NaN gradient on
    the very first batches before the GRU stabilizes. We just skip
    those steps. Without this the whole run goes to NaN.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config


class TrainResult:
    """Lightweight bag of outputs from a training run."""
    def __init__(self, best_val_mae, best_epoch, history, ckpt_path):
        self.best_val_mae = best_val_mae
        self.best_epoch = best_epoch
        self.history = history
        self.ckpt_path = ckpt_path


def _make_loader(X, y, sid, target_mean, target_std, batch_size, shuffle):
    """Wrap numpy arrays in a DataLoader. y is standardized for training."""
    y_norm = (y - target_mean) / target_std
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(sid).long(),
        torch.from_numpy(y_norm).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _evaluate(model, loader, target_mean, target_std, device):
    """Return (MAE, RMSE) on a loader, in original ug/m^3 units."""
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, sid, y in loader:
            x = x.to(device)
            sid = sid.to(device)
            p = model(x, sid).cpu().numpy()
            preds.append(p * target_std + target_mean)
            ys.append(y.numpy() * target_std + target_mean)
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    mae = float(np.mean(np.abs(preds - ys)))
    rmse = float(np.sqrt(np.mean((preds - ys) ** 2)))
    return mae, rmse


def train_sequence_model(
    model,
    train_data,
    val_data,
    target_stats,
    *,
    device,
    epochs=config.MAX_EPOCHS,
    batch_size=config.BATCH_SIZE,
    lr=config.LR,
    weight_decay=config.WEIGHT_DECAY,
    patience=config.EARLY_STOP_PATIENCE,
    grad_clip=config.GRAD_CLIP,
    ckpt_path,
    logger=None,
):
    mu = target_stats["mean"]
    sigma = target_stats["std"]

    tr_loader = _make_loader(train_data.X, train_data.y, train_data.station_id,
                             mu, sigma, batch_size, shuffle=True)
    va_loader = _make_loader(val_data.X, val_data.y, val_data.station_id,
                             mu, sigma, batch_size, shuffle=False)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    best_mae = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        epoch_losses = []

        for x, sid, y in tr_loader:
            x = x.to(device)
            sid = sid.to(device)
            y = y.to(device)

            opt.zero_grad()
            pred = model(x, sid)
            loss = loss_fn(pred, y)

            # safety: skip any batch that produced a NaN/Inf loss
            if not torch.isfinite(loss):
                if logger is not None:
                    logger.warning("non-finite loss, skipping step")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_losses.append(loss.item())

        sched.step()

        val_mae, val_rmse = _evaluate(model, va_loader, mu, sigma, device)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "lr": opt.param_groups[0]["lr"],
            "secs": elapsed,
        })

        if logger is not None:
            logger.info(
                f"epoch {epoch:02d} | "
                f"train_loss={np.mean(epoch_losses):.4f} | "
                f"val_MAE={val_mae:.2f} | val_RMSE={val_rmse:.2f} | "
                f"{elapsed:.1f}s"
            )

        # early stopping on val MAE
        if val_mae < best_mae - 1e-4:
            best_mae = val_mae
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_mae": val_mae},
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if logger is not None:
                    logger.info(f"early stop at epoch {epoch} (best={best_epoch})")
                break

    return TrainResult(best_mae, best_epoch, history, ckpt_path)


def predict(model, data, target_stats, *, device, batch_size=config.BATCH_SIZE):
    """Run the model on a WindowedDataset and return predictions in ug/m^3."""
    mu = target_stats["mean"]
    sigma = target_stats["std"]
    loader = _make_loader(data.X, data.y, data.station_id,
                           mu, sigma, batch_size, shuffle=False)
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for x, sid, _ in loader:
            x = x.to(device)
            sid = sid.to(device)
            p = model(x, sid).cpu().numpy()
            out.append(p * sigma + mu)
    return np.concatenate(out)
