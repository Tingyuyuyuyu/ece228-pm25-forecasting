"""Plotting helpers for the project. Nothing fancy, just matplotlib."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(history, path, title):
    """Two-axis plot: training loss (left) + validation MAE (right)."""
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(df["epoch"], df["train_loss"], "-o",
             color="tab:blue", label="train loss (norm)")
    ax2.plot(df["epoch"], df["val_mae"], "-s",
             color="tab:red", label="val MAE")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:blue")
    ax2.set_ylabel("val MAE (ug/m^3)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prediction_timeseries(timestamps, y_true, y_pred, path, title,
                                start=None, days=21):
    """Overlay observed PM2.5 and the model's prediction for `days` days."""
    ts = pd.to_datetime(timestamps)
    df = pd.DataFrame({"y": y_true, "yhat": y_pred}, index=ts).sort_index()

    if start is None:
        start = df.index.min()
    end = start + pd.Timedelta(days=days)
    df = df.loc[(df.index >= start) & (df.index <= end)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["y"], color="black", linewidth=1.2, label="observed")
    ax.plot(df.index, df["yhat"], color="tab:red", linewidth=1.0, alpha=0.85,
            label="predicted")
    ax.axhline(75, color="tab:orange", linestyle="--", linewidth=0.8,
               label="unhealthy (75 ug/m^3)")
    ax.set_ylabel("PM2.5 (ug/m^3)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter_pred_vs_true(y_true, y_pred, path, title):
    """2D histogram of predicted vs. observed PM2.5."""
    fig, ax = plt.subplots(figsize=(5, 5))

    h = ax.hist2d(y_true, y_pred, bins=80, cmin=1, cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="count")

    # Plot the y = yhat line up to the 99.5th percentile of either array
    lim = float(np.percentile(np.concatenate([y_true, y_pred]), 99.5))
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y = yhat")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("observed PM2.5 (ug/m^3)")
    ax.set_ylabel("predicted PM2.5 (ug/m^3)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(df, metric, path, title):
    """Grouped bar chart of one metric, with bars grouped by model and
    hue=(setting, horizon).

    df should have columns ['model', 'setting', 'horizon', metric].
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivoted = df.pivot_table(index="model", columns=["setting", "horizon"],
                              values=metric)
    pivoted.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_ylabel(metric)
    ax.set_xlabel("model")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="setting / horizon", fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
