"""Write a markdown summary of the experiments to results/REPORT.md.

This is the quick "what did we get?" version. The longer LaTeX report
lives in report/report.tex and pulls table fragments via
src.make_latex_tables.
"""
import pandas as pd

from . import config


def _fmt_table(df):
    return df.to_markdown(floatfmt=".3f")


def build_report(metrics_csv, out_path, figures_dir=config.FIGURES_DIR):
    df = pd.read_csv(metrics_csv)
    test = df[df["split"] == "test"].copy()

    lines = []
    lines.append("# PM2.5 Forecasting - Results")
    lines.append("")
    lines.append("ECE 228 Team 17. Models trained on the Beijing Multi-Site "
                 "Air Quality dataset (12 stations, March 2013 - February 2017).")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Window: {config.WINDOW} past hours of pollutant + weather + time features.")
    lines.append(f"- Horizons: {config.HORIZONS} hour(s) ahead.")
    lines.append("- Train / val / test (time-based):")
    lines.append("    - train: 2013-03-01 to 2015-12-31")
    lines.append("    - val:   2016-01-01 to 2016-06-30")
    lines.append("    - test:  2016-07-01 to 2017-02-28")
    lines.append("- Settings:")
    lines.append("    - **global** - one model across all stations, station id as a feature.")
    lines.append("    - **multi**  - same global model, each timestep also gets the "
                 "mean of the other 11 stations.")
    lines.append("")

    lines.append("## Test results")
    lines.append("")
    summary = test.pivot_table(
        index=["model", "setting"], columns="horizon",
        values=["MAE", "RMSE", "R2", "spike_recall"],
    )
    lines.append(_fmt_table(summary))
    lines.append("")
    lines.append("- Lower MAE / RMSE is better. Higher R^2 / spike_recall is better.")
    lines.append("- spike_recall = fraction of high-pollution timesteps "
                 "(PM2.5 >= 75 ug/m^3) where the model also predicts >= 75.")
    lines.append("")

    lines.append("## Best model per horizon")
    lines.append("")
    for h in sorted(test["horizon"].unique()):
        sub = test[test["horizon"] == h].sort_values("MAE")
        best = sub.iloc[0]
        lines.append(f"- **{h}-hour ahead**: `{best['model']}` / `{best['setting']}` "
                     f"-- MAE={best['MAE']:.2f}, RMSE={best['RMSE']:.2f}, "
                     f"R^2={best['R2']:.3f}, spike_recall={best['spike_recall']:.2f}")
    lines.append("")

    lines.append("## Does multi-site context help?")
    lines.append("")
    pivot = test.pivot_table(index=["model", "horizon"], columns="setting",
                              values="MAE")
    if {"global", "multi"}.issubset(pivot.columns):
        pivot["delta (multi - global)"] = pivot["multi"] - pivot["global"]
        lines.append(_fmt_table(pivot))
        lines.append("")
        avg_delta = float(pivot["delta (multi - global)"].mean())
        direction = "lower" if avg_delta < 0 else "higher"
        lines.append(f"Average MAE change from adding multi-site context: "
                     f"**{avg_delta:+.3f}** ug/m^3 (multi is {direction}).")
        lines.append("")

    lines.append("## Figures")
    lines.append("")
    figs = sorted(p.name for p in figures_dir.glob("*.png"))
    for f in figs:
        lines.append(f"- `figures/{f}`")
    lines.append("")
    lines.append("## Raw metrics")
    lines.append("")
    lines.append(f"All per-run metrics: [`results.csv`]({metrics_csv.name})")
    lines.append(f"Pivoted test summary: `summary_test.csv`")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    build_report(
        config.METRICS_DIR / "results.csv",
        config.RESULTS_DIR / "REPORT.md",
    )
