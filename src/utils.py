from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Newey-West t-stat for a time series (accounts for serial correlation)
# ---------------------------------------------------------------------------

def _newey_west_tstat(series: pd.Series, max_lags: Optional[int] = None) -> float:
    """t-statistic for H0: mean = 0, using Newey-West HAC variance."""
    x = series.dropna().values
    n = len(x)
    if n < 4:
        return np.nan
    mu = x.mean()
    if max_lags is None:
        max_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    gamma0 = np.var(x, ddof=1)
    hac_var = gamma0
    for lag in range(1, max_lags + 1):
        w = 1 - lag / (max_lags + 1)          # Bartlett kernel
        gamma_lag = np.cov(x[lag:], x[:-lag], ddof=1)[0, 1]
        hac_var += 2 * w * gamma_lag

    se = np.sqrt(max(hac_var, 0) / n)
    return mu / se if se > 0 else np.nan


# ---------------------------------------------------------------------------
# Main diagnostic function
# ---------------------------------------------------------------------------

def diagnose(
    results: list[dict],
    ic_on_every: int,
    save_dir: Optional[str | Path] = None,
    model_name: str = "model",
) -> tuple[pd.Series, pd.Series]:
    """
    Compute IC, ICIR, long-short spread and produce diagnostic plots.

    Parameters
    ----------
    results      : list of fold dicts from WalkForwardBacktester.run()
    ic_on_every  : stride for sampling non-overlapping cross-sections
    save_dir     : if given, save charts and tables under this directory
    model_name   : used in plot titles and saved filenames

    Returns
    -------
    ic_series, ls_spread  (both as pd.Series indexed by Date)
    """
    # ── 1. Assemble predictions ────────────────────────────────────────────
    frames = []
    for r in results:
        tmp = pd.DataFrame(
            {"pred": r["preds"], "actual": r["actuals"]},
            index=r["index"],
        )
        frames.append(tmp)

    pred_df = pd.concat(frames).sort_index()
    pred_df.index.name = "Date"

    sampled_dates = pred_df.index.unique().sort_values()[::ic_on_every]
    pred_df_sampled = pred_df[pred_df.index.isin(sampled_dates)]

    # ── 2. Information Coefficient ─────────────────────────────────────────
    ic_series = (
        pred_df_sampled.groupby("Date")
        .apply(lambda x: x["pred"].corr(x["actual"], method="spearman"))
        .rename("IC")
    )

    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic
    nw_t = _newey_west_tstat(ic_series)
    nw_p = 2 * (1 - stats.t.cdf(abs(nw_t), len(ic_series) - 1))
    hit_rate = (ic_series > 0).mean()

    # ── 3. Long-short quintile spread ──────────────────────────────────────
    def quintile_spread(x):
        top = x[x["pred"] >= x["pred"].quantile(0.8)]["actual"].mean()
        bottom = x[x["pred"] <= x["pred"].quantile(0.2)]["actual"].mean()
        return top - bottom

    ls_spread = (
        pred_df_sampled.groupby("Date")
        .apply(quintile_spread)
        .rename("LS_spread")
    )

    # ── 4. Print summary ───────────────────────────────────────────────────
    print("=" * 50)
    print(f"    CROSS-SECTIONAL DIAGNOSTIC  [{model_name.upper()}]")
    print("=" * 50)
    print(f"  Period:            {ic_series.index.min().date()} → {ic_series.index.max().date()}")
    print(f"  # Cross-sections:  {len(ic_series)}")
    print()
    print(f"  Mean IC:           {mean_ic:.4f}")
    print(f"  IC Std:            {std_ic:.4f}")
    print(f"  ICIR:              {icir:.4f}")
    print(f"  IC t-stat (NW):    {nw_t:.2f}  (p={nw_p:.4f})")
    print(f"  Hit Rate:          {hit_rate:.1%}  (% months IC > 0)")
    print()
    print(f"  Mean L/S Spread:   {ls_spread.mean():.4f}  (rank units)")
    print(f"  L/S Hit Rate:      {(ls_spread > 0).mean():.1%}")
    print("=" * 50)

    _print_interpretation(mean_ic, icir, nw_t)

    # ── 5. Plots ───────────────────────────────────────────────────────────
    fig = _make_diagnostic_plot(ic_series, ls_spread, mean_ic, model_name)

    if save_dir is not None:
        save_dir = Path(save_dir)
        _save_charts(fig, ic_series, ls_spread, save_dir, model_name)
        _save_tables(
            ic_series, ls_spread, mean_ic, std_ic, icir,
            nw_t, nw_p, hit_rate, ls_spread.mean(), (ls_spread > 0).mean(),
            save_dir, model_name,
        )

    plt.show()
    return ic_series, ls_spread


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_interpretation(mean_ic: float, icir: float, t_stat: float) -> None:
    print()
    sig = abs(t_stat) > 2
    print(f"  {'✓' if sig else '✗'} IC {'is' if sig else 'is NOT'} statistically significant (NW |t| {'>' if sig else '<'} 2)")

    if mean_ic > 0.05:
        print("  ✓ Mean IC > 0.05 — good for a cross-sectional model")
    elif mean_ic > 0.02:
        print("  ~ Mean IC 0.02–0.05 — acceptable, typical in practice")
    else:
        print("  ✗ Mean IC < 0.02 — weak signal, review features/target")

    if icir > 0.5:
        print("  ✓ ICIR > 0.5 — consistent predictions across time")
    else:
        print("  ~ ICIR < 0.5 — high variability in monthly performance")
    print()


def _make_diagnostic_plot(
    ic_series: pd.Series,
    ls_spread: pd.Series,
    mean_ic: float,
    model_name: str,
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    colors = ["steelblue" if v > 0 else "tomato" for v in ic_series]
    ax1.bar(ic_series.index, ic_series.values, color=colors, width=15)
    ax1.axhline(mean_ic, color="black", linestyle="--", linewidth=1,
                label=f"Mean IC = {mean_ic:.4f}")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.set_title("Monthly Information Coefficient (IC)")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    ic_series.cumsum().plot(ax=ax2, color="steelblue")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_title("Cumulative IC")

    ax3 = fig.add_subplot(gs[1, 1])
    ls_spread.cumsum().plot(ax=ax3, color="darkorange")
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.set_title("Cumulative L/S Rank Spread (Top vs Bottom Quintile)")

    fig.suptitle(
        f"Walk-Forward Validation — {model_name.upper()}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    return fig


def _save_charts(
    fig: plt.Figure,
    ic_series: pd.Series,
    ls_spread: pd.Series,
    save_dir: Path,
    model_name: str,
) -> None:
    charts_dir = save_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(charts_dir / f"{model_name}_diagnostic.png", dpi=150, bbox_inches="tight")

    fig_ic, ax = plt.subplots(figsize=(12, 4))
    colors = ["steelblue" if v > 0 else "tomato" for v in ic_series]
    ax.bar(ic_series.index, ic_series.values, color=colors, width=15)
    ax.axhline(ic_series.mean(), color="black", linestyle="--", linewidth=1,
               label=f"Mean = {ic_series.mean():.4f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"Monthly IC — {model_name.upper()}")
    ax.legend()
    plt.tight_layout()
    fig_ic.savefig(charts_dir / f"{model_name}_ic_monthly.png", dpi=150, bbox_inches="tight")
    plt.close(fig_ic)

    fig_cum, ax = plt.subplots(figsize=(12, 4))
    ic_series.cumsum().plot(ax=ax, color="steelblue")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"Cumulative IC — {model_name.upper()}")
    plt.tight_layout()
    fig_cum.savefig(charts_dir / f"{model_name}_ic_cumulative.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cum)

    fig_ls, ax = plt.subplots(figsize=(12, 4))
    ls_spread.cumsum().plot(ax=ax, color="darkorange")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"Cumulative L/S Rank Spread — {model_name.upper()}")
    plt.tight_layout()
    fig_ls.savefig(charts_dir / f"{model_name}_ls_spread.png", dpi=150, bbox_inches="tight")
    plt.close(fig_ls)

    print(f"  Charts saved to {charts_dir}")


def _save_tables(
    ic_series: pd.Series,
    ls_spread: pd.Series,
    mean_ic: float,
    std_ic: float,
    icir: float,
    nw_t: float,
    nw_p: float,
    hit_rate: float,
    mean_ls: float,
    ls_hit_rate: float,
    save_dir: Path,
    model_name: str,
) -> None:
    tables_dir = save_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        "metric": [
            "Mean IC", "IC Std", "ICIR",
            "IC t-stat (Newey-West)", "IC p-value (NW)",
            "IC Hit Rate",
            "Mean L/S Spread", "L/S Hit Rate",
            "N Cross-sections",
            "Period Start", "Period End",
        ],
        "value": [
            f"{mean_ic:.4f}", f"{std_ic:.4f}", f"{icir:.4f}",
            f"{nw_t:.2f}", f"{nw_p:.4f}",
            f"{hit_rate:.1%}",
            f"{mean_ls:.4f}", f"{ls_hit_rate:.1%}",
            str(len(ic_series)),
            str(ic_series.index.min().date()),
            str(ic_series.index.max().date()),
        ],
    })
    summary.to_csv(tables_dir / f"{model_name}_summary.csv", index=False)
    ic_series.to_frame("IC").to_csv(tables_dir / f"{model_name}_ic_monthly.csv")
    ls_spread.to_frame("LS_spread").to_csv(tables_dir / f"{model_name}_ls_spread_monthly.csv")

    print(f"  Tables saved to {tables_dir}")


# ---------------------------------------------------------------------------
# Feature importance helper
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance: pd.Series,
    model_name: str = "model",
    top_n: int = 20,
    save_dir: Optional[str | Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of feature importances (top_n features)."""
    imp = importance.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(imp.index, imp.values, color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {model_name.upper()} (top {top_n})")
    plt.tight_layout()

    if save_dir is not None:
        charts_dir = Path(save_dir) / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(charts_dir / f"{model_name}_feature_importance.png",
                    dpi=150, bbox_inches="tight")
        print(f"  Feature importance chart saved to {charts_dir}")

    return fig