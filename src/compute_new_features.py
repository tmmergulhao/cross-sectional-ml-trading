"""
Compute and save only the new features, reusing the existing beta.parquet
to avoid re-running the expensive 252-day rolling OLS.

New parquets written:
  return_zscore.parquet   (updated: adds window 5)
  ivol.parquet
  idio_momentum.parquet
  price_to_ma.parquet
  vol_ratio.parquet

Usage (from repo root):
    python -m src.compute_new_features
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.compute_features import (
    FEATURE_CONFIG,
    _load_long_panel,
    _save,
)
from src.features.features import (
    compute_idio_momentum,
    compute_ivol,
    compute_price_to_ma,
    compute_return_zscore,
    compute_vol_ratio,
)

RAW_PATH     = Path("data/raw/yahoo_S&P_500/raw.parquet")
FEATURES_DIR = Path("data/features/yahoo_S&P_500")
cfg          = FEATURE_CONFIG


def main() -> None:
    # ── Raw panel (price/volume only; no market factors needed) ───────────
    print(f"Loading raw panel from {RAW_PATH} ...")
    df = _load_long_panel(RAW_PATH)
    print(f"  {df.shape[0]:,} rows, {df['Ticker'].nunique()} tickers")

    # ── Return z-score (now includes window 5) ─────────────────────────────
    print("\nComputing return_zscore", cfg["zscore_windows"], "...")
    tmp = compute_return_zscore(df, cfg["zscore_windows"])
    _save(tmp, [f"return_zscore_{h}" for h in cfg["zscore_windows"]],
          FEATURES_DIR / "return_zscore.parquet")

    # ── Price relative to MA ───────────────────────────────────────────────
    print("\nComputing price_to_ma", cfg["price_to_ma_windows"], "...")
    tmp = compute_price_to_ma(df, cfg["price_to_ma_windows"])
    _save(tmp, [f"price_to_ma_{h}" for h in cfg["price_to_ma_windows"]],
          FEATURES_DIR / "price_to_ma.parquet")

    # ── Volatility ratio ───────────────────────────────────────────────────
    print("\nComputing vol_ratio", cfg["vol_ratio_pairs"], "...")
    tmp = compute_vol_ratio(df, cfg["vol_ratio_pairs"])
    _save(tmp, [f"vol_ratio_{s}_{l}" for s, l in cfg["vol_ratio_pairs"]],
          FEATURES_DIR / "vol_ratio.parquet")

    # ── IVOL + idio momentum: reuse existing beta.parquet ─────────────────
    print(f"\nLoading beta.parquet ...")
    beta_df = pd.read_parquet(FEATURES_DIR / "beta.parquet")
    # beta_df has (Date, Ticker) MultiIndex → reformat for _save / feature fns
    beta_reset = beta_df.reset_index(level="Ticker").sort_values(["Ticker", "Date"])

    print("Computing ivol", cfg["ivol_windows"], "...")
    tmp = compute_ivol(beta_reset, cfg["ivol_windows"])
    _save(tmp, [f"ivol_{h}" for h in cfg["ivol_windows"]],
          FEATURES_DIR / "ivol.parquet")

    print("Computing idio_momentum", cfg["idio_mom_windows"], "...")
    tmp = compute_idio_momentum(beta_reset, cfg["idio_mom_windows"])
    _save(tmp, [f"idio_mom_{l}_{r}" for l, r in cfg["idio_mom_windows"]],
          FEATURES_DIR / "idio_momentum.parquet")

    print("\nDone.")


if __name__ == "__main__":
    main()
