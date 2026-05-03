"""
Load raw price data, compute all features, and save one parquet per feature group.

Output layout:
    data/features/<dataset_name>/
        momentum.parquet        mom_* columns
        volatility.parquet      vol_*, downside_dev_* columns
        liquidity.parquet       dollar_volume_* columns
        extreme.parquet         dist_*_high, max_* columns
        skewness.parquet        skew_* columns
        beta.parquet            rolling_beta, idiosyncratic columns
        amihud.parquet          amihud_* columns
        parkinson_vol.parquet   parkinsons_vol_* columns
        volume_surprise.parquet vol_surprise_* columns

Each parquet has a (Date, Ticker) MultiIndex and only the feature columns.

Usage:
    python -m src.compute_features \\
        --raw   data/raw/yahoo_S&P_500/raw.parquet \\
        --french data/raw/french \\
        [--out  data/features]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.french import FrenchDataLoader
from src.features.features import (
    compute_amihud,
    compute_extreme_features,
    compute_idio_momentum,
    compute_ivol,
    compute_liquidity,
    compute_momentum_features,
    compute_parkinson_vol,
    compute_price_to_ma,
    compute_price_volume_interaction,
    compute_return_zscore,
    compute_skewness_features,
    compute_volatility_features,
    compute_vol_ratio,
    compute_volume_surprise,
)

FEATURE_CONFIG: dict = {
    "mom_windows":           [(1, 5), (1, 21), (21, 126), 
                              (21, 252), (126, 252), (252, 756)],
    "vol_windows":           [21, 63, 252],
    "liq_windows":           [21, 63],
    "high_windows":          [252],
    "max_windows":           [21],
    "skew_windows":          [21, 63, 252],
    "beta_window":           252,
    "amihud_windows":        [21, 63, 252],
    "parkinson_windows":     [21, 63, 252],
    "vol_surprise_windows":  [21, 63, 252],
    "zscore_windows":        [5, 21, 63, 252],
    "pv_interaction_windows": [21, 63, 252],
    # new signals
    "ivol_windows":          [21, 63, 252],
    "price_to_ma_windows":   [5, 20, 63, 200],
    "idio_mom_windows":      [(21, 252), (21, 126)],
    "vol_ratio_pairs":       [(5, 21), (10, 63), (21, 252)],
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_long_panel(raw_path: Path) -> pd.DataFrame:
    """
    Read wide parquet (Date x (field, Ticker)) and return a long panel.

    Result has Date as index, with columns: Ticker, Adj Close, Close,
    High, Low, Open, Volume.  Sorted by (Ticker, Date).
    """
    wide = pd.read_parquet(raw_path)
    wide.columns.names = ["field", "Ticker"]
    long = wide.stack(level="Ticker")          # → (Date, Ticker) MultiIndex
    long.index.names = ["Date", "Ticker"]
    long = long.reset_index(level="Ticker")    # Date stays as index; Ticker → column
    long = long.sort_values(["Ticker", "Date"])
    return long


def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df.groupby("Ticker")["Adj Close"].pct_change()
    return df


def _add_market_factors(df: pd.DataFrame, french_cache_dir: Path) -> pd.DataFrame:
    """
    Attach RF and Mkt-RF from Ken French daily factors.

    Uses .map() on Date values so repeated dates (one per ticker) are handled
    correctly without index-alignment issues.
    """
    loader = FrenchDataLoader(cache_dir=str(french_cache_dir))
    factors = loader.get_ff5_plus_mom(freq="daily")

    df = df.reset_index()               # Date becomes a regular column
    df["RF"]     = df["Date"].map(factors["RF"])
    df["Mkt_RF"] = df["Date"].map(factors["Mkt-RF"])
    df = df.set_index("Date")

    df["returns_minus_RF"] = df["returns"] - df["RF"]
    return df


# ---------------------------------------------------------------------------
# Beta (implemented here rather than via features.py because that function
# assumes a unique Date index, which a panel doesn't have)
# ---------------------------------------------------------------------------

def _compute_beta(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()

    def _rolling_beta(g: pd.DataFrame) -> pd.Series:
        cov = g["returns_minus_RF"].rolling(window).cov(g["Mkt_RF"])
        var = g["Mkt_RF"].rolling(window).var()
        return cov / var

    raw_betas = df.groupby("Ticker", group_keys=False).apply(_rolling_beta)
    df["rolling_beta"] = raw_betas
    df["rolling_beta"] = df.groupby("Ticker")["rolling_beta"].shift(1)
    df["idiosyncratic"] = df["returns_minus_RF"] - df["rolling_beta"] * df["Mkt_RF"]
    return df


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, cols: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["Ticker"] + cols].copy()
    out = out.reset_index().set_index(["Date", "Ticker"])
    out.to_parquet(out_path)
    n_nonnull = out.notna().sum().sum()
    print(f"  -> {out_path.name}  shape={out.shape}  non-null values={n_nonnull:,}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_and_save(
    raw_path: str | Path,
    french_cache_dir: str | Path,
    features_base_dir: str | Path = "data/features",
) -> None:
    raw_path          = Path(raw_path)
    french_cache_dir  = Path(french_cache_dir)
    features_base_dir = Path(features_base_dir)

    dataset_name = raw_path.parent.name          # e.g. "yahoo_S&P_500"
    out_dir = features_base_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = FEATURE_CONFIG

    print(f"Loading raw panel from {raw_path} ...")
    df = _load_long_panel(raw_path)
    df = _add_returns(df)

    print("Loading French factors ...")
    df = _add_market_factors(df, french_cache_dir)

    print(f"Panel: {df.shape[0]:,} rows, {df['Ticker'].nunique()} tickers")
    print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"Saving features to {out_dir}/\n")

    # Momentum
    print("Computing momentum ...")
    tmp = compute_momentum_features(df, cfg["mom_windows"])
    _save(tmp, [f"mom_{l}_{r}" for l, r in cfg["mom_windows"]], out_dir / "momentum.parquet")

    # Volatility
    print("Computing volatility ...")
    tmp = compute_volatility_features(df, cfg["vol_windows"])
    vol_cols = [f"vol_{h}" for h in cfg["vol_windows"]] + [f"downside_dev_{h}" for h in cfg["vol_windows"]]
    _save(tmp, vol_cols, out_dir / "volatility.parquet")

    # Liquidity
    print("Computing liquidity ...")
    tmp = compute_liquidity(df, cfg["liq_windows"])
    _save(tmp, [f"dollar_volume_{h}" for h in cfg["liq_windows"]], out_dir / "liquidity.parquet")

    # Extreme
    print("Computing extreme features ...")
    tmp = compute_extreme_features(df, cfg["high_windows"], cfg["max_windows"])
    ext_cols = [f"dist_{h}_high" for h in cfg["high_windows"]] + [f"max_{h}" for h in cfg["max_windows"]]
    _save(tmp, ext_cols, out_dir / "extreme.parquet")

    # Skewness
    print("Computing skewness ...")
    tmp = compute_skewness_features(df, cfg["skew_windows"])
    _save(tmp, [f"skew_{h}" for h in cfg["skew_windows"]], out_dir / "skewness.parquet")

    # Beta → also used as base for IVOL and idiosyncratic momentum
    print("Computing rolling beta ...")
    beta_tmp = _compute_beta(df, cfg["beta_window"])
    _save(beta_tmp, ["rolling_beta", "idiosyncratic"], out_dir / "beta.parquet")

    # IVOL (rolling std of idiosyncratic returns)
    print("Computing IVOL ...")
    tmp = compute_ivol(beta_tmp, cfg["ivol_windows"])
    _save(tmp, [f"ivol_{h}" for h in cfg["ivol_windows"]], out_dir / "ivol.parquet")

    # Idiosyncratic (alpha) momentum
    print("Computing idiosyncratic momentum ...")
    tmp = compute_idio_momentum(beta_tmp, cfg["idio_mom_windows"])
    _save(tmp, [f"idio_mom_{l}_{r}" for l, r in cfg["idio_mom_windows"]], out_dir / "idio_momentum.parquet")

    # Amihud illiquidity
    print("Computing Amihud illiquidity ...")
    tmp = compute_amihud(df, cfg["amihud_windows"])
    _save(tmp, [f"amihud_{h}" for h in cfg["amihud_windows"]], out_dir / "amihud.parquet")

    # Parkinson volatility
    print("Computing Parkinson volatility ...")
    tmp = compute_parkinson_vol(df, cfg["parkinson_windows"])
    _save(tmp, [f"parkinsons_vol_{h}" for h in cfg["parkinson_windows"]], out_dir / "parkinson_vol.parquet")

    # Volume surprise
    print("Computing volume surprise ...")
    tmp = compute_volume_surprise(df, cfg["vol_surprise_windows"])
    _save(tmp, [f"vol_surprise_{h}" for h in cfg["vol_surprise_windows"]], out_dir / "volume_surprise.parquet")

    # Return z-score (windows include 5 for short-term signal)
    print("Computing return z-score ...")
    tmp = compute_return_zscore(df, cfg["zscore_windows"])
    _save(tmp, [f"return_zscore_{h}" for h in cfg["zscore_windows"]], out_dir / "return_zscore.parquet")

    # Price-volume interaction (requires both zscore and vol_surprise columns)
    print("Computing price-volume interaction ...")
    windows = cfg["pv_interaction_windows"]
    tmp = compute_volume_surprise(df, windows)
    tmp = compute_return_zscore(tmp, windows)
    tmp = compute_price_volume_interaction(tmp, windows)
    _save(tmp, [f"pv_interaction_{h}" for h in windows], out_dir / "pv_interaction.parquet")

    # Price relative to moving average
    print("Computing price-to-MA ...")
    tmp = compute_price_to_ma(df, cfg["price_to_ma_windows"])
    _save(tmp, [f"price_to_ma_{h}" for h in cfg["price_to_ma_windows"]], out_dir / "price_to_ma.parquet")

    # Volatility ratio (short/long)
    print("Computing volatility ratio ...")
    tmp = compute_vol_ratio(df, cfg["vol_ratio_pairs"])
    _save(tmp, [f"vol_ratio_{s}_{l}" for s, l in cfg["vol_ratio_pairs"]], out_dir / "vol_ratio.parquet")

    print(f"\nDone. All features saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and cache features from raw price data.")
    parser.add_argument("--raw",    required=True,              help="Path to raw.parquet (wide format)")
    parser.add_argument("--french", required=True,              help="Path to Ken French data cache dir")
    parser.add_argument("--out",    default="../data/features",    help="Base output directory for features")
    args = parser.parse_args()
    compute_and_save(args.raw, args.french, args.out)

