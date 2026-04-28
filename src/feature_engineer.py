import pandas as pd
import numpy as np


def compute_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(close / close.shift(1))
    return log_returns.dropna(how="all")

def compute_momentum_features(df: pd.DataFrame, windows: list[tuple[int,int]]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    g = df.groupby("Ticker")["Adj Close"]

    for l, r in windows:
        if l >= r:
            raise ValueError(f"Expected l < r, got l={l}, r={r}")
        df[f"mom_{l}_{r}"] = g.shift(l) / g.shift(r) - 1
    return df

def compute_volatility_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    ret = df.groupby("Ticker")["Adj Close"].pct_change()
    neg_sq = np.minimum(ret, 0) ** 2
    for h in windows:
        df[f"vol_{h}"] = ret.groupby(df["Ticker"]).transform(lambda s: s.shift(1).rolling(h).std())
        df[f"downside_dev_{h}"] = neg_sq.groupby(df["Ticker"]).transform(
            lambda s: np.sqrt(s.shift(1).rolling(h).mean()))
    return df

def compute_liquidity(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    d_vol = df["Volume"] * df["Adj Close"]
    for h in windows:
        df[f"dollar_volume_{h}"] = (
            d_vol.groupby(df["Ticker"])
                .transform(lambda s: s.shift(1).rolling(h).mean())
        )
    return df

def compute_extreme_features(
    df: pd.DataFrame,
    high_windows: list[int] = [252],
    max_windows: list[int] = [21],
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])

    g_price = df.groupby("Ticker")["Adj Close"]
    prev_close = g_price.shift(1)

    for h in high_windows:
        rolling_high = g_price.transform(lambda s: s.shift(1).rolling(h).max())
        df[f"dist_{h}_high"] = prev_close / rolling_high - 1

    ret = g_price.pct_change()
    for h in max_windows:
        df[f"max_{h}"] = ret.groupby(df["Ticker"]).transform(
            lambda s: s.shift(1).rolling(h).max()
        )

    return df