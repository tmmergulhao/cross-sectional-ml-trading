import pandas as pd
import numpy as np

def compute_skewness_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    
    ret = df.groupby("Ticker")["Adj Close"].pct_change()
    
    for h in windows:
        df[f"skew_{h}"] = (
            ret.groupby(df["Ticker"])
               .transform(lambda s: s.shift(1).rolling(h).skew())
        )
    return df

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

def create_rolling_beta(df: pd.DataFrame, mkt_rf: pd.Series, window: int) -> pd.DataFrame:
    """
    df: Data frame with tickers, prices, etc
    mkt_rf: Market minus risk-free (downloaded from French-Fama interface)
    """
    df = df.copy()
    #raise error if the index of df and mkt_rf do not match
    if not df.index.equals(mkt_rf.index):
        raise ValueError("Expected df and mkt_rf to have the same index")
    
    df['mkt_rf'] = mkt_rf
    
    def rolling_beta(g: pd.DataFrame, window: int) -> pd.Series:
        """Rolling OLS beta: cov(r_i - rf, mkt_rf) / var(mkt_rf)."""
        cov = g['returns_minus_RF'].rolling(window).cov(g['mkt_rf'])
        var = g['mkt_rf'].rolling(window).var()
        return cov / var

    #compute the rolling beta and shift to avoid look-ahead
    df['rolling_beta'] = (
        df.groupby('Ticker', group_keys=False)
        .apply(lambda g: rolling_beta(g, window))
    )
    df['rolling_beta'] = (
        df.groupby('Ticker')['rolling_beta']
        .shift(1)
    )

    #save the idiosyncratic noise
    df['idiosyncratic'] = df['returns_minus_RF'] - df['rolling_beta'] * df['mkt_rf']
    return df

def compute_amihud(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    ret = df.groupby("Ticker")["Adj Close"].pct_change()
    dollar_vol = (df["Volume"] * df["Adj Close"]).replace(0, np.nan)
    illiq = ret.abs() / dollar_vol
    
    for h in windows:
        df[f"amihud_{h}"] = (
            illiq.groupby(df["Ticker"])
                 .transform(lambda s: s.shift(1).rolling(h).mean())
        )
    return df

def compute_parkinson_vol(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    log_hl = np.log(df["High"] / df["Low"]) ** 2
    parkinson = log_hl / (4 * np.log(2))
    for h in windows:
        df[f"parkinsons_vol_{h}"] = (
            parkinson.groupby(df["Ticker"])
                     .transform(lambda s: (s.shift(1).rolling(h).mean() ** 0.5))
        )
    return df

def compute_volume_surprise(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    for h in windows:
        avg_vol = (
            df.groupby("Ticker")["Volume"]
              .transform(lambda s: s.shift(1).rolling(h).mean())
        )
        df[f"vol_surprise_{h}"] = df["Volume"].shift(1) / avg_vol - 1
    return df

def compute_return_zscore(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    ret = df.groupby("Ticker")["Adj Close"].pct_change()
    
    for h in windows:
        rolling_mean = ret.groupby(df["Ticker"]).transform(
            lambda s: s.shift(2).rolling(h).mean()
        )
        rolling_std = ret.groupby(df["Ticker"]).transform(
            lambda s: s.shift(2).rolling(h).std()
        )
        df[f"return_zscore_{h}"] = (
            ret.groupby(df["Ticker"]).transform(lambda s: s.shift(1)) - rolling_mean
        ) / rolling_std
    return df

def compute_price_volume_interaction(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for h in windows:
        df[f"pv_interaction_{h}"] = (
            df[f"return_zscore_{h}"] * df[f"vol_surprise_{h}"]
        )
    return df

def compute_ivol(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Rolling std of idiosyncratic returns (Ang et al. 2006). Requires 'idiosyncratic' column."""
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    for h in windows:
        df[f"ivol_{h}"] = (
            df.groupby("Ticker")["idiosyncratic"]
            .transform(lambda s: s.shift(1).rolling(h).std())
        )
    return df

def compute_price_to_ma(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Price relative to SMA: prev_close / rolling_mean(close, N) - 1."""
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    g = df.groupby("Ticker")["Adj Close"]
    for h in windows:
        sma = g.transform(lambda s: s.shift(1).rolling(h).mean())
        df[f"price_to_ma_{h}"] = g.shift(1) / sma - 1
    return df

def compute_vol_ratio(df: pd.DataFrame, window_pairs: list[tuple]) -> pd.DataFrame:
    """Short/long realized volatility ratio: vol_short / vol_long."""
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    ret = df.groupby("Ticker")["Adj Close"].pct_change()
    for h_short, h_long in window_pairs:
        vol_s = ret.groupby(df["Ticker"]).transform(
            lambda s: s.shift(1).rolling(h_short).std()
        )
        vol_l = ret.groupby(df["Ticker"]).transform(
            lambda s: s.shift(1).rolling(h_long).std()
        )
        df[f"vol_ratio_{h_short}_{h_long}"] = vol_s / vol_l
    return df

def compute_idio_momentum(df: pd.DataFrame, windows: list[tuple]) -> pd.DataFrame:
    """
    Cumulative idiosyncratic (alpha) return over [L, R] trading-day window.
    Requires 'idiosyncratic' column from rolling beta.
    Equivalent to residual momentum: strips out market beta contribution.
    """
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    g = df.groupby("Ticker")["idiosyncratic"]
    for L, R in windows:
        df[f"idio_mom_{L}_{R}"] = g.transform(
            lambda s: s.shift(L).rolling(R - L).sum()
        )
    return df