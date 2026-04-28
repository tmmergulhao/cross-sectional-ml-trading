from __future__ import annotations

import numpy as np
import pandas as pd


def compute_simple_returns(close: pd.DataFrame) -> pd.DataFrame:
    returns = close.pct_change()
    return returns.dropna(how="all")


def compute_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(close / close.shift(1))
    return log_returns.dropna(how="all")


def drop_all_nan_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how="all")


def sort_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_index()