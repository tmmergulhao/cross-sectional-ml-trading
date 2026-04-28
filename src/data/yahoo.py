from __future__ import annotations

import pandas as pd
import yfinance as yf

from .base import AbstractMarketDataLoader


class YahooFinanceLoader(AbstractMarketDataLoader):
    """
    Yahoo Finance implementation of the abstract market-data loader.
    """

    def normalize_tickers(self, tickers: list[str]) -> list[str]:
        # Yahoo Finance uses '-' instead of '.' in some tickers, e.g. BRK.B -> BRK-B
        return [ticker.replace(".", "-") for ticker in tickers]

    def download_raw(self) -> pd.DataFrame:
        tickers = self.normalize_tickers(self.request.tickers)

        raw = yf.download(
            tickers=tickers,
            start=self.request.start,
            end=self.request.end,
            auto_adjust=False,
            progress=True,
            group_by="column",
            threads=True,
        )

        if raw.empty:
            raise ValueError("Yahoo Finance returned an empty dataset.")

        return raw.sort_index()

    def extract_close_prices(self, raw: pd.DataFrame) -> pd.DataFrame:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded Yahoo data does not contain a 'Close' field.")

        close = raw["Close"].copy()
        close = close.sort_index()
        close = close.dropna(axis=1, how="all")

        if close.empty:
            raise ValueError("Extracted close-price panel is empty.")

        return close