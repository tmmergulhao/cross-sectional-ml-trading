from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .models import CacheConfig, DataRequest, MarketDataBundle
from .transforms import compute_log_returns, compute_simple_returns
from .validators import validate_price_panel


class AbstractMarketDataLoader(ABC):
    """
    Abstract interface for all market-data loaders.
    Concrete implementations should define how raw data is downloaded
    and how close prices are extracted.
    """

    def __init__(self, request: DataRequest) -> None:
        self.request = request

    @abstractmethod
    def normalize_tickers(self, tickers: list[str]) -> list[str]:
        """Normalize tickers for the specific provider."""
        raise NotImplementedError

    @abstractmethod
    def download_raw(self) -> pd.DataFrame:
        """Download raw provider-specific data."""
        raise NotImplementedError

    @abstractmethod
    def extract_close_prices(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Extract close-price panel as date x ticker."""
        raise NotImplementedError

    def build_bundle(self) -> MarketDataBundle:
        raw = self.download_raw()
        close = self.extract_close_prices(raw)
        quality_report = validate_price_panel(close)
        returns = compute_simple_returns(close)
        log_returns = compute_log_returns(close)

        return MarketDataBundle(
            raw=raw,
            close=close,
            returns=returns,
            log_returns=log_returns,
            quality_report=quality_report,
        )

    def save_bundle(self, bundle: MarketDataBundle, cache: CacheConfig) -> None:
        if cache.raw_path is not None:
            cache.raw_path.parent.mkdir(parents=True, exist_ok=True)
            bundle.raw.to_parquet(cache.raw_path)

        if cache.close_path is not None:
            cache.close_path.parent.mkdir(parents=True, exist_ok=True)
            bundle.close.to_parquet(cache.close_path)

        if cache.returns_path is not None and bundle.returns is not None:
            cache.returns_path.parent.mkdir(parents=True, exist_ok=True)
            bundle.returns.to_parquet(cache.returns_path)

        if cache.log_returns_path is not None and bundle.log_returns is not None:
            cache.log_returns_path.parent.mkdir(parents=True, exist_ok=True)
            bundle.log_returns.to_parquet(cache.log_returns_path)