from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DataRequest:
    tickers: list[str]
    start: str
    end: str
    fields: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
    auto_adjust: bool = True


@dataclass
class DataQualityReport:
    n_rows: int
    n_assets: int
    missing_fraction_by_asset: pd.Series
    missing_fraction_total: float
    first_date: Optional[pd.Timestamp] = None
    last_date: Optional[pd.Timestamp] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class MarketDataBundle:
    raw: pd.DataFrame
    close: pd.DataFrame
    returns: Optional[pd.DataFrame] = None
    log_returns: Optional[pd.DataFrame] = None
    quality_report: Optional[DataQualityReport] = None


@dataclass
class CacheConfig:
    raw_path: Optional[Path | str] = None
    close_path: Optional[Path | str] = None
    returns_path: Optional[Path | str] = None
    log_returns_path: Optional[Path | str] = None

    def __post_init__(self) -> None:
        if self.raw_path is not None:
            self.raw_path = Path(self.raw_path)
        if self.close_path is not None:
            self.close_path = Path(self.close_path)
        if self.returns_path is not None:
            self.returns_path = Path(self.returns_path)
        if self.log_returns_path is not None:
            self.log_returns_path = Path(self.log_returns_path)