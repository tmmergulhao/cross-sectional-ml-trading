from __future__ import annotations

import pandas as pd

from src.data.models import CacheConfig, DataRequest
from src.data.yahoo import YahooFinanceLoader

#TODO: Improve a general ticker loader
def load_test_tickers() -> list[str]:
    return [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META",
        "NVDA", "JPM", "GS", "XOM", "CVX",
        "UNH", "LLY", "PG", "KO", "PEP"
    ]

def main() -> None:
    tickers = load_test_tickers()  # start small while testing

    request = DataRequest(
        tickers=tickers,
        start="2018-01-01",
        end="2026-04-01",
        auto_adjust=True,
    )

    loader = YahooFinanceLoader(request)
    bundle = loader.build_bundle()

    cache = CacheConfig(
        raw_path="data/raw/yahoo_raw.parquet",
        close_path="data/processed/close.parquet",
        returns_path="data/processed/returns.parquet",
        log_returns_path="data/processed/log_returns.parquet",
    )

    loader.save_bundle(bundle, cache)

    print("Close shape:", bundle.close.shape)
    print("Returns shape:", bundle.returns.shape if bundle.returns is not None else None)
    print("Missing fraction total:", bundle.quality_report.missing_fraction_total)
    print("Notes:", bundle.quality_report.notes)


if __name__ == "__main__":
    main()