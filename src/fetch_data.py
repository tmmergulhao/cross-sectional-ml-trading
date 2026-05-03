import pandas as pd

import sys
sys.path.insert(0, '..')
from src.data.yahoo import YahooFinanceLoader
from src.data.models import DataRequest, CacheConfig
from src.data.transforms import compute_log_returns
import requests
import numpy as np


def load_sp500_tickers() -> list[str]:
    URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    html = requests.get(URL, headers=headers, timeout=30).text
    tickers = pd.read_html(html)[0]
    return tickers['Symbol'].to_list()

def fetch_raw_data(tickers_list: list[str]) -> pd.DataFrame:

    dr = DataRequest(
        tickers=tickers_list,
        start='2001-01-01',
        end='2026-03-31',
        auto_adjust=True
        )
    
    cc = CacheConfig(raw_path="../data/raw/yahoo_S&P_500/raw.parquet")

    data_loader = YahooFinanceLoader(dr)
    bundle = data_loader.build_bundle()
    data_loader.save_bundle(bundle, cc)
    return bundle

if __name__ == "__main__":
    tickers = load_sp500_tickers()
    fetch_raw_data(tickers)