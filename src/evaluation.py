from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.base import AbstractCrossSectionalModel


class WalkForwardBacktester:
    """
    Expanding-window walk-forward backtester for cross-sectional models.

    At each fold:
      - Training set: all data from the beginning up to train_end
      - Embargo: gap_days of data discarded after train_end
      - Test set: next test_months of trading data

    The panel must be in long format with a DatetimeIndex and a 'Ticker' column.
    """

    def __init__(
        self,
        model: AbstractCrossSectionalModel,
        initial_train_months: int = 84,   # 7 years
        test_months: int = 12,
        step_months: int = 1,
        embargo_days: int = 21,
        trading_days_per_month: int = 21,
    ) -> None:
        self.model = model
        self.initial_train_months = initial_train_months
        self.test_months = test_months
        self.step_months = step_months
        self.embargo_days = embargo_days
        self.trading_days_per_month = trading_days_per_month

    # ------------------------------------------------------------------
    def run(
        self,
        panel: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> list[dict]:
        """
        Execute the walk-forward loop.

        Parameters
        ----------
        panel : long-format DataFrame with DatetimeIndex, must have 'Ticker' column
        feature_cols : list of feature column names
        target_col : name of the target column (cross-sectional rank in [0,1])

        Returns
        -------
        List of fold result dicts with keys:
            fold, train_start, train_end, test_start, test_end,
            preds, actuals, index, tickers
        """
        dates = panel.index.unique().sort_values()
        test_days = self.test_months * self.trading_days_per_month
        step_days = self.step_months * self.trading_days_per_month

        start_idx = len(dates[dates < dates[0] + pd.DateOffset(months=self.initial_train_months)])
        fold_starts = range(start_idx, len(dates), step_days)

        results = []
        for fold, i in enumerate(tqdm(fold_starts, desc=f"Walk-forward [{self.model.name}]")):
            train_end = dates[i - self.embargo_days]
            test_start = dates[i]
            test_end_idx = min(i + test_days, len(dates) - 1)
            test_end = dates[test_end_idx]

            train = panel[panel.index <= train_end]
            test = panel[(panel.index >= test_start) & (panel.index <= test_end)]

            cols_needed = feature_cols + [target_col]
            train_clean = train[cols_needed].dropna(subset=[target_col])
            test_clean = test[cols_needed].dropna(subset=[target_col])

            if len(train_clean) == 0 or len(test_clean) == 0:
                continue

            X_train = train_clean[feature_cols]
            y_train = train_clean[target_col]
            X_test = test_clean[feature_cols]
            y_test = test_clean[target_col]

            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)

            tickers = (
                test_clean["Ticker"].values
                if "Ticker" in test_clean.columns
                else np.full(len(test_clean), "")
            )

            results.append({
                "fold": fold,
                "train_start": panel.index.min(),
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "preds": preds,
                "actuals": y_test.values,
                "index": X_test.index,
                "tickers": tickers,
            })

        return results

    # ------------------------------------------------------------------
    def summary(self, results: list[dict]) -> pd.DataFrame:
        """Return a DataFrame with one row per fold (train/test dates + fold index)."""
        rows = []
        for r in results:
            rows.append({
                "fold": r["fold"],
                "train_start": r["train_start"],
                "train_end": r["train_end"],
                "test_start": r["test_start"],
                "test_end": r["test_end"],
                "n_test_obs": len(r["preds"]),
            })
        return pd.DataFrame(rows)
