from __future__ import annotations

import pandas as pd

from .models import DataQualityReport


def validate_price_panel(close: pd.DataFrame) -> DataQualityReport:
    if close.empty:
        raise ValueError("Price panel is empty.")

    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("Price panel index must be a DatetimeIndex.")

    missing_fraction_by_asset = close.isna().mean(axis=0)
    missing_fraction_total = float(close.isna().mean().mean())

    notes: list[str] = []
    if missing_fraction_total > 0.05:
        notes.append("Overall missingness exceeds 5%.")

    highly_missing = missing_fraction_by_asset[missing_fraction_by_asset > 0.2]
    if not highly_missing.empty:
        notes.append(
            f"{len(highly_missing)} assets have more than 20% missing observations."
        )

    return DataQualityReport(
        n_rows=close.shape[0],
        n_assets=close.shape[1],
        missing_fraction_by_asset=missing_fraction_by_asset,
        missing_fraction_total=missing_fraction_total,
        first_date=close.index.min(),
        last_date=close.index.max(),
        notes=notes,
    )