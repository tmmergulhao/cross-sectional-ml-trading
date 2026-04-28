"""
Ken French Data Library loader.

Downloads factor returns and portfolio sorts directly from:
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

All data are returned in decimal form (French publishes as percentages;
we divide by 100 here so they are consistent with yfinance returns).

Usage example
-------------
from src.data.french import FrenchDataLoader

loader = FrenchDataLoader(cache_dir="data/raw/french")

# Most useful for cross-sectional work:
ff5 = loader.get_ff5_factors(freq="daily")
mom = loader.get_momentum_factor(freq="daily")

# Factor zoo reference / attribution
ff3 = loader.get_ff3_factors(freq="monthly")
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# Maps (dataset_key, freq) -> filename stem used in the French FTP URL.
# Full URL: BASE_URL + stem + "_CSV.zip"
BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

DATASET_REGISTRY: dict[tuple[str, str], str] = {
    # Fama/French 3 Factors
    ("ff3", "monthly"): "F-F_Research_Data_Factors",
    ("ff3", "daily"):   "F-F_Research_Data_Factors_daily",
    ("ff3", "weekly"):  "F-F_Research_Data_Factors_weekly",
    # Fama/French 5 Factors (2x3)
    ("ff5", "monthly"): "F-F_Research_Data_5_Factors_2x3",
    ("ff5", "daily"):   "F-F_Research_Data_5_Factors_2x3_daily",
    # Momentum Factor
    ("mom", "monthly"): "F-F_Momentum_Factor",
    ("mom", "daily"):   "F-F_Momentum_Factor_daily",
    # 25 Portfolios formed on Size and Book-to-Market (Value-Weighted)
    ("25_size_bm_vw", "monthly"): "25_Portfolios_5x5",
    ("25_size_bm_vw", "daily"):   "25_Portfolios_5x5_Daily",
    # 10 Portfolios formed on Prior Return (Momentum deciles)
    ("10_mom_vw", "monthly"): "10_Portfolios_Prior_12_2",
    ("10_mom_vw", "daily"):   "10_Portfolios_Prior_12_2_Daily",
    # 100 Portfolios formed on Size and Book-to-Market
    ("100_size_bm_vw", "monthly"): "100_Portfolios_10x10",
}

# Human-readable descriptions for each dataset key
DATASET_DESCRIPTIONS: dict[str, str] = {
    "ff3": (
        "Fama/French 3 Factors: Mkt-RF (market excess return), "
        "SMB (Small Minus Big), HML (High Minus Low book-to-market), RF (risk-free rate). "
        "The canonical benchmark for equity factor attribution since Fama & French (1993)."
    ),
    "ff5": (
        "Fama/French 5 Factors: adds RMW (Robust Minus Weak profitability) and "
        "CMA (Conservative Minus Aggressive investment) to the 3-factor model. "
        "Fama & French (2015) show these capture most CAPM/3F anomalies."
    ),
    "mom": (
        "Momentum Factor (Mom): prior 2-12 month winner-minus-loser return. "
        "From Carhart (1997). Combines with FF5 to give the 6-factor model "
        "used in most modern performance attribution."
    ),
    "25_size_bm_vw": (
        "25 Portfolios formed on Size and Book-to-Market (5x5 sort). "
        "Value-weighted returns. Classic test portfolios for cross-sectional models."
    ),
    "10_mom_vw": (
        "10 Portfolios formed on prior 2-12 month return (momentum deciles). "
        "Value-weighted. Shows the full cross-sectional spread of momentum returns."
    ),
    "100_size_bm_vw": (
        "100 Portfolios formed on Size and Book-to-Market (10x10 sort). "
        "Higher granularity for testing asset pricing models."
    ),
}


# ---------------------------------------------------------------------------
# Parser for French CSV format
# ---------------------------------------------------------------------------

def _detect_date_format(date_str: str) -> str:
    """Infer date format from the French date integer string."""
    s = date_str.strip()
    if len(s) == 8:
        return "%Y%m%d"   # daily: YYYYMMDD
    elif len(s) == 6:
        return "%Y%m"     # monthly: YYYYMM
    elif len(s) == 4:
        return "%Y"       # annual: YYYY
    raise ValueError(f"Unrecognised date string: {s!r}")


def _parse_french_csv(text: str) -> dict[str, pd.DataFrame]:
    """
    Parse a French data library CSV file (already decompressed to a string).

    The Ken French format is:
      - Description text (any number of prose lines)
      - Optional blank lines
      - Column header row that ALWAYS starts with a comma: ',Mkt-RF,SMB,...'
      - Data rows where the first field is an integer date (YYYYMMDD / YYYYMM / YYYY)
      - Possibly multiple such blocks (monthly, annual, EW vs VW, etc.)

    Returns a dict with keys:
      "primary"  — the first data table in the file
      "annual"   — the annual table (if present)
      <section_label> — other labelled sections (e.g. "Equal Weight Returns")

    All numeric values are divided by 100 (French publishes as percentages).
    Missing values (-99.99, -999, -9.99, -99) are replaced with NaN.
    """
    MISSING_VALUES = {-99.99, -999.0, -9.99, -99.0}

    lines = text.splitlines()
    results: dict[str, pd.DataFrame] = {}
    table_count = 0

    # State machine
    header_cols: list[str] = []
    data_rows: list[list[str]] = []
    in_data = False
    pending_label: str | None = None  # most recent non-blank, non-data prose line

    def _label_for_count(count: int, pending: str | None) -> str:
        if count == 0:
            return "primary"
        if pending:
            lbl = pending.lower()
            if "annual" in lbl:
                return "annual"
            return pending.strip()
        return f"section_{count}"

    def _flush(cols: list[str], rows: list[list[str]], label: str) -> None:
        if not rows or not cols:
            return
        try:
            # Header starts with empty field → first data column is the date;
            # prepend a synthetic 'date_raw' name.
            col_names = ["date_raw"] + [c for c in cols if c]  # drop the leading ""
            # If col alignment is off, pad or truncate
            n_data_cols = len(rows[0])
            if len(col_names) < n_data_cols:
                col_names += [f"col_{i}" for i in range(len(col_names), n_data_cols)]
            col_names = col_names[:n_data_cols]

            df = pd.DataFrame(rows, columns=col_names)
            date_col = "date_raw"
            fmt = _detect_date_format(df[date_col].iloc[0])
            if fmt == "%Y%m":
                df.index = pd.PeriodIndex(
                    df[date_col].str.strip(), freq="M"
                ).to_timestamp("M")
            elif fmt == "%Y":
                df.index = pd.PeriodIndex(
                    df[date_col].str.strip(), freq="Y"
                ).to_timestamp("Y")
            else:
                df.index = pd.to_datetime(df[date_col].str.strip(), format=fmt)
            df.index.name = "date"
            df = df.drop(columns=[date_col])
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.apply(lambda col: col.where(~col.isin(MISSING_VALUES)))
            df = df / 100.0
            results[label] = df
        except Exception:
            pass  # skip malformed sections

    for raw_line in lines:
        line = raw_line.strip()

        # ── Blank line ──────────────────────────────────────────────────────
        if not line:
            if in_data and data_rows:
                lbl = _label_for_count(table_count, pending_label)
                _flush(header_cols, data_rows, lbl)
                table_count += 1
                data_rows = []
                header_cols = []
                in_data = False
                pending_label = None
            continue

        parts = [p.strip() for p in line.split(",")]
        first = parts[0]

        # ── Data row ────────────────────────────────────────────────────────
        if first.isdigit():
            if in_data:
                data_rows.append(parts)
            # If we see data without a header (shouldn't happen in valid files),
            # just skip.
            continue

        # ── Column header row: starts with empty first field ────────────────
        if first == "" and len(parts) > 1:
            if in_data and data_rows:
                # A new header encountered while collecting data — flush first.
                lbl = _label_for_count(table_count, pending_label)
                _flush(header_cols, data_rows, lbl)
                table_count += 1
                data_rows = []
                pending_label = None
            header_cols = parts
            in_data = True
            continue

        # ── Prose / section label line ──────────────────────────────────────
        if in_data and data_rows:
            # Data block ended by a non-data line (e.g. "Annual Factors: ...")
            lbl = _label_for_count(table_count, pending_label)
            _flush(header_cols, data_rows, lbl)
            table_count += 1
            data_rows = []
            header_cols = []
            in_data = False
        pending_label = line  # remember as candidate label for the next block

    # Flush any trailing section
    if in_data and data_rows:
        lbl = _label_for_count(table_count, pending_label)
        _flush(header_cols, data_rows, lbl)

    return results


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------

class FrenchDataLoader:
    """
    Downloads and parses datasets from the Ken French Data Library.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for caching downloaded ZIP files. Defaults to "data/raw/french".
        Set to None to disable caching (always re-download).
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = "data/raw/french",
        timeout: int = 30,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Low-level download + parse
    # ------------------------------------------------------------------

    def _zip_path(self, stem: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{stem}_CSV.zip"

    def _download_zip(self, stem: str) -> bytes:
        """Download the ZIP bytes, using cache if available."""
        zip_path = self._zip_path(stem)
        if zip_path is not None and zip_path.exists():
            return zip_path.read_bytes()

        url = BASE_URL + stem + "_CSV.zip"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.content

        if zip_path is not None:
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(data)

        return data

    def _read_csv_from_zip(self, zip_bytes: bytes) -> str:
        """Extract the first CSV file from a ZIP archive and return its text."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("No CSV file found inside ZIP archive.")
            with zf.open(csv_names[0]) as f:
                return f.read().decode("latin-1")

    def load_dataset(
        self, dataset_key: str, freq: str = "daily"
    ) -> dict[str, pd.DataFrame]:
        """
        Download and parse a named dataset.

        Parameters
        ----------
        dataset_key : str
            One of: "ff3", "ff5", "mom", "25_size_bm_vw", "10_mom_vw", "100_size_bm_vw"
        freq : str
            "daily", "monthly", or "weekly" (availability varies by dataset).

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: "primary" (main frequency table), "annual" (if present), others.
            Values are DataFrames indexed by date, values in decimal form.
        """
        key = (dataset_key, freq)
        if key not in DATASET_REGISTRY:
            available = [k for k in DATASET_REGISTRY if k[0] == dataset_key]
            raise ValueError(
                f"Dataset ({dataset_key!r}, {freq!r}) not in registry. "
                f"Available frequencies for {dataset_key!r}: "
                f"{[f for _, f in available]}"
            )
        stem = DATASET_REGISTRY[key]
        zip_bytes = self._download_zip(stem)
        text = self._read_csv_from_zip(zip_bytes)
        return _parse_french_csv(text)

    def _primary(self, dataset_key: str, freq: str) -> pd.DataFrame:
        """Return only the primary (first) table from a dataset."""
        tables = self.load_dataset(dataset_key, freq)
        # Prefer "primary" key; fall back to first key
        if "primary" in tables:
            return tables["primary"]
        return next(iter(tables.values()))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_ff3_factors(self, freq: str = "daily") -> pd.DataFrame:
        """
        Fama/French 3 Factors: Mkt-RF, SMB, HML, RF.

        The workhorse benchmark. Use this to:
        - Estimate factor-adjusted alpha (Jensen's alpha in a 3-factor world)
        - Build a risk-free-rate series (RF column)
        - Attribute strategy performance to market / size / value tilts
        """
        return self._primary("ff3", freq)

    def get_ff5_factors(self, freq: str = "daily") -> pd.DataFrame:
        """
        Fama/French 5 Factors: Mkt-RF, SMB, HML, RMW, CMA, RF.

        The modern baseline for equity factor research. Adds:
        - RMW: profitability (robust - weak operating profitability)
        - CMA: investment (conservative - aggressive asset growth)

        **Most relevant for cross-sectional return prediction** when building
        features: CMA and RMW correspond to the investment and quality signals
        that are core predictors of the cross-section of returns.
        """
        return self._primary("ff5", freq)

    def get_momentum_factor(self, freq: str = "daily") -> pd.DataFrame:
        """
        Carhart Momentum Factor (Mom): prior 2-12 month winner minus loser.

        Combined with FF5 gives the 6-factor model that is the standard
        attribution framework in modern equity research. Momentum is also
        one of the most robust predictors in the cross-section of returns.
        """
        return self._primary("mom", freq)

    def get_ff5_plus_mom(self, freq: str = "daily") -> pd.DataFrame:
        """
        Combined FF5 + Momentum table (6-factor model).

        This is the most complete set of canonical factors for cross-sectional
        return prediction. Covers:
          Mkt-RF  — market beta (systematic risk)
          SMB     — size (small-cap premium)
          HML     — value (book-to-market)
          RMW     — quality/profitability
          CMA     — investment conservatism
          Mom     — price momentum
          RF      — risk-free rate (for excess-return calculations)

        For cross-sectional research: use these 5 factors (excluding Mkt-RF and RF)
        as your primary feature set or at minimum as benchmark controls.
        """
        ff5 = self.get_ff5_factors(freq)
        mom = self.get_momentum_factor(freq)
        mom_col = mom.drop(columns=["RF"], errors="ignore")
        return ff5.join(mom_col, how="inner")

    def get_size_bm_portfolios(self, freq: str = "monthly") -> pd.DataFrame:
        """
        25 Value-Weighted Portfolios formed on Size x Book-to-Market (5x5).

        The classic test assets for evaluating cross-sectional models.
        A good model should explain the spread in average returns across
        these 25 portfolios (the "value premium" and "size premium" in
        2-dimensional form).
        """
        return self._primary("25_size_bm_vw", freq)

    def get_momentum_portfolios(self, freq: str = "monthly") -> pd.DataFrame:
        """
        10 Value-Weighted Portfolios formed on Prior 2-12 Month Return.

        Shows the raw momentum effect: decile 10 (past winners) minus
        decile 1 (past losers) is the long-run average return spread.
        Use this to validate momentum signal construction.
        """
        return self._primary("10_mom_vw", freq)

    @staticmethod
    def list_datasets() -> pd.DataFrame:
        """Return a DataFrame listing all available datasets and their descriptions."""
        rows = []
        for (key, freq), stem in DATASET_REGISTRY.items():
            rows.append({
                "dataset_key": key,
                "freq": freq,
                "url_stem": stem,
                "description": DATASET_DESCRIPTIONS.get(key, ""),
            })
        return pd.DataFrame(rows).sort_values(["dataset_key", "freq"])
