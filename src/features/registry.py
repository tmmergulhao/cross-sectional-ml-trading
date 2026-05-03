from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    columns: list[str]
    file: str

    def load(self, base_dir: str | Path) -> pd.DataFrame:
        return pd.read_parquet(Path(base_dir) / self.file)[self.columns]


REGISTRY: list[FeatureGroup] = [
    FeatureGroup(
        "momentum",
        ["mom_1_5", "mom_1_21", "mom_21_126", "mom_21_252", "mom_126_252", "mom_252_756"],
        "momentum.parquet",
    ),
    FeatureGroup(
        "volatility",
        ["vol_21", "vol_63", "vol_252", "downside_dev_21", "downside_dev_63", "downside_dev_252"],
        "volatility.parquet",
    ),
    FeatureGroup(
        "liquidity",
        ["dollar_volume_21", "dollar_volume_63"],
        "liquidity.parquet",
    ),
    FeatureGroup(
        "extreme",
        ["dist_252_high", "max_21"],
        "extreme.parquet",
    ),
    FeatureGroup(
        "skewness",
        ["skew_21", "skew_63", "skew_252"],
        "skewness.parquet",
    ),
    FeatureGroup(
        "beta",
        ["rolling_beta", "idiosyncratic"],
        "beta.parquet",
    ),
    FeatureGroup(
        "amihud",
        ["amihud_21", "amihud_63", "amihud_252"],
        "amihud.parquet",
    ),
    FeatureGroup(
        "parkinson_vol",
        ["parkinsons_vol_21", "parkinsons_vol_63", "parkinsons_vol_252"],
        "parkinson_vol.parquet",
    ),
    FeatureGroup(
        "vol_surprise",
        ["vol_surprise_21", "vol_surprise_63", "vol_surprise_252"],
        "volume_surprise.parquet",
    ),
    FeatureGroup(
        "return_zscore",
        ["return_zscore_5", "return_zscore_21", "return_zscore_63", "return_zscore_252"],
        "return_zscore.parquet",
    ),
    FeatureGroup(
        "pv_interaction",
        ["pv_interaction_21", "pv_interaction_63", "pv_interaction_252"],
        "pv_interaction.parquet",
    ),
    FeatureGroup(
        "ivol",
        ["ivol_21", "ivol_63", "ivol_252"],
        "ivol.parquet",
    ),
    FeatureGroup(
        "price_to_ma",
        ["price_to_ma_5", "price_to_ma_20", "price_to_ma_63", "price_to_ma_200"],
        "price_to_ma.parquet",
    ),
    FeatureGroup(
        "idio_momentum",
        ["idio_mom_21_252", "idio_mom_21_126"],
        "idio_momentum.parquet",
    ),
    FeatureGroup(
        "vol_ratio",
        ["vol_ratio_5_21", "vol_ratio_10_63", "vol_ratio_21_252"],
        "vol_ratio.parquet",
    ),
]

_BY_NAME: dict[str, FeatureGroup] = {g.name: g for g in REGISTRY}


class FeatureStore:
    """Loads feature groups from the parquet files in a dataset directory.

    Usage
    -----
    store = FeatureStore("data/features/yahoo_S&P_500")

    store.describe()                          # DataFrame summary — renders as table in Jupyter
    store.load()                              # all features joined into one DataFrame
    store.load(["momentum", "volatility"])    # specific groups only
    """

    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)

    def describe(self) -> pd.DataFrame:
        """Return a summary DataFrame with one row per feature group."""
        rows = [
            {
                "group":      g.name,
                "n_features": len(g.columns),
                "file":       g.file,
                "features":   ", ".join(g.columns),
            }
            for g in REGISTRY
        ]
        return pd.DataFrame(rows).set_index("group")

    def load(self, groups: list[str] | None = None) -> pd.DataFrame:
        """Load and join feature groups into a single (Date, Ticker) DataFrame.

        Parameters
        ----------
        groups : list of group names to load, or None to load all groups.
        """
        if groups is not None:
            unknown = set(groups) - _BY_NAME.keys()
            if unknown:
                raise ValueError(f"Unknown groups: {unknown}. Available: {list(_BY_NAME)}")
            targets = [_BY_NAME[name] for name in groups]
        else:
            targets = REGISTRY

        return pd.concat([g.load(self._base) for g in targets], axis=1)

    def __repr__(self) -> str:
        return f"FeatureStore('{self._base}')\n\n" + self.describe().to_string()


# ---------------------------------------------------------------------------
# Derived views — keep for backward compatibility and direct dict access
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {g.name: g.columns for g in REGISTRY}
FEATURE_FILES  = {g.name: g.file    for g in REGISTRY}
ALL_FEATURES   = [col for g in REGISTRY for col in g.columns]
FEAT_TO_GROUP  = {col: g.name for g in REGISTRY for col in g.columns}
