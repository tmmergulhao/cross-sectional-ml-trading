from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from models.base import AbstractCrossSectionalModel


class XGBoostRankModel(AbstractCrossSectionalModel):
    """XGBoost regressor trained to predict cross-sectional rank targets."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.2,
        min_child_weight: int = 100,
        gamma: float = 0.5,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self._params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            tree_method="hist",
            random_state=random_state,
            missing=np.nan,
            n_jobs=n_jobs,
        )
        self._model: Optional[xgb.XGBRegressor] = None

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model = xgb.XGBRegressor(**self._params)
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X)

    def get_feature_importance(self) -> Optional[pd.Series]:
        if self._model is None:
            return None
        scores = self._model.get_booster().get_fscore()
        if not scores:
            return None
        return pd.Series(scores).sort_values(ascending=False)

    def get_params(self) -> dict:
        return self._params.copy()
