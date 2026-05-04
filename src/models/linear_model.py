from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from models.base import AbstractCrossSectionalModel


class RidgeRankModel(AbstractCrossSectionalModel):
    """Ridge regression trained to predict cross-sectional rank targets."""

    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha = alpha
        self._model: Optional[Ridge] = None
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return "linear_regression"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self._model = Ridge(alpha=self._alpha, fit_intercept=True)
        self._model.fit(X.fillna(X.median()), y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X.fillna(X.median()))

    def get_feature_importance(self) -> Optional[pd.Series]:
        if self._model is None or not self._feature_names:
            return None
        return pd.Series(
            np.abs(self._model.coef_), index=self._feature_names
        ).sort_values(ascending=False)

    def get_params(self) -> dict:
        return {"alpha": self._alpha}
