from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class AbstractCrossSectionalModel(ABC):
    """
    Interface for all cross-sectional return prediction models.

    A model receives a feature matrix X (observations × features) and a
    cross-sectional rank target y in [0, 1], then produces scalar predictions
    that can be ranked across the cross-section.

    Implement this interface to plug any model into WalkForwardBacktester.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in filenames and plot titles."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train on (X, y). Called once per walk-forward fold."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return scalar predictions aligned with X's row order."""

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Return feature importances as a Series (index = feature names).
        Returns None if the model does not support importance scores.
        Override in subclasses that do.
        """
        return None

    def get_params(self) -> dict:
        """Return hyperparameters for logging. Override in subclasses."""
        return {}
