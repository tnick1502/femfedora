"""Input contract for the UCI CTG feature preprocessing artifact."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


CTG_FEATURES = (
    "LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV",
    "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean",
    "Median", "Variance", "Tendency",
)


class CTGInputGuard(TransformerMixin, BaseEstimator):
    """Reject inputs unsupported by the complete-case UCI training sample.

    The official training source has no missing features. Median imputation or
    missingness indicators fitted on it would give an unvalidated behavior for
    new missing values, so this transformer fails before scaling instead.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self._checked_frame(X)
        self.feature_names_in_ = np.asarray(CTG_FEATURES, dtype=object)
        self.n_features_in_ = len(CTG_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._checked_frame(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(CTG_FEATURES, dtype=object)

    @staticmethod
    def _checked_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CTG input must be a pandas DataFrame with named features")
        if X.columns.has_duplicates:
            raise ValueError("CTG input has duplicate column names")
        missing_columns = sorted(set(CTG_FEATURES) - set(X.columns))
        extra_columns = sorted(set(X.columns) - set(CTG_FEATURES))
        if missing_columns or extra_columns:
            raise ValueError(
                f"CTG feature schema mismatch: missing={missing_columns}, extra={extra_columns}"
            )

        frame = X.loc[:, CTG_FEATURES].copy()
        for column in CTG_FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        missing_values = frame.isna().sum()
        if missing_values.any():
            affected = missing_values[missing_values > 0].to_dict()
            raise ValueError(f"CTG features contain missing values: {affected}")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("CTG features contain infinite values")
        if not frame["Tendency"].isin((-1, 0, 1)).all():
            raise ValueError("Tendency must be one of -1, 0, 1")

        numeric = frame.drop(columns="Tendency")
        if numeric.lt(0).any().any():
            raise ValueError("CTG numeric measurements must be non-negative")
        if frame[["ASTV", "ALTV"]].gt(100).any().any():
            raise ValueError("ASTV and ALTV must be percentages in [0, 100]")
        if not np.allclose(frame["Width"], frame["Max"] - frame["Min"], rtol=0, atol=1e-9):
            raise ValueError("CTG histogram Width must equal Max - Min")
        return frame
