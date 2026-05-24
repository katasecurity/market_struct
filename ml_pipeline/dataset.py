"""
ml_pipeline/dataset.py
Feature engineering and dataset preparation for the order-book ML pipeline.

Conventions:
- Method chaining: every mutating method returns self.
- Config injected via MLConfig dataclass.
- Type hints throughout (Python 3.10+).
- logging for structured logs; print for lightweight CLI progress.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from config import MLConfig

try:
    from config import MLConfig
except ModuleNotFoundError:
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class MLConfig:
        """Fallback config used when the real config module is absent."""

        target_horizon_min: int = 5
        return_threshold_bps: float = 2.0
        lags: list[int] = field(default_factory=lambda: [1, 2, 3])
        model_path: str = "models/candle_predictor.pkl"


logger = logging.getLogger(__name__)

# Only engineered features are used for training. Raw base columns
# (spread, imbalance, z_score, etc.) are excluded so that the feature set
# can be reproduced identically in CandlePredictor._apply_transformations()
# without a circular import. Exactly matches _FEATURE_COLUMNS in inference.py.
ALL_ENGINEERED_FEATURES: list[str] = [
    "imbalance_delta_1",
    "imbalance_delta_3",
    "imbalance_delta_5",
    "rel_spread",
    "imbalance_lag_1",
    "imbalance_lag_2",
    "imbalance_lag_3",
    "z_score_lag_1",
    "z_score_lag_2",
    "z_score_lag_3",
]


class MLDataLoader:
    """Chainable data-loading and feature-engineering class for the order-book ML pipeline.

    Typical usage:
        loader = (
            MLDataLoader(config)
            .load("data/processed/BTC_1min_features.parquet")
            .generate_target()
            .engineer_features()
        )
        X_train, X_test, y_train, y_test = loader.time_series_split()
    """

    def __init__(self, config: MLConfig) -> None:
        """Initialise the loader with an MLConfig instance."""
        self._config: MLConfig = config
        self._df: pd.DataFrame = pd.DataFrame()
        logger.debug("[dataset] MLDataLoader initialised with config: %s", config)

    def load(self, path: str | Path) -> "MLDataLoader":
        """Read a parquet file produced by OrderBookProcessor into self._df.

        Raises FileNotFoundError if path does not exist.
        """
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"[dataset] Parquet file not found: {resolved}")

        self._df = pd.read_parquet(resolved)
        n_rows = len(self._df)
        print(f"[dataset] Loaded {n_rows:,} rows from {resolved.name}")
        logger.info("[dataset] Loaded %d rows | columns: %s", n_rows, list(self._df.columns))
        return self

    def generate_target(
        self,
        horizon: int | None = None,
        threshold_bps: float | None = None,
    ) -> "MLDataLoader":
        """Generate a ternary directional target column ('target').

        Labels:
            1  (Bullish)  — future return exceeds +threshold
           -1  (Bearish)  — future return falls below -threshold
            0  (Neutral)  — future return within the band

        horizon defaults to config.target_horizon_min.
        threshold_bps defaults to config.return_threshold_bps.
        """
        self._validate_loaded()

        h: int = horizon if horizon is not None else self._config.target_horizon_min
        t_bps: float = threshold_bps if threshold_bps is not None else self._config.return_threshold_bps
        threshold: float = t_bps / 10_000.0

        mid: pd.Series = self._df["mid_price"]
        future_return: pd.Series = mid.shift(-h) / mid - 1.0

        target: pd.Series = pd.Series(0, index=self._df.index, dtype=np.int8)
        target[future_return > threshold] = 1
        target[future_return < -threshold] = -1

        self._df["target"] = target
        self._df.loc[future_return.isna(), "target"] = np.nan

        dist = self._df["target"].value_counts(dropna=True).sort_index().to_dict()
        print(f"[dataset] Target distribution: {dist}")
        logger.info("[dataset] Target distribution (horizon=%d, threshold=%.4f): %s", h, threshold, dist)
        return self

    def engineer_features(self) -> "MLDataLoader":
        """Add derived features in-place and drop rows with any remaining NaN.

        Features added:
            imbalance_delta_1/3/5  — period-over-period change in OBI
            rel_spread             — spread normalised by mid-price
            imbalance_lag_1/2/3    — lagged values of imbalance
            z_score_lag_1/2/3      — lagged values of z_score
        """
        self._validate_loaded()

        df = self._df

        for periods in (1, 3, 5):
            col = f"imbalance_delta_{periods}"
            df[col] = df["imbalance"].diff(periods)
            logger.debug("[dataset] Engineered %s", col)

        df["rel_spread"] = df["spread"] / df["mid_price"]
        logger.debug("[dataset] Engineered rel_spread")

        for lag in self._config.lags:
            col = f"imbalance_lag_{lag}"
            df[col] = df["imbalance"].shift(lag)
            logger.debug("[dataset] Engineered %s", col)

        for lag in self._config.lags:
            col = f"z_score_lag_{lag}"
            df[col] = df["z_score"].shift(lag)
            logger.debug("[dataset] Engineered %s", col)

        before = len(df)
        df.dropna(subset=self.get_feature_columns(), inplace=True)
        after = len(df)

        self._df = df
        print(f"[dataset] Feature engineering complete | dropped {before - after:,} NaN rows | {after:,} remaining")
        logger.info("[dataset] Rows before/after NaN drop: %d → %d", before, after)
        return self

    def get_feature_columns(self) -> list[str]:
        """Return the ordered list of all engineered feature column names.

        Safe to call before and after feature engineering.
        """
        if self._df.empty:
            return list(ALL_ENGINEERED_FEATURES)
        present = set(self._df.columns)
        return [c for c in ALL_ENGINEERED_FEATURES if c in present]

    def time_series_split(
        self,
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform a chronological (no-shuffle) train/test split.

        Args:
            test_size: Fraction of rows reserved for testing (tail of time series).

        Returns:
            X_train, X_test, y_train, y_test

        Raises:
            ValueError: If 'target' column is missing or test_size is out of range.
        """
        self._validate_loaded()

        if "target" not in self._df.columns:
            raise ValueError("[dataset] 'target' column not found. Call generate_target() first.")
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"[dataset] test_size must be in (0, 1), got {test_size}")

        df = self._df.dropna(subset=["target"]).copy()
        feature_cols = self.get_feature_columns()

        X: pd.DataFrame = df[feature_cols]
        y: pd.Series = df["target"].astype(np.int8)

        split_idx: int = int(len(df) * (1.0 - test_size))

        X_train = X.iloc[:split_idx]
        X_test  = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test  = y.iloc[split_idx:]

        print(f"[dataset] Train: {len(X_train):,} rows, Test: {len(X_test):,} rows (test_size={test_size:.0%})")
        logger.info(
            "[dataset] Chronological split — train=%d, test=%d, features=%d",
            len(X_train), len(X_test), len(feature_cols),
        )
        return X_train, X_test, y_train, y_test

    def _validate_loaded(self) -> None:
        """Raise RuntimeError if the dataframe has not been loaded yet."""
        if self._df.empty:
            raise RuntimeError("[dataset] No data loaded. Call load() before this method.")

    def __repr__(self) -> str:
        rows = len(self._df) if not self._df.empty else 0
        cols = list(self._df.columns) if not self._df.empty else []
        return f"MLDataLoader(rows={rows}, columns={cols}, config={self._config!r})"
