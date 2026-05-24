"""
ml_pipeline/inference.py
Real-time inference engine for next-candle direction prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import MLConfig
from ml_pipeline.model import ModelTrainer

logger = logging.getLogger(__name__)

# Exact feature column order matching MLDataLoader.get_feature_columns().
# Hardcoded here to avoid a circular import between inference and dataset.
_FEATURE_COLUMNS: list[str] = [
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

# Minimum rows required to guarantee valid lag/delta values in the final row.
_MIN_ROWS: int = 10


class CandlePredictor:
    """Real-time inference engine for next-candle direction prediction.

    Usage:
        predictor = CandlePredictor(MLConfig())
        result = predictor.predict_next_candle(recent_df)
        print(result['prediction'], result['confidence'])
    """

    LABEL_MAP: dict[int, str] = {1: "Bullish", -1: "Bearish", 0: "Neutral"}

    def __init__(
        self,
        config: MLConfig,
        model_path: str | Path | None = None,
    ) -> None:
        resolved_path: str | Path = model_path or config.model_path
        self._config = config
        self._trainer = ModelTrainer(config)
        self._trainer.load_model(resolved_path)
        logger.info("[inference] CandlePredictor ready. Model: %s", resolved_path)

    @staticmethod
    def _apply_transformations(recent_data: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature transformations as MLDataLoader.engineer_features.

        Returns only the last row so callers can call predict_next_candle
        incrementally without maintaining state.

        Args:
            recent_data: DataFrame with at least _MIN_ROWS rows containing
                         mid_price, spread, imbalance, z_score columns.

        Returns:
            Single-row DataFrame with columns in _FEATURE_COLUMNS order.
        """
        df = recent_data.copy()

        df["imbalance_delta_1"] = df["imbalance"].diff(1)
        df["imbalance_delta_3"] = df["imbalance"].diff(3)
        df["imbalance_delta_5"] = df["imbalance"].diff(5)

        df["rel_spread"] = df["spread"] / df["mid_price"]

        df["imbalance_lag_1"] = df["imbalance"].shift(1)
        df["imbalance_lag_2"] = df["imbalance"].shift(2)
        df["imbalance_lag_3"] = df["imbalance"].shift(3)

        df["z_score_lag_1"] = df["z_score"].shift(1)
        df["z_score_lag_2"] = df["z_score"].shift(2)
        df["z_score_lag_3"] = df["z_score"].shift(3)

        return df[_FEATURE_COLUMNS].iloc[[-1]]

    @staticmethod
    def get_feature_columns() -> list[str]:
        """Return the ordered list of engineered feature column names.

        Mirrors MLDataLoader.get_feature_columns() without importing from
        ml_pipeline.dataset to avoid circular imports.
        """
        return list(_FEATURE_COLUMNS)

    def predict_next_candle(self, recent_data: pd.DataFrame) -> dict:
        """Predict the direction of the next candle.

        Args:
            recent_data: DataFrame with at least 10 rows of raw order-book
                         features (mid_price, spread, imbalance, z_score).
                         Index should be a DatetimeIndex.

        Returns:
            {
                'prediction':       str,   # 'Bullish' | 'Bearish' | 'Neutral'
                'confidence':       float, # max class probability
                'raw_probabilities': dict, # label -> probability
                'timestamp':        str,   # ISO-8601 of last bar
                'above_threshold':  bool,  # confidence >= config.confidence_threshold
            }

        Raises:
            ValueError: When recent_data has fewer than _MIN_ROWS rows.
        """
        if len(recent_data) < _MIN_ROWS:
            raise ValueError(
                f"predict_next_candle requires at least {_MIN_ROWS} rows of data; "
                f"got {len(recent_data)}."
            )

        X = self._apply_transformations(recent_data)[_FEATURE_COLUMNS]

        proba_matrix = self._trainer.predict_proba(X)
        classes = self._trainer._model.classes_  # type: ignore[union-attr]

        confidence: float = float(proba_matrix.max())
        predicted_class: int = int(classes[proba_matrix.argmax()])
        label: str = self.LABEL_MAP.get(predicted_class, "Neutral")

        raw_probabilities = {
            self.LABEL_MAP.get(int(cls), "Neutral"): float(prob)
            for cls, prob in zip(classes, proba_matrix[0])
        }

        timestamp: str = recent_data.index[-1].isoformat()
        above_threshold: bool = confidence >= self._config.confidence_threshold

        result = {
            "prediction":        label,
            "confidence":        confidence,
            "raw_probabilities": raw_probabilities,
            "timestamp":         timestamp,
            "above_threshold":   above_threshold,
        }

        logger.debug(
            "[inference] %s | conf=%.3f | above_threshold=%s | ts=%s",
            label, confidence, above_threshold, timestamp,
        )

        return result
