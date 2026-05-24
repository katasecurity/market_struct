"""
ml_pipeline/model.py
LightGBM-based multi-class trainer for price-direction prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from config import MLConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Wraps a LightGBM classifier with a clean train / predict / persist API.

    All mutating methods return self to support method chaining.
    """

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self._model: lgb.LGBMClassifier | None = None
        self._logger = logging.getLogger(__name__)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ModelTrainer":
        """Fit a LightGBM classifier on the supplied training data.

        Returns self for chaining, e.g. trainer.train(...).save_model(...)
        """
        self._logger.info(
            "[model] Training on %d samples, %d features",
            len(X_train),
            X_train.shape[1],
        )

        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        self._model = model

        self._logger.info("[model] Training complete. Classes: %s", model.classes_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels for X.

        Raises RuntimeError if model has not been trained or loaded.
        """
        self._assert_trained()
        return self._model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability matrix of shape (N, 3).

        Column order follows self._model.classes_, e.g. [-1, 0, 1].
        Raises RuntimeError if model has not been trained or loaded.
        """
        self._assert_trained()
        return self._model.predict_proba(X)  # type: ignore[union-attr]

    def predict_with_threshold(
        self,
        X: pd.DataFrame,
        confidence_threshold: float = 0.6,
    ) -> np.ndarray:
        """Return predictions gated by a minimum confidence threshold.

        Samples whose maximum predicted probability is below confidence_threshold
        are assigned the Neutral label (0).
        """
        self._assert_trained()
        proba = self.predict_proba(X)
        max_proba = proba.max(axis=1)
        argmax_idx = proba.argmax(axis=1)

        classes: np.ndarray = self._model.classes_  # type: ignore[union-attr]
        mapped = classes[argmax_idx]

        result = np.where(max_proba >= confidence_threshold, mapped, 0)
        return result.astype(int)

    def save_model(self, path: str | Path) -> None:
        """Serialise the fitted model to disk using joblib.

        Parent directories are created automatically.
        Raises RuntimeError if model has not been trained or loaded.
        """
        self._assert_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        self._logger.info("[model] Saved to %s", path)

    def load_model(self, path: str | Path) -> "ModelTrainer":
        """Deserialise a previously saved model from path. Returns self."""
        path = Path(path)
        self._model = joblib.load(path)
        self._logger.info("[model] Loaded from %s", path)
        return self

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Return a DataFrame of feature importances sorted descending.

        Columns: ['feature', 'importance'].
        Raises RuntimeError if model has not been trained or loaded.
        """
        self._assert_trained()
        importances: np.ndarray = self._model.feature_importances_  # type: ignore[union-attr]
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def _assert_trained(self) -> None:
        """Raise RuntimeError when the model has not been fitted yet."""
        if self._model is None:
            raise RuntimeError("Model is not trained. Call .train() or .load_model() first.")
