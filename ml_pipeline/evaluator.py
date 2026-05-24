"""
ml_pipeline/evaluator.py
Model evaluation utilities: metrics, logging, optional confusion-matrix plot.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

if TYPE_CHECKING:
    from ml_pipeline.model import ModelTrainer

logger = logging.getLogger(__name__)

_LABELS: list[int] = [-1, 0, 1]
_TARGET_NAMES: list[str] = ["Bearish", "Neutral", "Bullish"]
_PRECISION_WARN_THRESHOLD = 0.5


def evaluate(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_trainer: "ModelTrainer",
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> dict:
    """Compute and log classification metrics for a three-class direction model.

    Metrics are computed for labels {-1, 0, 1} mapped to
    {'Bearish', 'Neutral', 'Bullish'} respectively.

    Returns dict with keys: precision_bullish, precision_bearish,
    recall_bullish, recall_bearish, f1_bullish, f1_bearish,
    accuracy, feature_importance_df.
    """
    report: str = classification_report(
        y_true,
        y_pred,
        labels=_LABELS,
        target_names=_TARGET_NAMES,
        zero_division=0,
    )
    logger.info("[evaluator] Classification report:\n%s", report)

    # precision_recall_fscore_support returns arrays ordered by labels=[-1, 0, 1]
    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=_LABELS,
        zero_division=0,
    )

    precision_bearish, _, precision_bullish = precision_arr
    recall_bearish,    _, recall_bullish    = recall_arr
    f1_bearish,        _, f1_bullish        = f1_arr

    accuracy: float = accuracy_score(y_true, y_pred)
    logger.info("[evaluator] Overall accuracy: %.4f", accuracy)

    _warn_if_low_precision("Bullish", precision_bullish)
    _warn_if_low_precision("Bearish", precision_bearish)

    fi_df: pd.DataFrame = model_trainer.get_feature_importance(feature_names)
    top10 = fi_df.head(10)
    logger.info("[evaluator] Top-10 features by importance:\n%s", top10.to_string(index=False))

    return {
        "precision_bullish": float(precision_bullish),
        "precision_bearish": float(precision_bearish),
        "recall_bullish":    float(recall_bullish),
        "recall_bearish":    float(recall_bearish),
        "f1_bullish":        float(f1_bullish),
        "f1_bearish":        float(f1_bearish),
        "accuracy":          float(accuracy),
        "feature_importance_df": fi_df,
    }


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plot a colour-coded confusion matrix for the three-class classifier.

    If save_path is provided the figure is saved there (parent dirs created
    automatically) and the window is not shown. Otherwise shown interactively.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    except ImportError as exc:
        logger.error("[evaluator] matplotlib is required for plot_confusion_matrix: %s", exc)
        return

    cm = confusion_matrix(y_true, y_pred, labels=_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=_TARGET_NAMES)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Price Direction Classifier")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("[evaluator] Confusion matrix saved to %s", save_path)
        plt.close(fig)
    else:
        plt.show()


def _warn_if_low_precision(class_name: str, value: float) -> None:
    """Emit a warning log when precision is below the alert threshold."""
    if value < _PRECISION_WARN_THRESHOLD:
        logger.warning(
            "[evaluator] WARNING: Low precision for %s: %.3f",
            class_name,
            value,
        )
