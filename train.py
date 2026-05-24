#!/usr/bin/env python3
"""
train.py — End-to-end ML training pipeline.

Usage:
    python train.py
    python train.py --data path/to/features.parquet
    python train.py --data path/to/features.parquet --model-out models/my_model.pkl
"""
from __future__ import annotations

import argparse
import logging
import sys

from config import MLConfig
from ml_pipeline.dataset import MLDataLoader
from ml_pipeline.evaluator import evaluate
from ml_pipeline.model import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = "data/processed/BTC_1min_features.parquet"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM candle-direction predictor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        metavar="PATH",
        help="Path to the feature Parquet file.",
    )
    parser.add_argument(
        "--model-out",
        default=None,
        metavar="PATH",
        help="Destination path for the serialised model. Defaults to MLConfig.model_path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = MLConfig()
    model_out_path: str = args.model_out or config.model_path

    logger.info("Data path : %s", args.data)
    logger.info("Model out : %s", model_out_path)
    logger.info("Horizon   : %d bars | Threshold: %.1f bps", config.target_horizon_min, config.return_threshold_bps)

    loader = MLDataLoader(config)
    (
        loader
        .load(args.data)
        .generate_target()
        .engineer_features()
    )

    X_train, X_test, y_train, y_test = loader.time_series_split()
    feature_names = loader.get_feature_columns()

    logger.info(
        "Split — train: %d rows, test: %d rows, features: %d",
        len(X_train), len(X_test), len(feature_names),
    )

    trainer = ModelTrainer(config)
    trainer.train(X_train, y_train)
    trainer.save_model(model_out_path)

    y_pred = trainer.predict(X_test)
    evaluate(y_test, y_pred, trainer, X_test, feature_names)

    print(f"\nModel saved: {model_out_path}\nTest samples: {len(X_test)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training pipeline failed with an unhandled exception.")
        sys.exit(1)
