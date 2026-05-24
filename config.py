from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineConfig:
    resample_interval: str = "1s"
    rolling_window: int = 1000
    output_compression: str = "snappy"
    timestamp_column: str = "system_time"


@dataclass(frozen=True)
class MLConfig:
    target_horizon_min: int = 5
    return_threshold_bps: float = 2.0
    lags: list = field(default_factory=lambda: [1, 2, 3])
    model_path: str = "models/candle_predictor.pkl"
    test_size: float = 0.2
    confidence_threshold: float = 0.6
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6