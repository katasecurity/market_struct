from config import PipelineConfig
from pipeline.processor import OrderBookProcessor

if __name__ == "__main__":
    config = PipelineConfig(
        resample_interval="1min",
        rolling_window=200,
    )

    (
        OrderBookProcessor(config)
        .load("data/raw/BTC_1min.csv")
        .clean()
        .extract_features()
        .export("data/processed/BTC_1min_features.parquet")
    )