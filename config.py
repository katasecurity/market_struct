from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    resample_interval: str = "100ms"
    rolling_window: int = 1000
    output_compression: str = "snappy"

    dtype_map: dict = None

    def __post_init__(self):
      object.__setattr__(self, "dtype_map", {
        "exchange_timestamp": "int64",
            "local_timestamp": "int64",
            "bid_price_1": "float32",
            "bid_qty_1": "float32",
            "ask_price_1": "float32",
            "ask_qty_1": "float32",
        })