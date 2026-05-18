from dataclasses import dataclass, field

@dataclass(frozen=True)
class PipelineConfig:
    resample_interval: str = "1s"
    rolling_window: int = 1000
    output_compression: str = "snappy"
    timestamp_column: str = "system_time"