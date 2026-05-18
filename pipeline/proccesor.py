import pandas as pd
from pathlib import Path

from config import PipelineConfig
from pipeline.features import compute_microstructure_features
from pipeline.validators import validate_schema, validate_times

class OrderBookProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.df : pd.DataFrame | None = None

    def load(self, path: str | Path) -> None:
        df = pd.read_csv(path, parse_dates=["exchange_timestamp", "local_timestamp"])

    def load(self, path: str | Path) -> "OrderBookProcessor":
        path = Path(path)
        
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path, dtype=self.config.dtype_map)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        validate_schema(df)
        self._df = df
        print(f"[load] Loaded {len(df):,} rows from {path.name}")
        return self

    def clean(self) -> "OrderBookProcessor":
        df = self._df.copy()
        initial_len = len(df)

        df = df.sort_values("exchange_timestamp")
        df = df.drop_duplicates(subset="exchange_timestamp", keep="last")

        valid_spread_mask = df["ask_price_1"] > df["bid_price_1"]
        df = df[valid_spread_mask]

        validate_times(df)


        df["datetime"] = pd.to_datetime(df["exchange_timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime")
        df = df.resample(self.config.resample_interval).last().ffill()

        dropped = initial_len - len(df)
        print(f"[clean] Removed {dropped:,} rows. Remaining: {len(df):,}")
        
        self._df = df
        return self

    def extract_features(self) -> "OrderBookProcessor":
        self._df = compute_microstructure_features(self._df, self.config.rolling_window)
        print(f"[features] Computed microstructure features. Columns: {list(self._df.columns)}")
        return self

    def export(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._df.to_parquet(
            output_path,
            compression=self.config.output_compression,
            index=True,
        )
        print(f"[export] Saved {len(self._df):,} rows → {output_path}")
