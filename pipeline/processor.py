import pandas as pd
from pathlib import Path

from config import PipelineConfig
from pipeline.features import compute_microstructure_features
from pipeline.validators import validate_schema, validate_non_empty


class OrderBookProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._df: pd.DataFrame | None = None

    def load(self, path: str | Path) -> "OrderBookProcessor":
        path = Path(path)

        if path.suffix == ".parquet":
            self._df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            self._df = pd.read_csv(
                path,
                index_col=0,
                parse_dates=["system_time"],
            )
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        validate_non_empty(self._df, "load")
        print(f"[load] {len(self._df):,} rows from '{path.name}'")
        return self

    def clean(self) -> "OrderBookProcessor":
        df = self._df.copy()
        initial_len = len(df)

        df["bid_price_1"] = (df["midpoint"] * (1 + df["bids_distance_0"])).astype("float32")
        df["ask_price_1"] = (df["midpoint"] * (1 + df["asks_distance_0"])).astype("float32")
        df["bid_qty_1"]   = df["bids_notional_0"].astype("float32")
        df["ask_qty_1"]   = df["asks_notional_0"].astype("float32")

        df = df.sort_values("system_time")
        df = df.drop_duplicates(subset="system_time", keep="last")

        df = df[df["ask_price_1"] > df["bid_price_1"]]

        validate_non_empty(df, "clean")

        df = df.set_index("system_time")
        df = df.resample(self.config.resample_interval).last().ffill()

        print(f"[clean] {initial_len:,} → {len(df):,} rows after resample. Remaining: {len(df):,}")
        self._df = df
        return self

    def extract_features(self) -> "OrderBookProcessor":
        self._df = compute_microstructure_features(self._df, self.config.rolling_window)
        print(f"[features] Done. Columns: {list(self._df.columns)}")
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