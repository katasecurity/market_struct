import pandas as pd

REQUIRED_COLS = {
    "exchange_timestamp", "local_timestamp",
    "bid_price_1", "bid_qty_1",
    "ask_price_1", "ask_qty_1",
}

def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def validate_times(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("DataFrame is empty")
        