import pandas as pd

REQUIRED_COLS = {
    "system_time",
    "midpoint",
    "spread",
    "bids_distance_0",
    "asks_distance_0",
    "bids_notional_0",
    "asks_notional_0",
}

def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def validate_non_empty(df: pd.DataFrame, stage: str) -> None:
    if df.empty:
        raise ValueError(f"DataFrame is empty after stage: '{stage}'")