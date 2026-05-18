import pandas as pd
import numpy as np


def compute_microstructure_features(df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    bid     = df["bid_price_1"].to_numpy(dtype=np.float32)
    ask     = df["ask_price_1"].to_numpy(dtype=np.float32)
    bid_qty = df["bid_qty_1"].to_numpy(dtype=np.float32)
    ask_qty = df["ask_qty_1"].to_numpy(dtype=np.float32)

    total_qty = bid_qty + ask_qty

    df = df.copy()
    df["mid_price"]    = (bid + ask) / 2
    df["spread"]       = ask - bid
    df["imbalance"]    = (bid_qty - ask_qty) / total_qty
    df["weighted_mid"] = (bid * ask_qty + ask * bid_qty) / total_qty

    rolling = df["mid_price"].rolling(window=rolling_window, min_periods=1)
    df["rolling_mean"] = rolling.mean()
    df["rolling_std"]  = rolling.std()

    mean = df["rolling_mean"].to_numpy(dtype=np.float32)
    std  = df["rolling_std"].to_numpy(dtype=np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        z_score = np.where(std > 0, (df["mid_price"].to_numpy(dtype=np.float32) - mean) / std, 0.0)

    df["z_score"] = z_score.astype(np.float32)

    return df