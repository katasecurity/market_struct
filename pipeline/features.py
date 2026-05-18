import zipfile
import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    
    bid = df['bid_price_1'].to_numpy(dtype=np.float32)
    ask = df['ask_price_1'].to_numpy(dtype=np.float32)
    bid_qty = df['bid_qty_1'].to_numpy(dtype=np.float32)
    ask_qty = df['ask_qty_1'].to_numpy(dtype=np.float32)

    mid_price = (bid + ask) / 2
    spread = ask - bid

    total_qty = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total_qty
    weighted_mid = (bid * ask_qty + ask * bid_qty) / total_qty
    
    df = df.copy()
    df["mid_price"] = mid_price
    df["spread"] = spread
    df["imbalance"] = imbalance
    df["weighted_mid"] = weighted_mid
     
    mid_series = df["mid_price"]
    rolling = mid_series.rolling(window=rolling_window, min_periods=1)

    df["rolling_mean"] = rolling.mean()
    df["rolling_std"] = rolling.std()
     
    std = df["rolling_std"].to_numpy(dtype=np.float32)
    mean = df["rolling_mean"].to_numpy(dtype=np.float32)
     
    with np.errstate(invalid="ignore", divide="ignore"):
        z_score = np.where(std > 0, (mid_price - mean) / std, 0.0)

        df["z_score"] = z_score.astype(np.float32)

        return df