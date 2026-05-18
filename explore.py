import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/BTC_1min_features.parquet")

print("Shape:")
print(df.shape)

print("Stats")
print(df[["mid_price", "spread", "imbalance", "z_score"]].describe())

print("Nan")
print(df[["mid_price", "spread", "imbalance", "z_score"]].isna().sum())

print("Z-score")
anomalies = df[df["z_score"].abs() > 2]
print(f"Count: {len(anomalies):,} / {len(df):,} ({100 * len(anomalies)/len(df):.1f}%)")
print(anomalies[["mid_price", "spread", "imbalance", "z_score"]].head(10))