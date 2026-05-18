import pandas as pd

df = pd.read_parquet("data/processed/BTC_1min_features.parquet")

for horizon in [1, 5, 10]:
    df[f"future_return_{horizon}"] = df["mid_price"].shift(-horizon) / df["mid_price"] - 1

print("OBI")
for horizon in [1, 5, 10]:
    corr = df["imbalance"].corr(df[f"future_return_{horizon}"])
    print(f"  horizon={horizon:>2} min:  {corr:+.4f}")

df["obi_bucket"] = pd.qcut(df["imbalance"], q=5, labels=["very negative", "negative", "neutral", "positive", "very positive"])

print("After 5 minutes OBI")
print(
    df.groupby("obi_bucket", observed=True)["future_return_5"]
    .mean()
    .mul(10_000)
    .round(2)
    .to_string()
)
print("Basis points (1 bp = 0.01%)")