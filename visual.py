import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_parquet("data/processed/BTC_1min_features.parquet")

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("BTC/USD — Market Microstructure Analysis", fontsize=14, fontweight="bold")

axes[0].plot(df.index, df["mid_price"], color="#2196F3", linewidth=0.8)
axes[0].set_ylabel("Mid Price (USD)")
axes[0].set_title("Mid Price")

axes[1].plot(df.index, df["spread"], color="#FF9800", linewidth=0.6)
axes[1].set_ylabel("Spread (USD)")
axes[1].set_title("Bid-Ask Spread")
axes[1].set_ylim(0, df["spread"].quantile(0.99))

axes[2].fill_between(
    df.index, df["imbalance"], 0,
    where=df["imbalance"] > 0, color="#4CAF50", alpha=0.7, label="Buy pressure"
)
axes[2].fill_between(
    df.index, df["imbalance"], 0,
    where=df["imbalance"] < 0, color="#F44336", alpha=0.7, label="Sell pressure"
)
axes[2].set_ylabel("Imbalance")
axes[2].set_title("Order Book Imbalance")
axes[2].legend(loc="upper right", fontsize=8)
axes[2].axhline(0, color="black", linewidth=0.5)

axes[3].plot(df.index, df["z_score"], color="#9C27B0", linewidth=0.6)
axes[3].axhline(2,  color="#F44336", linewidth=1, linestyle="--", label="+2σ")
axes[3].axhline(-2, color="#2196F3", linewidth=1, linestyle="--", label="-2σ")
axes[3].axhline(0,  color="black",   linewidth=0.5)
axes[3].set_ylabel("Z-Score")
axes[3].set_title("Price Z-Score (anomaly detection)")
axes[3].legend(loc="upper right", fontsize=8)

axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("data/processed/BTC_microstructure.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → data/processed/BTC_microstructure.png")