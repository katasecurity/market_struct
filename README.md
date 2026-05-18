# Crypto Market Pipeline

## What it does
- **Data Cleaning & Alignment:** Takes messy, asynchronous order book snapshots, filters out anomalies (like negative spreads), and forward-fills them into a clean time grid (e.g., 1-second or 1-minute intervals).
- **Feature Engineering:** Calculates key microstructure metrics:
  - Mid Price & Bid-Ask Spread
  - Order Book Imbalance (OBI) - *to measure buy/sell pressure*
  - Volume-Weighted Mid Price
  - Rolling Z-Score - *for anomaly detection*
- **Exploratory Data Analysis (`explore.py`):** Checks the statistical correlation between the Order Book Imbalance and future price returns (1m, 5m, 10m horizons).
- **Visualization (`visual.py`):** Generates graphs to visually inspect market regimes.


- ## Architecture
I used a classic Pandas method-chaining approach to make the pipeline clean and readable:


- ## Visuals
<img width="1456" height="819" alt="image" src="https://github.com/user-attachments/assets/50f62a5e-8442-45a6-8c2e-8c2fa367864d" />
 Figure 1 BTC/USD Analysis (2021)
<img width="1456" height="819" alt="image" src="https://github.com/user-attachments/assets/bf09b14b-b19e-40e8-8ef3-debc9e1739e5" />
## Figure 2 Z-Score (1 min)

Backtest Results:
Total Trades: 734
Win Rate: 45.77%
Total Return: -30.36%
Market Return: 1.63%
Sharpe Ratio: -21.21

Tech Stack

    Python: The core logic

    Pandas & NumPy: Strict vectorization and rolling statistics

    PyArrow / FastParquet: For compressed, column-oriented storage

    Matplotlib: For data visualization


How to run

    Install dependencies: pip install -r requirements.txt

    Place your raw order book CSV in data/raw/

    Run the pipeline: python main.py

    Generate charts: python visual.py



Data Source & Acknowledgments

A huge thank you to Martin for publishing the High-Frequency Crypto Limit Order Book Data on Kaggle. 
Finding clean, open-source tick data of this quality is incredibly rare. 

License
MIT

