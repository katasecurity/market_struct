# Crypto Market Pipeline

A Python pipeline for processing limit order book (LOB) data, extracting microstructure features, and training a machine learning model to predict short-term price direction.

## What it does

**ETL & Feature Engineering**
- Cleans asynchronous order book snapshots, filters anomalies (negative spreads), and resamples into a uniform time grid (1s or 1min).
- Computes key microstructure metrics: mid price, bid-ask spread, order book imbalance (OBI), volume-weighted mid, rolling z-score.

**ML Pipeline**
- Generates a ternary directional target (Bullish / Neutral / Bearish) from future returns with a configurable basis-point threshold.
- Engineers lag features and imbalance deltas to give the model temporal context.
- Trains a LightGBM classifier with balanced class weights to handle market flat-line dominance.
- Evaluates precision for directional classes (Bullish / Bearish) — the metric that matters for trade entry quality.
- Exports a serialised model for real-time inference via `CandlePredictor`.

**Exploratory Analysis**
- `explore.py`: correlation between OBI and future price returns at 1m, 5m, 10m horizons.
- `visual.py`: market regime charts and z-score visualisations.

## Visuals

<img width="1456" height="819" alt="image" src="https://github.com/user-attachments/assets/50f62a5e-8442-45a6-8c2e-8c2fa367864d" />

*Figure 1 — BTC/USD Order Book Analysis (2021)*

<img width="1456" height="819" alt="image" src="https://github.com/user-attachments/assets/bf09b14b-b19e-40e8-8ef3-debc9e1739e5" />

*Figure 2 — Rolling Z-Score (1 min)*

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# 1. ETL: place raw order book CSV in data/raw/ then run
python main.py

# 2. Train ML model (requires data/processed/BTC_1min_features.parquet)
python train.py

# 3. Charts
python visual.py
```

## Inference example

```python
from config import MLConfig
from ml_pipeline.inference import CandlePredictor

predictor = CandlePredictor(MLConfig())
result = predictor.predict_next_candle(recent_df)  # DataFrame with >= 10 rows
# {'prediction': 'Bullish', 'confidence': 0.72, 'above_threshold': True, ...}
```

## Data Source

High-Frequency Crypto Limit Order Book Data published by Martin on Kaggle.

## License

MIT
