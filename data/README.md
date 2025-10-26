# Energy Market Data

This folder contains Belgian energy market data for the hackathon.

## 📊 Data Files

### `dam_prices.csv`
Day-Ahead Market electricity prices for Belgium (2024)
- 15-minute intervals
- EUR/MWh

### `imbalance_forecast.csv`
Forecasted imbalance prices (2025)
- 1-minute intervals  
- EUR/MWh

### `imbalance_actual.csv`
Actual imbalance prices (2024)
- 15-minute intervals
- EUR/MWh

## 📝 Notes

- All timestamps are in **UTC**
- Prices can be **negative** (excess renewable energy)
- See [`../docs/NOX_Energy_tech_guidelines.pdf`](../docs/NOX_Energy_tech_guidelines.pdf) for complete data specifications

## 💻 Quick Start

```python
import pandas as pd

# Load data
df = pd.read_csv('dam_prices.csv')
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

# Explore
print(df.head())
print(df.describe())
```
