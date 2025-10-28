<p align="center">
  <img src="docs/NOX_logo.png" alt="NOX Energy Logo" height="200">
</p>

# NOX Energy Hackathon ⚡

Welcome to the NOX Energy Hackathon! This repository contains everything you need to participate in the challenge.

## 📋 Overview

This hackathon challenges you to work with **real Belgian energy market data** to build solutions that demonstrate your understanding of energy systems, data analysis, and innovative thinking.

**Read the full challenge details in [`docs/NOX_Energy_tech_guidelines.pdf`](docs/NOX_Energy_tech_guidelines.pdf)**

## 📊 Available Data

You have access to three datasets containing Belgian energy market information:

### 1. Day-Ahead Market (DAM) Prices
**File**: [`data/dam_prices.csv`](data/dam_prices.csv)

Contains electricity prices traded one day in advance:
- **Frequency**: 15-minute intervals
- **Period**: 2024-2025
- **Unit**: EUR/MWh
- **Columns**: `datetime_utc`, `date`, `hour`, `minute`, `price_eur_mwh`

### 2. Imbalance Price Forecasts
**File**: [`data/imbalance_forecast.csv`](data/imbalance_forecast.csv)

Forecasted prices for grid balancing:
- **Frequency**: 1-minute intervals
- **Period**: May 2024 - 2025
- **Unit**: EUR/MWh
- **Columns**: `datetime_utc`, `date`, `hour`, `minute`, `second`, `price_eur_mwh`

### 3. Actual Imbalance Prices
**File**: [`data/imbalance_actual.csv`](data/imbalance_actual.csv)

Real imbalance prices from the grid:
- **Frequency**: 15-minute intervals
- **Period**: July 2024 to 2025
- **Unit**: EUR/MWh
- **Columns**: `datetime_utc`, `date`, `hour`, `minute`, `price_eur_mwh`

⚠️ **Important**: All timestamps are in **UTC**

## 🚀 Getting Started

### 1. Read the Guidelines

**Start here**: Open [`docs/NOX_Energy_tech_guidelines.pdf`](docs/NOX_Energy_tech_guidelines.pdf) for:
- Complete challenge description
- Technical requirements
- Evaluation criteria
- Submission guidelines
- API access information (if needed)

### 2. Explore the Data

Quick exploration script:
```bash
# View first few rows
head -20 data/dam_prices.csv
head -20 data/imbalance_forecast.csv
head -20 data/imbalance_actual.csv
```

### 3. Set Up Your Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install common dependencies (adjust as needed)
pip install pandas numpy matplotlib
```

### 4. Start Building

Develop your solution based on the requirements in the technical guidelines.

## 📁 Repository Structure

```
nox-energy-hackathon/
├── README.md                    # This file
├── docs/
│   └── NOX_Energy_tech_guidelines.pdf  # FULL CHALLENGE DETAILS - READ THIS!
│   └── NOX KULeuven Hackaton Presentation.pdf  # NOX & Case presentation
└── data/                        # Energy market data
    ├── dam_prices.csv          # Day-Ahead Market prices
    ├── imbalance_forecast.csv  # Imbalance forecasts
    └── imbalance_actual.csv    # Actual imbalance prices
```

## 💡 Quick Tips

### Understanding the Data

**Day-Ahead Market (DAM)**
- Electricity prices set for the next day
- Traders buy/sell energy 24 hours in advance
- More stable and predictable

**Imbalance Prices**
- Real-time grid balancing costs
- Occur when actual consumption ≠ planned consumption
- More volatile than DAM prices

**Negative Prices**
- Yes, electricity prices can be negative!
- Happens during excess renewable generation
- Producers pay consumers to use electricity

### Working with the Data

```python
import pandas as pd

# Load data
dam = pd.read_csv('data/dam_prices.csv')
dam['datetime_utc'] = pd.to_datetime(dam['datetime_utc'])

# Basic analysis
print(f"Average price: {dam['price_eur_mwh'].mean():.2f} EUR/MWh")
print(f"Min price: {dam['price_eur_mwh'].min():.2f} EUR/MWh")
print(f"Max price: {dam['price_eur_mwh'].max():.2f} EUR/MWh")
```

### API Access (Optional but recommended)

You can also query live data from:
- **ENTSOE Transparency Platform**: [Day-Ahead Market Prices](https://newtransparency.entsoe.eu/)
- **Elia Open Data**: [Imbalance Prices](https://opendata.elia.be/)

See the technical guidelines PDF for API details.

## 📤 Submission & Expected Output

### Required Output Format

Your solution must produce a **CSV file with predictions** in the following format:

**Columns:**
- `datetime_utc` - Timestamp in UTC (e.g., `2024-01-01 00:00:00`)
- `price_eur_mwh` - Predicted price in EUR/MWh (numeric)

**Example output file:**
```csv
datetime_utc,price_eur_mwh
2024-01-01 00:00:00,45.50
2024-01-01 00:15:00,43.20
2024-01-01 00:30:00,41.80
2024-01-01 00:45:00,40.15
```

⚠️ **Important**:
- Use **UTC timezone** for all timestamps
- Match the time interval specified in the guidelines
- Ensure no missing values
- Sort by datetime ascending

### Submission Requirements

**Detailed submission instructions are in the technical guidelines PDF.**

General checklist to provide at:
1. ✅ Follow the deadline specified in guidelines for the submissions of the predictions by email
2. ✅ At 19:30, provide the source code and solution documentation
3. ✅ Include any required visualizations or reports

## 📞 Support

**Technical & Logistics**: [Martin Michaux](https://www.linkedin.com/in/martin-michaux/)

**Technical**: [Adrien Debray](https://www.linkedin.com/in/adrien-debray-3820281aa/)

## 🔗 Useful Resources

### Energy Market Information
- [ENTSOE Transparency Platform](https://newtransparency.entsoe.eu/)
- [Elia Open Data Portal](https://opendata.elia.be/)
- [Open-Meteo Weather API](https://open-meteo.com/en/docs) (for renewable correlation)

---

## About NOX Energy

NOX Energy is pioneering smart energy management solutions, combining AI, IoT, and energy expertise to optimize energy consumption, reduce costs, and accelerate the renewable energy transition.

**Good luck!** ⚡

---

*For complete challenge details, evaluation criteria, and requirements, please refer to the technical guidelines PDF in the docs folder.*
