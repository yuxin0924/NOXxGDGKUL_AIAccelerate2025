<p align="center">
  <img src="docs/NOX_logo.png" alt="NOX Energy Logo" height="200">
</p>


## Overview
his hackathon challenges you to work with real Belgian energy market data to build solutions that demonstrate your understanding of energy systems, data analysis, and innovative thinking.

Our Core Strategy: Develop a prediction pipeline using RandomForest and LightGBM models, integrating the Day-Ahead Market (DAM) and high-resolution Imbalance Forecast data.

Read the full challenge details in docs/NOX_Energy_tech_guidelines.pdf

Available Data (Focus on these two for features)

1. Day-Ahead Market (DAM) Prices

File: data/dam_prices.csv
Used as a key predictive feature.

2. Imbalance Price Forecasts

File: data/imbalance_forecast.csv
Used as the high-frequency real-time feature.

3. Actual Imbalance Prices

File: data/imbalance_actual.csv
Used as the Target Variable (Y) for model training.

⚠️ Important: All timestamps are in UTC


## Instruction 
1. Set Up Your Environment
pip install -r requirements.txt

2. Start Building the Prediction Pipeline

Feature Engineering: Merge dam_prices.csv (15-min), dam_1028.csv (latest) and aggregate the imbalance_forecast.csv (1-min) data to generate input features for the 15-minute prediction window.

Model Training: Train both RandomForestRegressor and LightGBM on historical data, targeting imbalance_actual.csv.


## Submission & Expected Output (Updated for Email Delivery)

Your solution must produce a CSV file with predictions (datetime_utc, price_eur_mwh) and send it via email to the jury.

Generate the single prediction CSV file.

Attach the CSV to a new email.

Use the email.mime modules to ensure proper attachment handling.

Send the email using an SMTP library.

