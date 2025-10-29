import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load and preprocess the data"""
    # Load the datasets
    imbalance_actual = pd.read_csv('data/imbalance_actual.csv')
    imbalance_forecast = pd.read_csv('data/imbalance_forecast.csv')
    dam_prices = pd.read_csv('data/dam_prices.csv')

    # Convert datetime columns
    for df in [imbalance_actual, imbalance_forecast, dam_prices]:
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        df.set_index('datetime_utc', inplace=True)

    return imbalance_actual, imbalance_forecast, dam_prices

def engineer_features(df):
    """Create features for the model"""
    df = df.copy()
    
    # Time-based features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Rolling statistics
    df['price_rolling_mean_15'] = df['price_eur_mwh'].rolling(window=15, min_periods=1).mean()
    df['price_rolling_std_15'] = df['price_eur_mwh'].rolling(window=15, min_periods=1).std()
    
    # Lag features
    df['price_lag_1'] = df['price_eur_mwh'].shift(1)
    df['price_lag_2'] = df['price_eur_mwh'].shift(2)
    df['price_lag_3'] = df['price_eur_mwh'].shift(3)
    
    # Fill NaN values
    df = df.fillna(method='ffill')
    
    return df

def prepare_training_data(actual, forecast):
    """Prepare data for training"""
    # Engineer features for both datasets
    actual_features = engineer_features(actual)
    forecast_features = engineer_features(forecast)
    
    # Align the datasets on timestamps
    common_index = actual_features.index.intersection(forecast_features.index)
    actual_aligned = actual_features.loc[common_index]
    forecast_aligned = forecast_features.loc[common_index]
    
    # Combine features
    X = pd.DataFrame(index=common_index)
    X['forecast_price'] = forecast_aligned['price_eur_mwh']
    for col in actual_aligned.columns:
        if col != 'price_eur_mwh':
            X[f'actual_{col}'] = actual_aligned[col]
    
    y = actual_aligned['price_eur_mwh']
    
    return X, y

def train_model(X, y):
    """Train the prediction model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    
    return model

def generate_predictions(model, forecast_data, target_datetime):
    """Generate predictions for the target datetime"""
    # Prepare forecast data
    forecast_features = engineer_features(forecast_data)
    
    # Filter data for prediction
    prediction_data = forecast_features.loc[target_datetime:target_datetime + timedelta(hours=1)]
    
    # Prepare features for prediction
    X_pred = pd.DataFrame(index=prediction_data.index)
    X_pred['forecast_price'] = prediction_data['price_eur_mwh']
    for col in prediction_data.columns:
        if col != 'price_eur_mwh':
            X_pred[f'actual_{col}'] = prediction_data[col]
    
    # Make predictions
    predictions = model.predict(X_pred)
    
    # Create output DataFrame
    output = pd.DataFrame(index=prediction_data.index)
    output['price_eur_mwh'] = predictions
    output = output.reset_index()
    
    return output

def plot_predictions(actual, predictions):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual', alpha=0.7)
    plt.plot(predictions.index, predictions, label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Imbalance Prices')
    plt.xlabel('Time')
    plt.ylabel('Price (EUR/MWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predictions_plot.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    imbalance_actual, imbalance_forecast, dam_prices = load_data()
    
    # Prepare training data
    print("Preparing training data...")
    X, y = prepare_training_data(imbalance_actual, imbalance_forecast)
    
    # Train model
    print("Training model...")
    model = train_model(X, y)
    
    # Generate predictions for current datetime
    target_datetime = datetime(2025, 10, 29, 18, 30)  # Example target datetime
    predictions = generate_predictions(model, imbalance_forecast, target_datetime)
    
    # Save predictions
    output_file = 'predictions.csv'
    predictions.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    print("\nPredictions:")
    print(predictions)

    # Plot actual vs predicted values if we have actual values
    if target_datetime in imbalance_actual.index:
        actual_prices = imbalance_actual.loc[target_datetime:target_datetime + timedelta(hours=1), 'price_eur_mwh']
        plot_predictions(actual_prices, predictions['price_eur_mwh'])
        print("\nPrediction plot saved as 'predictions_plot.png'")

if __name__ == "__main__":
    main()