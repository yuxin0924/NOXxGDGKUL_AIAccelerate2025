import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
import time

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
    df['day_of_week'] = df.index.map(lambda x: x.weekday())
    df['month'] = df.index.month
    
    # Ensure price_eur_mwh is numeric
    df['price_eur_mwh'] = pd.to_numeric(df['price_eur_mwh'], errors='coerce')
    
    # Rolling statistics
    df['price_rolling_mean_15'] = df['price_eur_mwh'].rolling(window=15, min_periods=1).mean()
    df['price_rolling_std_15'] = df['price_eur_mwh'].rolling(window=15, min_periods=1).std()
    
    # Lag features
    df['price_lag_1'] = df['price_eur_mwh'].shift(1)
    df['price_lag_2'] = df['price_eur_mwh'].shift(2)
    df['price_lag_3'] = df['price_eur_mwh'].shift(3)
    
    # Forward fill NaN values
    df = df.ffill()
    
    # Fill any remaining NaN with 0
    df = df.fillna(0)
    
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
    
    # Define the exact features we want to use
    selected_features = [
        'forecast_price',  # The main forecast price
        'actual_hour',     # Time-based features
        'actual_minute',
        'actual_day_of_week',
        'actual_month',
        'actual_price_rolling_mean_15',  # Statistical features
        'actual_price_rolling_std_15',
        'actual_price_lag_1',            # Lag features
        'actual_price_lag_2',
        'actual_price_lag_3'
    ]
    
    # Add forecast price
    X['forecast_price'] = forecast_aligned['price_eur_mwh']
    
    # Add time-based features
    X['actual_hour'] = actual_aligned['hour']
    X['actual_minute'] = actual_aligned['minute']
    X['actual_day_of_week'] = actual_aligned['day_of_week']
    X['actual_month'] = actual_aligned['month']
    
    # Add statistical features
    X['actual_price_rolling_mean_15'] = actual_aligned['price_rolling_mean_15']
    X['actual_price_rolling_std_15'] = actual_aligned['price_rolling_std_15']
    
    # Add lag features
    X['actual_price_lag_1'] = actual_aligned['price_lag_1']
    X['actual_price_lag_2'] = actual_aligned['price_lag_2']
    X['actual_price_lag_3'] = actual_aligned['price_lag_3']
    
    # Ensure all features are numeric and properly filled
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    # Get target variable
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

def generate_predictions(model, forecast_data, current_time):
    """Generate predictions for the current minute and next quarter-hour period"""
    # Round current time to the nearest minute
    current_minute = current_time.replace(second=0, microsecond=0)
    
    # Get the latest available forecast data
    forecast_data = forecast_data[forecast_data.index <= current_minute]
    if len(forecast_data) == 0:
        raise ValueError("No forecast data available for the current time")
    
    latest_forecast = forecast_data.iloc[-1]
    forecast_time = current_minute
    forecast_price = latest_forecast['price_eur_mwh']
    
    # Calculate the target quarter-hour datetime
    minutes_to_next_quarter = 15 - (current_minute.minute % 15)
    target_datetime = current_minute + timedelta(minutes=minutes_to_next_quarter)
    
    # Prepare forecast data
    forecast_features = engineer_features(forecast_data)
    
    # Get the latest available data for prediction
    latest_data = forecast_features.loc[:forecast_time].iloc[-15:]  # Use last 15 minutes of data
    
    if len(latest_data) < 1:
        raise ValueError("Not enough recent data for prediction")
    
    # Prepare features for prediction with exact matching features
    X_pred = pd.DataFrame(index=[target_datetime])
    
    # Add forecast price
    X_pred['forecast_price'] = latest_data['price_eur_mwh'].mean()
    
    # Add time-based features
    X_pred['actual_hour'] = target_datetime.hour
    X_pred['actual_minute'] = target_datetime.minute
    X_pred['actual_day_of_week'] = target_datetime.weekday()
    X_pred['actual_month'] = target_datetime.month
    
    # Add statistical features from latest data
    X_pred['actual_price_rolling_mean_15'] = latest_data['price_rolling_mean_15'].iloc[-1]
    X_pred['actual_price_rolling_std_15'] = latest_data['price_rolling_std_15'].iloc[-1]
    
    # Add lag features from latest data
    X_pred['actual_price_lag_1'] = latest_data['price_lag_1'].iloc[-1]
    X_pred['actual_price_lag_2'] = latest_data['price_lag_2'].iloc[-1]
    X_pred['actual_price_lag_3'] = latest_data['price_lag_3'].iloc[-1]
    
    # Make prediction
    prediction = model.predict(X_pred)[0]
    
    # Create output DataFrame with both current forecast and prediction
    output = pd.DataFrame({
        'datetime_utc': [forecast_time, target_datetime],
        'price_eur_mwh': [forecast_price, prediction],
        'type': ['current_forecast', 'prediction']
    })
    
    return output

def send_email(predictions, recipient_email, price_change, price_change_pct):
    """Send predictions via email"""
    # Email configuration
    sender_email = "yuxin_olivia_qiu@outlook.com"  # Your Outlook email
    password = "gppnfirktwsafqpf"  # Your Outlook app password
    
    # Create message
    msg = MIMEMultipart()
    current_time = datetime.now()
    msg['Subject'] = f'NOX Energy - Imbalance Price Prediction Update ({current_time.strftime("%Y-%m-%d %H:%M:%S")} UTC)'
    msg['From'] = sender_email
    msg['To'] = recipient_email
    
    # Format the current forecast and prediction
    current_forecast = predictions[predictions['type'] == 'current_forecast'].iloc[0]
    next_prediction = predictions[predictions['type'] == 'prediction'].iloc[0]
    
    # Create email body
    body = f"""
    ⚡ NOX Energy Imbalance Price Prediction ⚡
    
    📊 Current Market Status:
    Timestamp (UTC): {current_forecast['datetime_utc'].strftime('%Y-%m-%d %H:%M:%S')}
    Current Price: {current_forecast['price_eur_mwh']:.2f} EUR/MWh
    
    🔮 Next Quarter-Hour Prediction:
    Timestamp (UTC): {next_prediction['datetime_utc'].strftime('%Y-%m-%d %H:%M:%S')}
    Predicted Price: {next_prediction['price_eur_mwh']:.2f} EUR/MWh
    
    📈 Price Change Analysis:
    Absolute Change: {price_change:.2f} EUR/MWh
    Percentage Change: {price_change_pct:.1f}%
    
    ⏰ Report Generated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
    
    Note: This is an automated prediction from the NOX Energy AI system.
    """
    msg.attach(MIMEText(body, 'plain'))
    
    # Save predictions to CSV and attach
    csv_file = 'prediction.csv'
    predictions.to_csv(csv_file, index=False)
    with open(csv_file, 'rb') as f:
        attachment = MIMEApplication(f.read(), _subtype='csv')
        attachment.add_header('Content-Disposition', 'attachment', filename=csv_file)
        msg.attach(attachment)
    
    # Send email
    try:
        server = smtplib.SMTP('smtp.office365.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print(f"Prediction sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        if isinstance(e, smtplib.SMTPAuthenticationError):
            print("Authentication failed. Please check your email and app password.")
        elif isinstance(e, smtplib.SMTPServerDisconnected):
            print("Server disconnected. Please check your internet connection.")
    finally:
        if os.path.exists(csv_file):
            os.remove(csv_file)

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

def run_prediction_pipeline(model, imbalance_forecast, current_time):
    """Run the prediction pipeline and return results"""
    predictions = generate_predictions(model, imbalance_forecast, current_time)
    
    # Calculate prediction metrics
    current_forecast = predictions[predictions['type'] == 'current_forecast'].iloc[0]
    next_prediction = predictions[predictions['type'] == 'prediction'].iloc[0]
    
    # Calculate price change
    price_change = next_prediction['price_eur_mwh'] - current_forecast['price_eur_mwh']
    price_change_pct = (price_change / current_forecast['price_eur_mwh']) * 100
    
    return predictions, price_change, price_change_pct

def main():
    """Main function to run a single immediate prediction"""
    # Configuration
    recipient_email = "yuxin-qiu@qq.com"  # Your email address
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Load data
        print("\n1. Loading data...")
        imbalance_actual, imbalance_forecast, dam_prices = load_data()
        print("✓ Data loaded successfully")
        
        # Print data summary
        print("\nData Summary:")
        print(f"Imbalance Actual: {len(imbalance_actual)} records")
        print(f"Imbalance Forecast: {len(imbalance_forecast)} records")
        print(f"Latest forecast time: {imbalance_forecast.index[-1]}")
        
        # Prepare training data
        print("\n2. Preparing training data...")
        X, y = prepare_training_data(imbalance_actual, imbalance_forecast)
        print("✓ Training data prepared")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of training samples: {X.shape[0]}")
        
        # Train model
        print("\n3. Training model...")
        model = train_model(X, y)
        print("✓ Model trained successfully")
        
        # Get current time
        current_time = datetime.now()
        print(f"\n4. Current time (UTC): {current_time}")
        
        # Generate predictions
        print("\n5. Generating predictions...")
        predictions, price_change, price_change_pct = run_prediction_pipeline(
            model, imbalance_forecast, current_time
        )
        
        # Print detailed prediction results
        print("\nPrediction Results:")
        for _, row in predictions.iterrows():
            print(f"Time (UTC): {row['datetime_utc']}")
            print(f"Price: {row['price_eur_mwh']:.2f} EUR/MWh")
        
        print(f"\nPrice Change: {price_change:.2f} EUR/MWh ({price_change_pct:.1f}%)")
        
        # Save predictions and visualization
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        predictions_file = os.path.join(results_dir, f'predictions_{timestamp}.csv')
        predictions.to_csv(predictions_file, index=False)
        print(f"\n✓ Predictions saved to {predictions_file}")
        
        # Create and save visualization
        if len(imbalance_actual) > 0:
            plot_file = os.path.join(results_dir, f'prediction_plot_{timestamp}.png')
            recent_actual = imbalance_actual.iloc[-48:]  # Last 48 points
            plot_predictions(recent_actual['price_eur_mwh'], predictions['price_eur_mwh'])
            print(f"✓ Prediction plot saved to {plot_file}")
        
        # Send email with predictions
        print("\n6. Sending email notification...")
        send_email(predictions, recipient_email, price_change, price_change_pct)
        print(f"✓ Email sent to {recipient_email}")
        
        # Final status
        print("\n=== Pipeline Completed Successfully ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Results directory: {os.path.abspath(results_dir)}")
        
    except FileNotFoundError as e:
        print(f"\nError: Required data files not found. Please check the data directory.")
        print(f"Details: {str(e)}")
    except ValueError as e:
        print(f"\nError: Invalid data or prediction error.")
        print(f"Details: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error occurred:")
        print(f"Details: {str(e)}")
    finally:
        print("\nProcess finished.")

if __name__ == "__main__":
    print("🔋 NOX Energy Imbalance Price Prediction 🔋")
    print("=========================================")
    print(f"Current Time (UTC): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Email will be sent to: yuxin_olivia_qiu@outlook.com")
    print("=========================================\n")
    
    main()