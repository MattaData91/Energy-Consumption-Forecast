import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

# Define utility functions

def load_and_clean_data(use_all_btu_path, msn_codes_path):
    """Load and clean the datasets, returning a merged dataframe."""
    # Load datasets
    use_all_btu_df = pd.read_csv(use_all_btu_path)
    msn_codes_df = pd.read_excel(msn_codes_path, sheet_name='MSN descriptions')

    # Clean the MSN descriptions
    msn_cleaned_df = msn_codes_df[['MSN', 'Description', 'Unit']].dropna().reset_index(drop=True)

    # Merge datasets
    merged_df = pd.merge(use_all_btu_df, msn_cleaned_df, on="MSN", how="left")
    return merged_df

def filter_nuclear_data(merged_df):
    """Filter nuclear-related data and reshape it to long format."""
    nuclear_data = merged_df[merged_df['Description'].str.contains('nuclear', case=False, na=False)]
    nuclear_long = pd.melt(
        nuclear_data,
        id_vars=['Data_Status', 'State', 'MSN', 'Description', 'Unit'],
        var_name='Year',
        value_name='Energy_Consumption'
    )
    nuclear_long['Year'] = pd.to_numeric(nuclear_long['Year'], errors='coerce')
    nuclear_long.dropna(subset=['Energy_Consumption'], inplace=True)
    return nuclear_long

def create_lagged_features(data, lags=3):
    """Create lagged features for supervised learning."""
    df = data.copy()
    for lag in range(1, lags + 1):
        df[f"Lag_{lag}"] = df['Energy_Consumption'].shift(lag)
    df.dropna(inplace=True)
    return df

def evaluate_model(model_name, y_test, y_pred):
    """Evaluate the model and return MAE, RMSE, and MAPE metrics."""
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = (mae / y_test.mean()) * 100
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    return mae, rmse, mape

def plot_results(y_train, y_test, y_pred_xgb, y_pred_rf, title):
    """Plot training data, actual values, and model predictions."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_train.index, y_train, label="Training Data")
    plt.plot(y_test.index, y_test, label="Actual Data", color="orange")
    plt.plot(y_test.index, y_pred_xgb, label="XGBoost Forecast", color="green", linestyle="--")
    plt.plot(y_test.index, y_pred_rf, label="Random Forest Forecast", color="blue", linestyle="--")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Energy Consumption (Billion BTU)")
    plt.legend()
    plt.grid()
    plt.show()

# Main script

# Paths to files
use_all_btu_path = r"C:\Code\energy-demand-forecasting\data\raw\use_all_btu.csv"
msn_codes_path = r"C:\Code\energy-demand-forecasting\docs\msn_codes_and_descriptions.xlsx"

# Step 1: Load and clean data
merged_df = load_and_clean_data(use_all_btu_path, msn_codes_path)

# Step 2: Filter for nuclear data
nuclear_long = filter_nuclear_data(merged_df)

# Step 3: Prepare national-level data
national_data = nuclear_long[(nuclear_long['State'] == 'US') & (nuclear_long['MSN'] == 'NUEGB')]
forecast_data = national_data[['Year', 'Energy_Consumption']].set_index('Year')

# Step 4: Train-test split and feature creation
forecast_data_1960 = forecast_data[forecast_data.index >= 1960]
ml_data = create_lagged_features(forecast_data_1960, lags=3)
X_train, y_train = ml_data.iloc[:-5, 1:], ml_data.iloc[:-5, 0]
X_test, y_test = ml_data.iloc[-5:, 1:], ml_data.iloc[-5:, 0]

# Step 5: Train and evaluate models
# XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_forecast = xgb_model.predict(X_test)
evaluate_model("XGBoost", y_test, xgb_forecast)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_forecast = rf_model.predict(X_test)
evaluate_model("Random Forest", y_test, rf_forecast)

# Step 6: Plot results
plot_results(y_train, y_test, xgb_forecast, rf_forecast, "Nuclear Energy Consumption Forecast: XGBoost vs Random Forest")
