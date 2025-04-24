import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def find_best_arima_model(time_series, p_range, d_range, q_range, seasonal=False, m=None):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    # Fit ARIMA model
                    model = ARIMA(time_series, order=(p, d, q), seasonal_order=(0, 0, 0, m) if seasonal else None)
                    fitted_model = model.fit()

                    # Check if this model has the lowest AIC
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        best_model = fitted_model
                except Exception as e:
                    # Skip invalid parameter combinations
                    continue

    print(f"Best ARIMA order: {best_order} with AIC: {best_aic}")
    return best_model, best_order

def train_arima_model(df_train_sample, df_test_sample):
    # Check for Stationarity Using Augmented Dickey-Fuller (ADF) Test
    result = adfuller(df_train_sample['sum_orig_bytes_log'].dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1] < 0.05:
        print("The time series is stationary (reject H0).")
    else:
        print("The time series is non-stationary (fail to reject H0).")

    # Apply log transformation to stabilize variance
    df_train_sample['log_sum_orig_bytes'] = np.log1p(df_train_sample['sum_orig_bytes_log'])
    df_test_sample['log_sum_orig_bytes'] = np.log1p(df_test_sample['sum_orig_bytes_log'])

    # Define parameter ranges for ARIMA
    p_range = range(0, 5)  # Adjust based on your data
    d_range = range(0, 2)
    q_range = range(0, 5)

    # Detect seasonality
    seasonal = False
    seasonal_period = None
    acf_values = acf(df_train_sample['sum_orig_bytes_log'], nlags=50)
    significant_lags = [lag for lag, value in enumerate(acf_values) if abs(value) > (1.96 / np.sqrt(len(df_train_sample)))]

    for j in range(1, len(significant_lags)):
        lag_gap = significant_lags[j] - significant_lags[j - 1]
        if lag_gap > 1:
            seasonal = True
            seasonal_period = lag_gap
            break

    print(f"Seasonality detected: {seasonal}, Seasonal period: {seasonal_period}")

    # Find the best ARIMA model
    print("Finding the best ARIMA model...")
    best_model, best_order = find_best_arima_model(
        df_train_sample['log_sum_orig_bytes'], p_range, d_range, q_range, seasonal=seasonal, m=seasonal_period
    )

    '''
    # Save the ARIMA model to a file
    best_model.save("arima_model.pkl")
    print("ARIMA model saved successfully.")
    '''

    # Forecast using the best model
    forecast_steps = len(df_test_sample)
    forecast = best_model.forecast(steps=forecast_steps)

    # Exponentiate the forecasted values to revert the log transformation
    forecast = np.expm1(forecast)

    # Compare forecasted values with actual values
    actual_values = np.expm1(df_test_sample['log_sum_orig_bytes'].values)
    print(f"Forecasted values: {forecast}")
    print(f"Actual values: {actual_values}")

    # Calculate error metrics
    mae = np.mean(np.abs(forecast - actual_values))
    print(f"Mean Absolute Error (MAE): {mae}")

    # Create a DataFrame with the forecasted and actual values
    forecast_df = pd.DataFrame({
        'Time': df_test_sample['window_start'].values,  # Revert to using 'window_start'
        'Actual': actual_values,
        'Forecast': forecast
    })

    # Plot the test data, forecasted line, and threshold
    plt.figure(figsize=(12, 6))
    plt.plot(df_test_sample['window_start'], actual_values, label='Actual Values', color='blue')  # Revert to 'window_start'
    plt.plot(df_test_sample['window_start'], forecast, label='Forecasted Values', color='red')  # Revert to 'window_start'
    plt.fill_between(df_test_sample['window_start'], forecast - mae, forecast + mae, color='gray', alpha=0.2, label='MAE Threshold')  # Revert to 'window_start'
    plt.title('Test Data vs Forecasted Values')
    plt.xlabel('Time')
    plt.ylabel('Sum of Original Bytes')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return forecast_df

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def process_forecast(df_test_original, forecast_df):
    try:
        # Print the column names of forecast_df
        print("Columns in forecast_df:")
        print(forecast_df.columns)

        # Print the column names of df_test_original
        print("Columns in df_test_original:")
        print(df_test_original.columns)

        # Check the format of the Timestamp column in forecast_df
        print(forecast_df['Time'].head())

        # Ensure the Timestamp column is in datetime format
        forecast_df['Time'] = pd.to_datetime(forecast_df['Time'])

        # Merge forecast_df with the original dataset on the Time column
        merged_df = pd.merge(forecast_df, df_test_original, left_on='Time', right_on='window_start', how='left')  # Revert to 'window_start'

        # Rename columns to match the required decision tree format
        result_df = merged_df.rename(columns={
            'Time': 'window_start',  # Revert to 'window_start'
            'sum_duration': 'duration',
            'sum_orig_bytes': 'orig_bytes',
            'sum_resp_bytes': 'resp_bytes',
            'sum_orig_ip_bytes': 'orig_ip_bytes',
            'sum_resp_ip_bytes': 'resp_ip_bytes',
            'label_tactic_split_agg': 'label_tactic'
        })

        # Select the desired columns
        result_df = result_df[['window_start', 'duration', 'Forecast', 'Actual', 'orig_bytes', 'resp_bytes', 'orig_ip_bytes', 'resp_ip_bytes', 'label_tactic']]

        # Define forecast and mae for the MAE threshold calculation
        forecast = result_df['Forecast']
        mae = (result_df['Actual'] - result_df['Forecast']).abs().mean()

        # Calculate the lower and upper bounds of the MAE threshold
        lower_bound = forecast - mae
        upper_bound = forecast + mae

        # Filter the DataFrame to exclude rows where Actual is outside the MAE threshold
        filtered_result_df = result_df[(result_df['Actual'] >= lower_bound) & (result_df['Actual'] <= upper_bound)]

        # Add column label tactic type string using .loc to avoid SettingWithCopyWarning
        filtered_result_df = filtered_result_df.assign(label_tactic_str=filtered_result_df['label_tactic'].apply(lambda x: str(x[0]).strip("[]'") if isinstance(x, list) and len(x) > 0 else ''))

        # Remove the column label_tactic and rename label_tactic_str to label_tactic
        filtered_result_df = filtered_result_df.drop(columns=['label_tactic']).rename(columns={'label_tactic_str': 'label_tactic'}) 

        return filtered_result_df

    except Exception as e:
        print(f"An error occurred in process_forecast: {e}")
        return None