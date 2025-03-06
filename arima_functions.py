import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

def train_arima_model(df_train_sample, df_test_sample):
    # Check for Stationarity Using Augmented Dickey-Fuller (ADF) Test on the sampled data
    result = adfuller(df_train_sample['sum_orig_bytes_log'].dropna())

    # Extract and print test statistics
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values: ')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    # Interpret the result
    if result[1] < 0.05:
        print(f"The time series for the sampled dataset is stationary (reject H0)")
    else:
        print(f"The time series for the sampled dataset is non-stationary (fail to reject H0)")

    # Check for seasonality and plot ACF for the sampled data
    seasonal_list = []  # 1 for seasonal, 0 for non-seasonal
    seasonal_periods = []  # Record the seasonal period (m)

    # Plot the original time series for the sampled data
    plt.figure(figsize=(10,6))
    plt.plot(df_train_sample['sum_orig_bytes_log'], label='Sum of Originator Bytes')
    plt.title('Time Series Plot (Checking for Seasonality) - Sampled Dataset')
    plt.xlabel('Time')
    plt.ylabel('Bytes')
    plt.legend()
    plt.show()

    # Calculate the maximum number of lags based on data length
    max_lags = min(50, len(df_train_sample['sum_orig_bytes_log']) - 1)  # Ensure we don't exceed the data length
    acf_values = acf(df_train_sample['sum_orig_bytes_log'], nlags=max_lags)

    # Threshold for significance (1.96/sqrt(N), where N is the number of observations)
    threshold = 1.96 / (len(df_train_sample['sum_orig_bytes_log']) ** 0.5)

    # Identify significant lags
    significant_lags = [lag for lag, value in enumerate(acf_values) if abs(value) > threshold]

    # Determine if the dataset is seasonal
    seasonal = False
    seasonal_period = None

    for j in range(1, len(significant_lags)):
        # Calculate the gap between significant lags
        lag_gap = significant_lags[j] - significant_lags[j-1]
        if lag_gap > 1:  # Avoid immediate autocorrelations
            seasonal = True
            seasonal_period = lag_gap
            break

    # Record results for the sampled dataset
    if seasonal:
        print(f"The sampled time series shows evidence of seasonality with a period of {seasonal_period}.")
        seasonal_list.append(1)
        seasonal_periods.append(seasonal_period)
    else:
        print(f"The sampled time series does not show significant evidence of seasonality.")
        seasonal_list.append(0)
        seasonal_periods.append(None)  # No seasonality detected

    # Plot the ACF for visual purposes
    plt.figure(figsize=(10, 6))
    plot_acf(df_train_sample['sum_orig_bytes_log'], lags=max_lags)
    plt.title("ACF Plot for Sampled Dataset")
    plt.show()

    # Print the seasonal periods found for reference
    print("Seasonal periods detected:", seasonal_periods)

    # Define lists for p, d, q
    p_list = []
    d_list = []
    q_list = []

    # Check if the seasonal period is not None before fitting the ARIMA model
    seasonal_period = seasonal_periods[0] if seasonal_list[0] == 1 else None

    if seasonal_period is not None:
        print("Fitting auto_arima with seasonality...")

        try:
            # Fit auto_arima to find optimal p, d, q values based on seasonality
            auto_model = auto_arima(df_train_sample['sum_orig_bytes_log'],
                                    start_p=0, max_p=14,  # Range for p
                                    start_q=0, max_q=5,   # Range for q
                                    d=None,                # Let auto_arima determine d
                                    seasonal=True,         # Seasonality is detected
                                    m=seasonal_period,     # Use the seasonal period found
                                    trace=True,            # Output the process
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)

            # Print the summary of the best model found
            print("Summary for Sampled Dataset:")
            print(auto_model.summary())

            # Get the best values of p, d, and q
            p, d, q = auto_model.order

            p_list.append(p)
            d_list.append(d)
            q_list.append(q)
        except Exception as e:
            print(f"An error occurred while fitting auto_arima for the sampled dataset: {e}")
    else:
        print("No seasonality detected. Fitting auto_arima without seasonality...")

        try:
            # Fit auto_arima to find optimal p, d, q values for non-seasonal data
            auto_model = auto_arima(df_train_sample['sum_orig_bytes_log'],
                                    start_p=0, max_p=14,  # Range for p
                                    start_q=0, max_q=5,   # Range for q
                                    d=None,                # Let auto_arima determine d
                                    seasonal=False,        # No seasonality detected
                                    trace=True,            # Output the process
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)

            # Print the summary of the best model found
            print("Summary for Sampled Dataset:")
            print(auto_model.summary())

            # Get the best values of p, d, and q
            p, d, q = auto_model.order

            p_list.append(p)
            d_list.append(d)
            q_list.append(q)
        except Exception as e:
            print(f"An error occurred while fitting auto_arima for the sampled dataset: {e}")

    # Apply log transformation to the training data
    df_train_sample['log_sum_orig_bytes'] = np.log1p(df_train_sample['sum_orig_bytes_log'])

    # Apply log transformation to the test data
    df_test_sample['log_sum_orig_bytes'] = np.log1p(df_test_sample['sum_orig_bytes_log'])

    # After training the ARIMA model, you can use the fitted model to make forecasts
    try:
        model = auto_arima(df_train_sample['log_sum_orig_bytes'],
                           p=p_list[0], d=d_list[0], q=q_list[0],
                           seasonal=True if seasonal_list[0] == 1 else False,
                           m=seasonal_periods[0] if seasonal_list[0] == 1 else None,
                           trace=True, error_action='ignore', suppress_warnings=True)

        # Forecasting the next steps (adjust the forecast horizon as needed)
        forecast_steps = int(len(df_test_sample)*1.1)  # Forecast for the length of the test dataset increased by 10%
        log_forecast = model.predict(n_periods=forecast_steps)

        # Exponentiate the forecasted values to ensure they are positive
        forecast = np.expm1(log_forecast)

        print(f"Forecast for Sampled Dataset:")
        print(forecast)

        # Compare forecasted values with actual values from the test dataset
        actual_values = np.expm1(df_test_sample['log_sum_orig_bytes'].values)
        print(f"Actual values for Test Dataset:")
        print(actual_values)

        # Ensure the lengths match before calculating errors
        forecast = forecast[:len(actual_values)]  # Trim the forecast to match the length of actual values
        print(f"Trimmed forecast length: {len(forecast)}")
        print(f"Actual values length: {len(actual_values)}")
        print(f"Forecasted values: {forecast}")
        print(f"Actual values: {actual_values}")

        # Calculate and print the Mean Absolute Error (MAE) for the forecast
        mae = np.mean(np.abs(forecast - actual_values))
        print(f"Mean Absolute Error (MAE) for the forecast: {mae}")

        # Calculate and print the Mean Squared Error (MSE) for the forecast
        mse = np.mean((forecast - actual_values) ** 2)
        print(f"Mean Squared Error (MSE) for the forecast: {mse}")

        # Calculate and print the Root Mean Squared Error (RMSE) for the forecast
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error (RMSE) for the forecast: {rmse}")

        # Trim the time column to match the forecast length
        time_values = df_test_sample['window_start'].values[:len(forecast)]

        # Create a DataFrame with the forecasted and actual values
        forecast_df = pd.DataFrame({
            'Time': time_values,
            'Actual': actual_values,
            'Forecast': forecast
        })

        # Display the DataFrame
        print(forecast_df)

        # Plot the test data, forecasted line, and threshold
        plt.figure(figsize=(12, 6))
        plt.plot(df_test_sample['window_start'], actual_values, label='Actual Values', color='blue')
        plt.plot(df_test_sample['window_start'], forecast, label='Forecasted Values', color='red')
        plt.fill_between(df_test_sample['window_start'], forecast - mae, forecast + mae, color='gray', alpha=0.2, label='MAE Threshold')
        plt.title('Test Data vs Forecasted Values')
        plt.xlabel('Time')
        plt.ylabel('Sum of Original Bytes')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred during forecasting for the sampled dataset: {e}")

    return forecast_df

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

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

        # Check the format of the window_start column in df_test_original
        print(df_test_original['window_start'].head())

        # If necessary, convert the Timestamp column to match the format of window_start
        # For example, if window_start is in datetime format, convert Timestamp to datetime
        forecast_df['Time'] = pd.to_datetime(forecast_df['Time'])

        # Rename the window_start column to Time in df_test_original
        df_test_original = df_test_original.rename(columns={'window_start': 'Time'})

        # Merge forecast_df with the original dataset on the Time column
        merged_df = pd.merge(forecast_df, df_test_original, on='Time', how='left')

        # Rename columns to match the required decision tree format
        result_df = merged_df.rename(columns={
            'Time': 'ts',
            'sum_duration': 'duration',
            'sum_orig_bytes': 'orig_bytes',
            'sum_resp_bytes': 'resp_bytes',
            'sum_orig_ip_bytes': 'orig_ip_bytes',
            'sum_resp_ip_bytes': 'resp_ip_bytes',
            'label_tactic_split_agg': 'label_tactic'
        })

        # Select the desired columns
        result_df = result_df[['ts', 'duration', 'Forecast', 'Actual', 'orig_bytes', 'resp_bytes', 'orig_ip_bytes', 'resp_ip_bytes', 'label_tactic']]

        # Display the result
        print(result_df.head())

        # Define forecast and mae for the MAE threshold calculation
        forecast = result_df['Forecast']
        mae = (result_df['Actual'] - result_df['Forecast']).abs().mean()

        # Calculate the lower and upper bounds of the MAE threshold
        lower_bound = forecast - mae
        upper_bound = forecast + mae

        # Filter the DataFrame to exclude rows where Actual is outside the MAE threshold
        filtered_result_df = result_df[(result_df['Actual'] >= lower_bound) & (result_df['Actual'] <= upper_bound)]

        # Add column label tactic type string using .loc to avoid SettingWithCopyWarning
        #filtered_result_df.loc[:, 'label_tactic_str'] = filtered_result_df['label_tactic'].apply(lambda x: str(x[0]).strip("[]'") if isinstance(x, list) and len(x) > 0 else '')
        filtered_result_df = filtered_result_df.assign(label_tactic_str=filtered_result_df['label_tactic'].apply(lambda x: str(x[0]).strip("[]'") if isinstance(x, list) and len(x) > 0 else ''))

        # Remove the column label_tactic and rename label_tactic_str to label_tactic
        filtered_result_df = filtered_result_df.drop(columns=['label_tactic']).rename(columns={'label_tactic_str': 'label_tactic'}) 

        # Display the filtered DataFrame
        print(filtered_result_df.head())

        return filtered_result_df

    except Exception as e:
        print(f"An error occurred in process_forecast: {e}")
        return None