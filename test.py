import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta

def donchian_channel(df, ticker, window, end_date):
    # Fetch historical data for the past 365 days from end_date
    reference_date = pd.to_datetime(end_date)
    one_year_ago = reference_date - timedelta(days=365)
    
    historical_data = yf.Ticker(ticker).history(start=one_year_ago, end=reference_date, interval="1d")

    if historical_data.empty:
        raise ValueError("No data fetched. Check the ticker symbol or date range.")

    # Reset index to avoid multi-index issues
    historical_data = historical_data.reset_index()

    # Compute Donchian Channels
    historical_data['Upper_Channel'] = historical_data['High'].rolling(window=window).max()
    historical_data['Lower_Channel'] = historical_data['Low'].rolling(window=window).min()

    # Ensure both DataFrames have matching date formats
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  
    historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.tz_localize(None)  

  
    
    # Select only needed columns for merging
    historical_data = historical_data[['Date', 'Upper_Channel', 'Lower_Channel']]

    # Merge historical data with df
    result_df = pd.merge(df, historical_data, on='Date', how='left')
    
    

    # Debug: Print some sample values to understand the relationship
    for i in range(min(5, len(result_df))):
        print(f"Date: {result_df['Date'].iloc[i]}, Close: {result_df['Close'].iloc[i]}, "
              f"Upper: {result_df['Upper_Channel'].iloc[i]}, Lower: {result_df['Lower_Channel'].iloc[i]}")

    # Generate buy/sell signals
    result_df['Buy_Signal'] = result_df['High'] > result_df['Upper_Channel']
    result_df['Sell_Signal'] = result_df['Low'] < result_df['Lower_Channel']

    return result_df

def get_stock_data(ticker, start_date, end_date, interval='1d'):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval=interval)
    data.reset_index(inplace=True)
    return data

# Example Usage:
ticker = 'AAPL'
start_date = '2025-03-03'
end_date = '2025-03-17'
window = 200  # More reasonable window size

# Get the stock data (input data for evaluation period)
df = get_stock_data(ticker, start_date, end_date)

# Run the trend-following strategy
result = donchian_channel(df, ticker, window, end_date)

# Output the result
print("\nFinal results:")
print(result)