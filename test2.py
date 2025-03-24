import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta

def calculate_rsi(ticker, window, start_date):
    # Download historical data for the past year
    reference_date = pd.to_datetime(start_date)
    one_year_ago = reference_date - timedelta(days=365)
    end_date = reference_date + timedelta(days=1)
    historical_data = yf.download(ticker, start=one_year_ago, end=end_date)[['Close']]
    if historical_data.empty:
        raise ValueError("No data fetched. Check the ticker symbol or date range.")
    
    # Calculate RSI
    delta = historical_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # First RSI calculation
    rs = avg_gain / avg_loss
    historical_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Define buy and sell signals based on RSI thresholds
    historical_data['Buy_Signal'] = historical_data['RSI'] < 30
    historical_data['Sell_Signal'] = historical_data['RSI'] > 70
    
    
    return historical_data
# Example usage
ticker = "AAPL"
window = 20
start_date = "2025-03-17"

result = calculate_rsi(ticker, window, start_date)
print(result)
