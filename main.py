from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)

def get_stock_data(ticker, start_date, end_date, interval='1d'):
    """
    Fetch historical stock data for a given ticker.
    
    :param ticker: Stock symbol (e.g., 'AAPL' for Apple)
    :param start_date: Start date for fetching data (YYYY-MM-DD)
    :param end_date: End date for fetching data (YYYY-MM-DD)
    :param interval: Data interval ('1d', '1h', etc.)
    :return: DataFrame with stock data
    """
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval=interval)
    
    # Reset index to make 'Date' a column
    data.reset_index(inplace=True)
    
    return data


# 1. Moving Average Crossover (SMA/EMA) Strategy
# Description: This strategy involves two moving averages — a fast and a slow moving average. The crossover between these averages is used as a signal to buy or sell.
# - Buy Signal: When the fast MA crosses above the slow MA.
# - Sell Signal: When the fast MA crosses below the slow MA.
def moving_average_crossover(df, short_window, long_window):
    df['Short_MA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
    prev_short_MA = df['Short_MA'].shift(1)
    prev_long_MA = df['Long_MA'].shift(1)
    df['Signal'] = 0
    df.loc[(prev_short_MA <= prev_long_MA) & (df['Short_MA'] > df['Long_MA']), 'Signal'] = 1
    df.loc[(prev_short_MA >= prev_long_MA) & (df['Short_MA'] < df['Long_MA']), 'Signal'] = -1
    return df

# 2. Bollinger Bands Strategy
# Description: This strategy uses a moving average and two standard deviations (upper and lower bands). It identifies volatility and potential buy/sell opportunities.
# - Buy Signal: When the price touches the lower band.
# - Sell Signal: When the price touches the upper band.
def bollinger_bands(df, window, num_std_dev, eps=1e-6):
    df['MA'] = df['Close'].rolling(window=window, min_periods=1).mean()
    df['STD'] = df['Close'].rolling(window=window, min_periods=1).std()
    df['Upper_Band'] = df['MA'] + (df['STD'] * num_std_dev)
    df['Lower_Band'] = df['MA'] - (df['STD'] * num_std_dev)
    df.dropna(subset=['Lower_Band', 'Upper_Band'], inplace=True)
    df['Buy_Signal'] = df['Close'] <= (df['Lower_Band'] - eps)
    df['Sell_Signal'] = df['Close'] >= (df['Upper_Band'] + eps)
    return df

# 3. MACD (Moving Average Convergence Divergence) Strategy
# Description: MACD is used to find changes in the strength, direction, momentum, and duration of a trend in a stock’s price.
# - Buy Signal: When the MACD line crosses above the Signal line.
# - Sell Signal: When the MACD line crosses below the Signal line.
def calculate_macd(df):
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Buy_Signal'] = df['MACD'] > df['Signal_Line']
    df['Sell_Signal'] = df['MACD'] < df['Signal_Line']
    return df

# 4. Mean Reversion Strategy
# Description: This strategy assumes that asset prices will revert to their mean over time.
# - Buy Signal: When the price is significantly below the mean.
# - Sell Signal: When the price is significantly above the mean.
def mean_reversion(df, ticker, window, z_score_threshold, start_date):
    # Download historical data for the past year
    reference_date = pd.to_datetime(start_date)
    one_year_ago = reference_date - timedelta(days=365)
    end_date = reference_date + timedelta(days=1)
    historical_data = yf.download(ticker, start=one_year_ago, end=end_date)[['Close']]
    if historical_data.empty:
        raise ValueError("No data fetched. Check the ticker symbol or date range.")
    historical_data.columns = ['Close']
    historical_data['Mean'] = historical_data['Close'].rolling(window=window).mean()
    historical_data['Std'] = historical_data['Close'].rolling(window=window).std()
    historical_data = historical_data.dropna(subset=['Mean', 'Std'])
    historical_data['Z_Score'] = (historical_data['Close'] - historical_data['Mean']) / historical_data['Std']
    historical_data['Buy_Signal'] = historical_data['Z_Score'] < -z_score_threshold
    historical_data['Sell_Signal'] = historical_data['Z_Score'] > z_score_threshold
    df['Date'] = df['Date'].dt.tz_localize(None)
    df = df.merge(historical_data[['Mean', 'Std', 'Z_Score', 'Buy_Signal', 'Sell_Signal']], 
                  left_on='Date', right_index=True, how='left')
    return df

# 5. RSI (Relative Strength Index) Strategy
# Description: RSI measures the speed and change of price movements. It is typically used to identify overbought or oversold conditions.
# - Buy Signal: When RSI is below 30 (oversold).
# - Sell Signal: When RSI is above 70 (overbought).
def calculate_rsi(df, ticker, window, start_date):
    # Download historical data (enough for RSI calculation)
    reference_date = pd.to_datetime(start_date)
    one_year_ago = reference_date - timedelta(days=365)
    end_date = reference_date + timedelta(days=1)
    historical_data = yf.download(ticker, start=one_year_ago, end=end_date)[['Close']]
    delta = historical_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    avg_loss = avg_loss.replace(0, 0.001)
    rs = avg_gain / avg_loss
    historical_data['RSI'] = 100 - (100 / (1 + rs))
    historical_data['Buy_Signal'] = historical_data['RSI'] < 30
    historical_data['Sell_Signal'] = historical_data['RSI'] > 70
    historical_data = historical_data.reset_index()
    historical_data['Date_str'] = historical_data['Date'].dt.strftime('%Y-%m-%d')
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
    rsi_dict = dict(zip(historical_data['Date_str'], historical_data['RSI']))
    buy_dict = dict(zip(historical_data['Date_str'], historical_data['Buy_Signal']))
    sell_dict = dict(zip(historical_data['Date_str'], historical_data['Sell_Signal']))
    df['RSI'] = df['Date_str'].map(rsi_dict)
    df['Buy_Signal'] = df['Date_str'].map(buy_dict)
    df['Sell_Signal'] = df['Date_str'].map(sell_dict)
    df = df.drop(columns=['Date_str'])
    return df

# 6. Pairs Trading Strategy
# Description: Involves two stocks that historically correlate with each other. When their spread deviates from the mean, it’s an opportunity to bet on the mean reversion of the spread.
# - Buy Signal: When the Z-Score is less than -2.
# - Sell Signal: When the Z-Score is greater than 2.
def pairs_trading(df, ticker1, ticker2, window, z_score_threshold, start_date):
    # Download historical data for both tickers
    reference_date = pd.to_datetime(start_date)
    one_year_ago = reference_date - timedelta(days=365)
    end_date = reference_date + timedelta(days=1)
    historical_data1 = yf.download(ticker1, start=one_year_ago, end=end_date)[['Close']]
    historical_data2 = yf.download(ticker2, start=one_year_ago, end=end_date)[['Close']]
    if historical_data1.empty or historical_data2.empty:
        raise ValueError("No data fetched. Check the ticker symbols or date range.")
    historical_data1.columns = ['Close_1']
    historical_data2.columns = ['Close_2']
    data = pd.merge(historical_data1, historical_data2, left_index=True, right_index=True)
    data['Spread'] = data['Close_1'] - data['Close_2']
    data['Spread_Mean'] = data['Spread'].rolling(window=window).mean()
    data['Spread_Std'] = data['Spread'].rolling(window=window).std()
    data = data.dropna(subset=['Spread_Mean', 'Spread_Std'])
    data['Z_Score'] = (data['Spread'] - data['Spread_Mean']) / data['Spread_Std']
    data['Buy_Signal'] = data['Z_Score'] < -z_score_threshold
    data['Sell_Signal'] = data['Z_Score'] > z_score_threshold
    df['Date'] = df['Date'].dt.tz_localize(None)
    df = df.merge(data[['Spread', 'Spread_Mean', 'Spread_Std', 'Z_Score', 'Buy_Signal', 'Sell_Signal']], 
                  left_on='Date', right_index=True, how='left')
    
    return df

# 7. Trend Following Strategy
# Description: Involves identifying and following the prevailing market trend. It usually combines indicators like moving averages or ADX to confirm trends.
# - Buy Signal: When the trend is up.
# - Sell Signal: When the trend is down.
def trend_following(df, ticker, window, start_date):
    # Download historical data for the past year
    reference_date = pd.to_datetime(start_date)
    one_year_ago = reference_date - timedelta(days=365)
    end_date = reference_date + timedelta(days=1)
    historical_data = yf.download(ticker, start=one_year_ago, end=end_date)[['Close']]
    if historical_data.empty:
        raise ValueError("No data fetched. Check the ticker symbol or date range.")
    historical_data['Moving_Avg'] = historical_data['Close'].rolling(window=window).mean()
    historical_data = historical_data.dropna()
    historical_data = historical_data.reset_index()
    historical_data['Date_str'] = historical_data['Date'].dt.strftime('%Y-%m-%d')
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
    moving_avg_dict = dict(zip(historical_data['Date_str'], historical_data['Moving_Avg']))
    df['Moving_Avg'] = df['Date_str'].map(moving_avg_dict)
    df = df.drop(columns=['Date_str'])
    df = df.dropna(subset=['Moving_Avg'])
    df['Buy_Signal'] = df['Close'] > df['Moving_Avg']
    df['Sell_Signal'] = df['Close'] < df['Moving_Avg']
    return df

# 8. Donchian Channel Strategy
# Description: This strategy uses the highest high and the lowest low over a given period to define upper and lower channels.
# - Buy Signal: When the price breaks above the upper channel.
# - Sell Signal: When the price breaks below the lower channel.
def donchian_channel(df, ticker, window, end_date):
    # Fetch historical data for the past 365 days from end_date
    reference_date = pd.to_datetime(end_date)
    one_year_ago = reference_date - timedelta(days=365)
    historical_data = yf.Ticker(ticker).history(start=one_year_ago, end=reference_date, interval="1d")
    if historical_data.empty:
        raise ValueError("No data fetched. Check the ticker symbol or date range.")
    historical_data = historical_data.reset_index()
    historical_data['Upper_Channel'] = historical_data['High'].rolling(window=window).max()
    historical_data['Lower_Channel'] = historical_data['Low'].rolling(window=window).min()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  
    historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.tz_localize(None)  
    historical_data = historical_data[['Date', 'Upper_Channel', 'Lower_Channel']]
    result_df = pd.merge(df, historical_data, on='Date', how='left')
    result_df['Buy_Signal'] = result_df['High'] > result_df['Upper_Channel']
    result_df['Sell_Signal'] = result_df['Low'] < result_df['Lower_Channel']
    return result_df

# 9. Machine Learning (ML) Strategy
# Description: A more advanced strategy using machine learning algorithms (like decision trees, random forests, or neural networks) to predict price movement based on historical data.
# - This will require a separate model training and prediction step but can add value for users looking for cutting-edge strategies.
def ml_strategy(df):
    # Placeholder for ML model (e.g., Decision Trees, Random Forest)
    pass  # This would require integration with ML libraries and models.

# 10. Arbitrage Strategy
# Description: Exploits price differences between markets or related assets to make a risk-free profit.
# - This strategy typically requires cross-market data and is more suitable for advanced backtesting.
def arbitrage_strategy(df):
    # Placeholder for Arbitrage logic, will depend on market data from multiple sources
    pass

@app.route('/')
def home():
    return render_template('home.html')



def resolve_ticker(company_name):
    """
    Converts a company name to its ticker symbol using Yahoo Finance API.
    
    Args:
        company_name (str): Name of the company
        
    Returns:
        str: Ticker symbol if found, otherwise error message
    """
    try:
        # Yahoo Finance uses a different endpoint for search
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # Parse the search results
        if 'quotes' in data and data['quotes']:
            # Filter for equity securities
            equity_results = [quote for quote in data['quotes'] 
                              if 'quoteType' in quote and quote['quoteType'] == 'EQUITY']
            
            if equity_results:
                # Return the first equity ticker found
                return equity_results[0]['symbol']
            elif data['quotes']:
                # If no equity found but other results exist
                return data['quotes'][0]['symbol']
        
        return "Ticker not found"
        
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_percent_return(df, initial_capital=10000, position_size=1.0):
    """
    Calculate the percent return from a trading strategy.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with trading data
    initial_capital (float): Starting capital amount
    position_size (float): Percentage of capital to use per trade (0.0 to 1.0)
    
    Returns:
    float: Percent return on initial investment
    """
    # Initialize capital
    capital = initial_capital
    
    # Process each row in the dataframe
    for i in range(len(df)):
        # Get values from the current row
        row = df.iloc[i]
        
        # Extract the buy and sell signals (last two columns)
        columns = df.columns.tolist()
        buy_signal_col = columns[-2]
        sell_signal_col = columns[-1]
        
        # Get the signals
        buy_signal = row[buy_signal_col]
        sell_signal = row[sell_signal_col]
        
        # Get open and close prices
        open_price = row['Open']
        close_price = row['Close']
        
        # Skip if both signals are the same
        if buy_signal == sell_signal:
            continue
            
        # Calculate position size for this trade
        available_capital = capital * position_size
        
        # Case 1: Sell signal is True and Buy signal is False (shorting)
        if sell_signal and not buy_signal:
            # Calculate number of shares based on open price
            shares = available_capital / open_price
            
            # Calculate profit/loss: selling at open, buying back at close
            trade_profit = shares * (open_price - close_price)
            
            # Update capital
            capital += trade_profit
                
        # Case 2: Buy signal is True and Sell signal is False (going long)
        elif buy_signal and not sell_signal:
            # Calculate number of shares based on open price
            shares = available_capital / open_price
            
            # Calculate profit/loss: buying at open, selling at close
            trade_profit = shares * (close_price - open_price)
            
            # Update capital
            capital += trade_profit
    
    # Calculate percent return
    percent_return = ((capital / initial_capital) - 1) * 100
    
    return round(percent_return, 3)

def calculate_success_rate(df):
    """
    Calculate the success rate of a trading strategy based on specific rules.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with trading data
    
    Returns:
    float: Success rate as a percentage (successful trades / total trades * 100)
    """
    # Initialize counters
    successful_trades = 0
    trade_count = 0
    
    # Process each row in the dataframe
    for i in range(len(df)):
        # Get values from the current row
        row = df.iloc[i]
        
        # Extract the buy and sell signals (last two columns)
        columns = df.columns.tolist()
        buy_signal_col = columns[-2]
        sell_signal_col = columns[-1]
        
        # Get the signals
        buy_signal = row[buy_signal_col]
        sell_signal = row[sell_signal_col]
        
        # Get open and close prices
        open_price = row['Open']
        close_price = row['Close']
        
        # Skip if both signals are the same
        if buy_signal == sell_signal:
            continue
            
        # Case 1: Sell signal is True and Buy signal is False (shorting)
        if sell_signal and not buy_signal:
            trade_count += 1
            # Success if close price < open price (price went down)
            if close_price < open_price:
                successful_trades += 1
                
        # Case 2: Buy signal is True and Sell signal is False (going long)
        elif buy_signal and not sell_signal:
            trade_count += 1
            # Success if close price > open price (price went up)
            if close_price > open_price:
                successful_trades += 1
    
    # Calculate success rate
    if trade_count > 0:
        success_rate = (successful_trades / trade_count) * 100
    else:
        success_rate = 0
        
    # Return the success rate rounded to 2 decimal places
    return round(success_rate, 2)


def get_search_suggestions(query):
    """Fetch search suggestions from Yahoo Finance."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()

        suggestions = []
        if 'quotes' in data:
            for item in data['quotes']:
                if 'symbol' in item and 'shortname' in item:
                    suggestions.append({"symbol": item['symbol'], "name": item['shortname']})
        return suggestions
    except Exception as e:
        return []

@app.route('/search_suggestions')
def search_suggestions():
    query = request.args.get('q', '')
    if query:
        results = get_search_suggestions(query)
        return jsonify(results)
    return jsonify([])

# Route to process the strategy and return results
@app.route('/backtest', methods=['GET', 'POST'])
def backtest():
    if request.method == 'GET':
        return render_template('index.html')
    
    strategy = request.form.get('strategy').upper()

    # Extract form data
    ticker_input = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    ticker_input2 = request.form['second_ticker']

    try:
        # Resolve the ticker symbol if the input is a company name
        ticker = resolve_ticker(ticker_input)
        ticker2 = resolve_ticker(ticker_input2)
        # Fetch stock data for the resolved ticker
        df = get_stock_data(ticker, start_date, end_date)

        # Apply the selected strategy
        if strategy.lower() == 'moving_average_crossover':
            short_window = int(request.form['short_window']) if request.form['short_window'] else 50  # Default 50
            long_window = int(request.form['long_window']) if request.form['long_window'] else 200  # Default 200
            df = moving_average_crossover(df, short_window, long_window)

        elif strategy.lower() == 'bollinger_bands':
            window = int(request.form['bollinger_window']) if request.form['bollinger_window'] else 20  # Default 20
            num_std_dev = float(request.form['num_std_dev']) if request.form['num_std_dev'] else 2.0  # Default 2.0
            df = bollinger_bands(df, window, num_std_dev)

        elif strategy.lower() == 'macd':
            df = calculate_macd(df)

        elif strategy.lower() == 'mean_reversion':
            window = int(request.form['mean_window']) if request.form['mean_window'] else 20  # Default 20
            z_score_threshold = float(request.form['z_score_threshold']) if request.form['z_score_threshold'] else 2.0  # Default 2.0
            df = mean_reversion(df, ticker, window, z_score_threshold, end_date)

        elif strategy.lower() == 'rsi':
            window = int(request.form['rsi_window']) if request.form['rsi_window'] else 14  # Default 14
            df = calculate_rsi(df, ticker, window, end_date)

        elif strategy.lower() == 'pairs_trading':
            ticker2 = request.form['second_ticker']
            window = int(request.form['pairs_window']) if request.form['pairs_window'] else 20  # Default 20
            z_score_threshold = float(request.form['z_score_threshold']) if request.form['z_score_threshold'] else 2.0  # Default 2.0
            df = pairs_trading(df, ticker, ticker2, window, z_score_threshold, end_date)

        elif strategy.lower() == 'trend_following':
            window = int(request.form['trend_window']) if request.form['trend_window'] else 50  # Default 50
            df = trend_following(df, ticker, window, end_date)

        elif strategy.lower() == 'donchian_channel':
            window = int(request.form['donchian_window']) if request.form['donchian_window'] else 20  # Default 20
            df = donchian_channel(df, ticker, window, end_date)

        success_rate = calculate_success_rate(df)
        percent_return = calculate_percent_return(df)
        # Render the results page with the selected strategy
        return render_template('results.html', results=df.to_html(), strategy=strategy, success_rate=success_rate, percent_return=percent_return)

    except Exception as e:
        # Handle any other exceptions
        app.logger.error(f"Error: {str(e)}")
        return "An error occurred. Please try again later."


@app.route('/guide')
def guide():
    def split_description_into_paragraphs(description):
        # Split the description by two newlines and wrap each part in <p> tags
        paragraphs = description.split('\n\n')
        return ''.join(f'<p>{p}</p>' for p in paragraphs)
    
    strategies = [
        {
            "name": "Moving Average Crossover (SMA/EMA) Strategy",
            "description": """
            The Moving Average Crossover strategy involves two moving averages — a fast and a slow moving average. The crossover between these averages is used as a signal to buy or sell. A moving average smooths out price data by calculating the average closing price over a set period. The strategy generates a buy signal when the fast (short-term) moving average crosses above the slow (long-term) moving average, indicating a potential upward trend. Conversely, a sell signal is generated when the fast moving average crosses below the slow moving average, suggesting a potential downward trend.

            The **short_window** input defines the period for the short-term moving average, which is more responsive to recent price changes. The **long_window** input specifies the period for the long-term moving average, which is slower to react to price changes. A smaller short window and larger long window make the strategy more sensitive to recent market movements and more conservative, respectively. This strategy works well in trending markets but can generate false signals in sideways or choppy market conditions. If the signal equals 1 it's a buy signal and if the signal equals -1 it's a sell signal.

            To avoid false signals, traders often combine the moving average crossover with other indicators, such as volume or momentum oscillators, to confirm the trend’s strength. A risk management strategy, such as setting stop-loss orders, is essential to protect against significant price reversals after a crossover occurs.
            """
        },
        {
            "name": "Bollinger Bands Strategy",
            "description": """
            The Bollinger Bands strategy uses a moving average (the middle band) and two standard deviation bands (the upper and lower bands) to gauge volatility and potential price reversal points. The upper and lower bands expand and contract based on market volatility. A buy signal occurs when the price touches or falls below the lower band, suggesting the asset may be oversold and due for a rebound. A sell signal occurs when the price reaches or exceeds the upper band, indicating potential overbought conditions and a price pullback.

            The **window** input defines the period for calculating the moving average and standard deviation, typically set between 10 and 20 periods for daily charts. The **num_std_dev** input defines the number of standard deviations for the upper and lower bands. A higher number for **num_std_dev** results in wider bands, implying a higher tolerance for price movement before signaling buy or sell. Traders often use this strategy in conjunction with other indicators to avoid false signals during periods of extreme volatility.

            Bollinger Bands work well in range-bound or mean-reverting markets but may generate false signals in trending markets where price movements consistently stay outside the bands. Traders often adjust the **num_std_dev** parameter to fine-tune the strategy's sensitivity to price movements and volatility.
            """
        },
        {
            "name": "MACD (Moving Average Convergence Divergence) Strategy",
            "description": """
            The MACD strategy identifies changes in the strength, direction, momentum, and duration of a trend. The MACD line is derived by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA. The Signal line is a 9-period EMA of the MACD line. A buy signal is generated when the MACD line crosses above the Signal line, indicating the potential start of an upward trend. A sell signal occurs when the MACD line crosses below the Signal line, signaling a potential reversal or downward trend.

            The MACD strategy works well in trending markets and is often used by traders to capture momentum. The **window** inputs for the short (12 periods) and long (26 periods) EMAs, as well as the Signal line (9 periods), can be adjusted based on the asset and time frame being analyzed. Shorter periods for the MACD line and Signal line will make the strategy more sensitive to recent price movements, resulting in more frequent buy and sell signals.

            Traders often combine the MACD with other indicators, such as RSI or trend-following tools, to confirm the strength of the trend. The strategy is most effective in strong trending markets, but in choppy or range-bound conditions, the MACD can generate false signals and whipsaws.
            """
        },
        {
            "name": "Mean Reversion Strategy",
            "description": """
            The Mean Reversion strategy assumes that asset prices will revert to their mean (average) over time. When the price deviates significantly from its historical average, the strategy assumes the price will eventually return to this mean level. A buy signal occurs when the price is significantly below the mean, indicating the asset is oversold, while a sell signal is triggered when the price is above the mean, suggesting the asset is overbought.

            The **window** input defines the period used to calculate the mean (average) and standard deviation, typically set between 10 and 30 periods. The **z_score_threshold** input determines the threshold at which the price is considered too far from the mean to trigger a signal. A higher z-score threshold makes the strategy more selective, waiting for more significant deviations before triggering a buy or sell signal.

            This strategy works best in range-bound or oscillating markets but can perform poorly in trending markets, where prices may not revert to the mean for extended periods. Traders often combine the Mean Reversion strategy with volatility indicators like Bollinger Bands to confirm when a price is unusually far from the mean.
            """
        },
        {
            "name": "RSI (Relative Strength Index) Strategy",
            "description": """
            The RSI strategy is used to identify overbought or oversold conditions by measuring the speed and change of price movements. RSI values range from 0 to 100. A buy signal is generated when the RSI falls below 30, indicating that the asset is oversold and may experience a price reversal to the upside. A sell signal occurs when the RSI rises above 70, signaling that the asset is overbought and may reverse to the downside.

            The **window** input determines the period used to calculate the RSI, typically set to 14 periods. Shorter periods for RSI make the strategy more sensitive to recent price changes, leading to more frequent signals, while longer periods make it less sensitive and more conservative. Traders often use RSI in conjunction with other indicators to confirm potential reversals or continuation patterns.

            RSI is best used in range-bound markets where prices oscillate between overbought and oversold conditions. However, during strong trends, RSI can remain overbought or oversold for extended periods, leading to false signals. Traders often adjust the **RSI threshold** or combine it with trend-following indicators to avoid acting on early signals.
            """
        },
        {
            "name": "Pairs Trading Strategy",
            "description": """
            Pairs Trading involves two stocks that historically move in correlation with each other. The strategy assumes that when the spread between the two assets diverges significantly from the mean, the spread will eventually revert to its historical relationship. A buy signal occurs when the spread is below the mean by a certain threshold, suggesting the spread will widen back toward the mean, while a sell signal is generated when the spread is above the mean, indicating a potential contraction.

            The **window** input defines the period used to calculate the rolling mean and standard deviation of the spread. A **z_score_threshold** input determines the number of standard deviations from the mean required to trigger buy or sell signals. A higher z-score threshold requires the spread to deviate more significantly from the mean before a signal is triggered, reducing the number of trades but increasing the quality of signals.

            Pairs Trading works best when there is a strong historical correlation between the two assets. It is a market-neutral strategy, meaning it can work in both bullish and bearish market conditions. However, it requires careful selection of asset pairs with a strong historical relationship, and incorrect pair selection can result in false signals and losses.
            """
        },
        {
            "name": "Trend Following Strategy",
            "description": """
            The Trend Following strategy involves identifying the prevailing market trend and making trades that align with that trend. The strategy assumes that assets will continue to move in the direction of the trend for an extended period. A buy signal is generated when the price is above a moving average or another trend-following indicator, indicating an uptrend, while a sell signal is generated when the price is below the moving average, signaling a downtrend.

            The **window** input specifies the period for calculating the trend indicator, such as a moving average. The strategy typically uses a longer-term moving average (e.g., 50 or 200 periods) to confirm the overall trend. A shorter-term moving average can also be used to identify more immediate trends. Trend-following strategies are often combined with momentum indicators or volume analysis to confirm the strength of the trend.

            Trend Following is best suited for markets in a strong directional movement and can generate false signals in sideways or range-bound markets. The strategy requires discipline to hold positions during pullbacks and reversals, and proper risk management, such as trailing stop-losses, is critical to protect profits.
            """
        },
        {
            "name": "Donchian Channel Strategy",
            "description": """
            The Donchian Channel strategy uses the highest high and lowest low over a given period to create an upper and lower channel. The strategy assumes that when the price breaks out of this channel, it indicates the start of a new trend. A buy signal occurs when the price breaks above the upper channel, suggesting an upward breakout, while a sell signal is triggered when the price breaks below the lower channel, indicating a downward breakout.

            The **window** input defines the period used to calculate the highest high and lowest low, often set to 20 periods. The Donchian Channel is typically used to capture breakout trades, and the width of the channel changes with market volatility. In volatile markets, the channel will be wider, while in quieter markets, it will be narrower.

            The Donchian Channel strategy works well in trending markets and is often used in combination with other trend-following indicators. However, it can generate false signals during periods of low volatility, where the price may break out temporarily before reversing. Traders may use confirmation from other indicators, such as volume or MACD, to improve the strategy's reliability.
            """
        }
    ]
    
    # Process each strategy's description before rendering it
    for strategy in strategies:
        strategy['description'] = split_description_into_paragraphs(strategy['description'])
    
    return render_template('guide.html', strategies=strategies)

stock_data_cache = {}
last_updated = None  # Store last update timestamp

# Selenium WebDriver options
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# URLs for different stock lists
urls = {
    "Trending Now": "https://finance.yahoo.com/markets/stocks/trending/",
    "Top Gainers": "https://finance.yahoo.com/gainers/",
    "Top Losers": "https://finance.yahoo.com/losers/",
    "Most Active": "https://finance.yahoo.com/most-active/"
}

def get_stock_list(url):
    """Extracts stock symbols, names, and prices from Yahoo Finance tables."""
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    stock_data = []
    rows = driver.find_elements(By.XPATH, '//table//tr')[1:11]  # Get first 10 rows (skip header)

    for row in rows:
        cols = row.find_elements(By.TAG_NAME, 'td')
        if len(cols) >= 3:
            ticker = cols[0].text
            name = cols[1].text
            price = cols[2].text
            stock_data.append({
                "ticker": ticker,
                "name": name,
                "price": yf.Ticker(ticker).info.get('currentPrice', 'N/A'),
                "link": f"https://finance.yahoo.com/quote/{ticker}"
            })

    driver.quit()
    return stock_data

def update_stock_data():
    """Runs once per day at midnight to update stock data."""
    global stock_data_cache, last_updated

    while True:
        try:
            # Get new stock data
            new_data = {category: get_stock_list(url) for category, url in urls.items()}
            stock_data_cache = new_data
            last_updated = datetime.now()  # Update timestamp
            print(f"Stock data updated at {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")

            # Calculate time until next midnight
            now = datetime.now()
            next_update = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            sleep_time = (next_update - now).total_seconds()
            print(f"Next update in {sleep_time / 3600:.2f} hours")

        except Exception as e:
            print("Error updating stock data:", e)
            sleep_time = 86400  # Try again in 24 hours if an error occurs

        time.sleep(sleep_time)  # Sleep until next update

@app.route('/market_data')
def market_data():
    """Serve the cached stock data instantly."""
    return render_template('market_data.html', stock_data=stock_data_cache, last_updated=last_updated)

# Start the background thread when the app runs
threading.Thread(target=update_stock_data, daemon=True).start()


@app.route('/about')
def about():
    return render_template('about.html')


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)