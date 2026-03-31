import pandas as pd
import numpy as np
import yfinance as yf
import ta

# --------------------------
# CONFIG
# --------------------------
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "XOM", "UNH"
]

START_DATE = "2024-01-01"
END_DATE = "2026-03-29"

all_data = []

# --------------------------
# MARKET DATA (load once)
# --------------------------
market = yf.download("^GSPC", start=START_DATE, end=END_DATE)

if isinstance(market.columns, pd.MultiIndex):
    market.columns = market.columns.get_level_values(0)
    
market['Market_Return_1d'] = market['Close'].pct_change()
market['Market_Return_5d'] = market['Close'].pct_change(5)
market = market[['Market_Return_1d', 'Market_Return_5d']]
market = market.reset_index()

# --------------------------
# LOOP THROUGH STOCKS
# --------------------------
for ticker in TICKERS:
    print(f"Processing {ticker}...")

    df = yf.download(ticker, start=START_DATE, end=END_DATE)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if df.empty:
        continue

    # Add ticker column
    df['Ticker'] = ticker

    # --------------------------
    # FEATURES
    # --------------------------
    df['Return_1d'] = df['Close'].pct_change()
    df['Return_3d'] = df['Close'].pct_change(3)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Trend
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Price_to_MA10'] = df['Close'] / df['MA10']
    df['Price_to_MA50'] = df['Close'] / df['MA50']

    # Momentum
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['Momentum_5d'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10d'] = df['Close'] - df['Close'].shift(10)

    # Volatility
    df['Volatility_10d'] = df['Return_1d'].rolling(10).std()
    df['Volatility_20d'] = df['Return_1d'].rolling(20).std()
    atr = ta.volatility.AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )
    df['ATR'] = atr.average_true_range()

    # Volume
    df['Volume_MA10'] = df['Volume'].rolling(10).mean()
    df['Relative_Volume'] = df['Volume'] / df['Volume_MA10']
    df['Volume_Change'] = df['Volume'].pct_change()

    # Price structure
    df['Intraday_Range'] = df['High'] - df['Low']
    df['Close_to_High'] = (df['High'] - df['Close']) / df['High']
    df['Close_to_Low'] = (df['Close'] - df['Low']) / df['Low']

    # Merge market data
    df = pd.merge(df, market, on='Date', how='left')

    # Targets
    df['Target_Return_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    df['Target_Return_10d'] = df['Close'].shift(-10) / df['Close'] - 1
    df['Target_Return_20d'] = df['Close'].shift(-20) / df['Close'] - 1
    df['Target_Direction_5d'] = (df['Target_Return_5d'] > 0).astype(int)

    # Clean
    df = df.dropna()

    all_data.append(df)

# --------------------------
# COMBINE ALL STOCKS
# --------------------------
final_df = pd.concat(all_data, ignore_index=True)

# --------------------------
# SAVE
# --------------------------
final_df.to_csv("data/final/sp500_top10_dataset.csv", index=False)

print("Dataset created!")
print(final_df.shape)