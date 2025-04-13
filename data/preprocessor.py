# Clean and add indicators (RSI, MACD, ATR)
import pandas as pd
import talib
from data.data_fetcher import fetch_forex_data

def preprocess_data(df):
    """Clean and enrich forex data with indicators."""
    required = ["open", "high", "low", "close"]
    if not all(col in df.columns for col in required):
        raise ValueError("Data must include OHLC columns")
    
    df["sma20"] = talib.SMA(df["close"], timeperiod=20)
    df["sma50"] = talib.SMA(df["close"], timeperiod=50)
    df["rsi"] = talib.RSI(df["close"], timeperiod=14)
    df["macd"], df["signal"], _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    
    df.dropna(inplace=True)
    return df

def load_forex_data():
    """Load and preprocess real EUR/USD 1H data."""
    df = fetch_forex_data()
    return preprocess_data(df)

if __name__ == "__main__":
    data = load_forex_data()
    print(data.head())
    print(f"Preprocessed {len(data)} rows with indicators.")