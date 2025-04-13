# Fetch forex data (e.g., MetaTrader, Dukascopy)
import pandas as pd
import os

def fetch_forex_data(pair="EURUSD", timeframe="1H", start_date="2023-01-01", end_date="2025-04-12", data_dir="data"):
    """
    Fetch EUR/USD 1H data from CSV file.
    Args:
        pair (str): Currency pair (e.g., "EURUSD").
        timeframe (str): Timeframe (e.g., "1H").
        start_date (str): Start date in YYYY-MM-DD.
        end_date (str): End date in YYYY-MM-DD.
        data_dir (str): Directory where CSV is stored.
    Returns:
        pd.DataFrame: OHLC data.
    """
    file_name = f"{pair}_{timeframe}_2023_2025.csv"
    file_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found. Download from Dukascopy or update path.")
    
    # Load CSV with headers
    df = pd.read_csv(file_path)
    
    # Rename columns to match expected format
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    
    # Convert timestamp to datetime (already IST, just parse)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y %H:%M:%S.000 GMT+0530")
    
    # Filter date range
    df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
    
    # Drop volume and set index
    df = df[["timestamp", "open", "high", "low", "close"]]
    df.set_index("timestamp", inplace=True)
    
    return df

if __name__ == "__main__":
    try:
        data = fetch_forex_data()
        print(data.head())
        print(f"Loaded {len(data)} rows of EUR/USD 1H data.")
    except Exception as e:
        print(f"Error: {e}")