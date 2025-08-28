# src/data_fetcher.py

"""
Generic data fetching module for Kriterion Quant Trading System.
Handles data acquisition from a specified source (default: Yahoo Finance).
"""

import pandas as pd
import yfinance as yf
import time
import os
import requests # Aggiungiamo requests per renderlo più versatile

from config import Config

class DataFetcher:
    """Class to handle data fetching."""

    def __init__(self, data_path: str):
        """Initializes the data fetcher."""
        self.data_path = data_path

    def fetch_data_from_yahoo(self, ticker: str, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
        """Fetches historical OHLC data from Yahoo Finance."""
        print(f"📡 Fetching data for {ticker} from {start_date} to {end_date} via Yahoo Finance...")
        
        retries = 0
        while retries < max_retries:
            try:
                asset = yf.Ticker(ticker)
                df = asset.history(start=start_date, end=end_date)

                if df.empty:
                    raise ValueError(f"No data returned for {ticker}.")

                df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
                if 'adj_close' in df.columns:
                    df.rename(columns={'adj_close': 'close'}, inplace=True)

                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns for {ticker}")
                
                df = df[required_cols].dropna()
                print(f"✅ Successfully fetched {len(df)} days of data for {ticker} from Yahoo Finance.")
                return df

            except Exception as e:
                retries += 1
                print(f"❌ Error fetching from Yahoo: {e}. Retrying ({retries}/{max_retries})...")
                if retries >= max_retries:
                    raise Exception(f"Failed to fetch {ticker} from Yahoo Finance after {max_retries} retries.")
                time.sleep(2 * retries)
        return pd.DataFrame() # Ritorna un df vuoto in caso di fallimento non gestito


    def fetch_data_from_eodhd(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetches historical OHLC data from EODHD."""
        api_key = getattr(Config, 'EODHD_API_KEY', None)
        if not api_key:
            raise ValueError("EODHD_API_KEY not found in Config.")

        url = f"https://eodhd.com/api/eod/{ticker}?api_token={api_key}&from={start_date}&to={end_date}&period=d&fmt=json"
        print(f"📡 Fetching data for {ticker} from EODHD...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                raise ValueError(f"No data returned from EODHD for {ticker}")

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.rename(columns={'adjusted_close': 'close'}, inplace=True)
            
            print(f"✅ Successfully fetched {len(df)} days of data for {ticker} from EODHD.")
            return df[~df.index.duplicated(keep='last')]
        
        except Exception as e:
            print(f"❌ Error fetching from EODHD: {e}")
            raise

    def save_data(self, df: pd.DataFrame, ticker: str):
        """Saves DataFrame to a CSV file."""
        filepath = os.path.join(self.data_path, 'historical_data.csv')
        os.makedirs(self.data_path, exist_ok=True)
        df.index.name = 'date'
        df.to_csv(filepath)
        print(f"💾 Data for {ticker} saved to {filepath}")
