# src/data_fetcher.py

import pandas as pd
import yfinance as yf
import time
import os

from config import Config

class DataFetcher:
    """Class to handle data fetching, saving, and loading."""

    def __init__(self, data_path: str):
        """
        Initialize the data fetcher.

        Parameters
        ----------
        data_path : str
            The path to the directory where data for a specific ticker is saved.
        """
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

    def fetch_historical_data(self, ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
        """Fetches historical OHLC data from Yahoo Finance."""
        print(f"📡 Attempting to fetch '{ticker}' from Yahoo Finance...")
        retries = 0
        while retries < max_retries:
            try:
                asset = yf.Ticker(ticker)
                df = asset.history(start=start, end=end)
                
                if df.empty:
                    raise ValueError("No data returned from Yahoo Finance")
                
                df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
                if 'adj_close' in df.columns:
                    df.rename(columns={'adj_close': 'close'}, inplace=True)
                
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Data for {ticker} is missing required columns.")
                
                print(f"✅ Successfully fetched {len(df)} rows for '{ticker}'.")
                return df[required_cols].dropna()

            except Exception as e:
                retries += 1
                print(f"❌ Failed to fetch '{ticker}' (Attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"Could not fetch data for '{ticker}' after {max_retries} attempts.")
                time.sleep(retries * 2)
        
        return pd.DataFrame()

    def save_data(self, df: pd.DataFrame):
        """Saves the DataFrame to historical_data.csv in the instance's data_path."""
        filepath = os.path.join(self.data_path, 'historical_data.csv')
        df.index.name = 'date'
        df.to_csv(filepath)
        print(f"💾 Data saved to {filepath}")

    def load_data(self) -> pd.DataFrame:
        """Loads data from historical_data.csv in the instance's data_path."""
        filepath = os.path.join(self.data_path, 'historical_data.csv')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        print(f"📂 Loaded {len(df)} days of data from {filepath}")
        return df

    def update_latest_data(self, ticker: str) -> pd.DataFrame:
        """Updates the local data file with the latest data from the source."""
        try:
            existing_df = self.load_data()
            last_date = existing_df.index[-1]
            start_update_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"📅 Last data point is {last_date.strftime('%Y-%m-%d')}. Fetching new data from {start_update_date}...")
            
            new_df = self.fetch_historical_data(
                ticker=ticker,
                start=start_update_date,
                end=datetime.now().strftime('%Y-%m-%d')
            )
            
            if not new_df.empty:
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')] # De-duplicate just in case
                self.save_data(combined_df)
                return combined_df
            else:
                print("✅ No new data to update.")
                return existing_df

        except FileNotFoundError:
            print("📥 No existing data found. Fetching full history...")
            full_df = self.fetch_historical_data(
                ticker=ticker,
                start=Config.START_DATE,
                end=Config.END_DATE
            )
            self.save_data(full_df)
            return full_df
