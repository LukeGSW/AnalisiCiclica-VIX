# src/data_fetcher.py

import pandas as pd
import time
import os
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries

from config import Config

class DataFetcher:
    """
    A robust class to fetch clean, adjusted historical data from Alpha Vantage.
    This replaces the unreliable yfinance source for tickers like VXX.
    """

    def __init__(self, data_path: str):
        """Initializes the data fetcher."""
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        if not Config.ALPHA_VANTAGE_API_KEY:
            raise ValueError("ALPHA_VANTAGE_API_KEY is not set in the environment variables.")
        self.ts = TimeSeries(key=Config.ALPHA_VANTAGE_API_KEY, output_format='pandas')

    def fetch_historical_data(self, ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
        """
        Fetches and processes historical OHLC data from Alpha Vantage.
        """
        print(f"📡 Attempting to fetch '{ticker}' from Alpha Vantage...")
        retries = 0
        while retries < max_retries:
            try:
                # Alpha Vantage free tier provides up to 5 years of data with 'compact'
                # and full history with 'full'. We use 'full'.
                data, meta_data = self.ts.get_daily_adjusted(symbol=ticker, outputsize='full')
                
                if data.empty:
                    raise ValueError("No data returned from Alpha Vantage.")

                # Rename columns to match the project's convention
                data.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '6. volume': 'volume',
                    '5. adjusted close': 'adj_close' # Keep for reference if needed
                }, inplace=True)
                
                # The data is returned in descending order, so we reverse it
                data = data.iloc[::-1]
                
                # Filter by date range
                data.index = pd.to_datetime(data.index)
                mask = (data.index >= start) & (data.index <= end)
                df_filtered = data.loc[mask]

                # Use the adjusted close as the primary close price
                df_filtered['close'] = df_filtered['adj_close']
                
                # Select final columns and drop any remaining NaNs
                final_df = df_filtered[['open', 'high', 'low', 'close', 'volume']].copy()
                final_df.dropna(inplace=True)

                print(f"✅ Successfully fetched {len(final_df)} rows for '{ticker}' from Alpha Vantage.")
                return final_df

            except Exception as e:
                retries += 1
                # The free API has a limit of 5 calls per minute. We wait to avoid hitting it.
                wait_time = 15 * retries 
                print(f"❌ Failed to fetch '{ticker}' (Attempt {retries}/{max_retries}): {e}. Waiting {wait_time}s...")
                if retries >= max_retries:
                    raise Exception(f"Could not fetch data for '{ticker}' after {max_retries} attempts.")
                time.sleep(wait_time)
        
        return pd.DataFrame()

    def save_data(self, df: pd.DataFrame):
        """Saves the DataFrame to historical_data.csv."""
        filepath = os.path.join(self.data_path, 'historical_data.csv')
        df.index.name = 'date'
        df.to_csv(filepath)
        print(f"💾 Data saved to {filepath}")

    def load_data(self) -> pd.DataFrame:
        """Loads data from historical_data.csv."""
        filepath = os.path.join(self.data_path, 'historical_data.csv')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        print(f"📂 Loaded {len(df)} days of data from {filepath}")
        return df

    def update_latest_data(self, ticker: str) -> pd.DataFrame:
        """Updates the local data file with the latest data."""
        try:
            existing_df = self.load_data()
            # Alpha Vantage can have a delay, so we re-fetch the last few days
            start_update_date = (existing_df.index[-1] - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
            print(f"📅 Last data point is {existing_df.index[-1].strftime('%Y-%m-%d')}. Fetching new data...")
            
            # Fetching the last 100 data points is the most reliable way with AV
            data, meta_data = self.ts.get_daily_adjusted(symbol=ticker, outputsize='compact')
            data.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '6. volume': 'volume', '5. adjusted close': 'adj_close'}, inplace=True)
            data = data.iloc[::-1]
            data.index = pd.to_datetime(data.index)
            new_df = data.copy()
            new_df['close'] = new_df['adj_close']
            new_df = new_df[['open', 'high', 'low', 'close', 'volume']]
            
            if not new_df.empty:
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df.sort_index(inplace=True)
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
