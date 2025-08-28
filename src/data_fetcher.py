# src/data_fetcher.py

import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime

from config import Config

class DataFetcher:
    """
    A robust class to fetch and clean historical data from Yahoo Finance,
    with manual adjustment and sanity checks to ensure data integrity.
    """

    def __init__(self, data_path: str):
        """
        Initializes the data fetcher.
        """
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

    def fetch_historical_data(self, ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
        """
        Fetches, manually adjusts, and validates historical OHLC data.
        This method is crucial for assets with frequent splits like VXX.
        """
        print(f"📡 Attempting to fetch '{ticker}' from Yahoo Finance...")
        retries = 0
        while retries < max_retries:
            try:
                # Use yf.download as it can be more robust for complex tickers
                df = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    auto_adjust=False, # We need raw data to perform manual adjustment
                    progress=False,
                    actions=True # Ensure splits/dividends are included
                )
                
                if df.empty or 'Adj Close' not in df.columns:
                    raise ValueError("No data returned or 'Adj Close' column is missing.")

                # Manually adjust the data to prevent errors
                df_adj = pd.DataFrame(index=df.index)
                
                # The adjustment ratio correctly accounts for splits and dividends
                adjustment_ratio = df['Adj Close'] / df['Close']
                
                df_adj['open'] = df['Open'] * adjustment_ratio
                df_adj['high'] = df['High'] * adjustment_ratio
                df_adj['low'] = df['Low'] * adjustment_ratio
                df_adj['close'] = df['Adj Close'] # The adjusted close is the ground truth
                df_adj['volume'] = df['Volume']
                df_adj.dropna(inplace=True)

                # ================================================================= #
                #               <<< SANITY CHECK FONDAMENTALE >>>                 #
                # ================================================================= #
                # Controlla se il prezzo più recente è plausibile.
                # VXX non è sopra i $150 da anni. Se lo è, i dati non sono aggiustati.
                last_price = df_adj['close'].iloc[-1]
                if last_price > 150: # Soglia di sicurezza molto alta
                    raise ValueError(
                        f"Data for {ticker} appears unadjusted and corrupted. "
                        f"Last close price is {last_price:.2f}, which is unrealistic. "
                        "This is a known issue with yfinance for this ticker in some environments."
                    )
                # ================================================================= #
                
                print(f"✅ Successfully fetched and adjusted {len(df_adj)} rows for '{ticker}'.")
                return df_adj[['open', 'high', 'low', 'close', 'volume']]

            except Exception as e:
                retries += 1
                print(f"❌ Failed to fetch '{ticker}' (Attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"Could not fetch data for '{ticker}' after {max_retries} attempts. The data source is likely unreliable for this ticker in this environment.")
                time.sleep(retries * 2)
        
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
            last_date = existing_df.index[-1]
            start_update_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"📅 Last data point is {last_date.strftime('%Y-%m-%d')}. Fetching new data...")
            
            new_df = self.fetch_historical_data(
                ticker=ticker,
                start=start_update_date,
                end=datetime.now().strftime('%Y-%m-%d')
            )
            
            if not new_df.empty:
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
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
