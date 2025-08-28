# src/data_fetcher.py

import pandas as pd
import yfinance as yf
import time
import os

class DataFetcher:
    """A generic class to fetch data from Yahoo Finance."""

    def fetch_data(self, ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
        """Fetches historical OHLC data from Yahoo Finance."""
        print(f"📡 Attempting to fetch '{ticker}' from Yahoo Finance...")
        retries = 0
        while retries < max_retries:
            try:
                asset = yf.Ticker(ticker)
                # Usiamo un periodo lungo per assicurarci di avere dati anche per ticker con meno storia
                df = asset.history(start=start, end=end)
                
                if df.empty:
                    raise ValueError("No data returned from Yahoo Finance")
                
                # Pulisce i nomi delle colonne
                df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
                if 'adj_close' in df.columns:
                    df.rename(columns={'adj_close': 'close'}, inplace=True)
                
                # Assicura la presenza delle colonne necessarie
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
                time.sleep(retries * 2) # Backoff esponenziale
        
        return pd.DataFrame() # Ritorna un DataFrame vuoto in caso di fallimento
