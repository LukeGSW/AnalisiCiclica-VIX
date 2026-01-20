# src/data_fetcher.py

import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime, timedelta # Aggiunto timedelta

from config import Config

class DataFetcher:
    """
    A generic class to fetch, save, and load data from Yahoo Finance.
    """

    def __init__(self, data_path: str):
        """
        Initializes the data fetcher.
        """
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

    def fetch_historical_data(self, ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
        """
        Fetches historical OHLC data from Yahoo Finance using a robust method.
        """
        print(f"📡 Attempting to fetch '{ticker}' from Yahoo Finance ({start} -> {end})...")
        retries = 0
        while retries < max_retries:
            try:
                # Using yf.Ticker().history() is generally stable for indices
                asset = yf.Ticker(ticker)
                df = asset.history(start=start, end=end, auto_adjust=True)
                
                # FIX: Se il dataframe è vuoto, non è necessariamente un errore critico (es. vacanza)
                # Restituiamo vuoto e lasciamo gestire al chiamante, oppure solleviamo errore solo se retries finiti
                if df.empty:
                    print(f"⚠️ Warning: No data returned for range {start} to {end}. Market might be closed.")
                    # Non solleviamo subito ValueError, lasciamo riprovare o uscire
                    if retries == max_retries - 1:
                         return pd.DataFrame() 
                    else:
                         raise ValueError("Empty DataFrame received")
                
                # Clean column names
                df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
                
                # Ensure all required columns are present
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                # Alcuni indici come il VIX potrebbero non avere volume, gestiamo l'eccezione se necessario
                # Per ora manteniamo il check ma logghiamo warning invece di crashare se manca volume
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                     if 'volume' in missing_cols and ticker == '^VIX':
                         df['volume'] = 0 # Fix specifico per VIX che a volte non ha volume
                     else:
                        raise ValueError(f"Data for {ticker} is missing columns: {missing_cols}")
                
                print(f"✅ Successfully fetched {len(df)} rows for '{ticker}'.")
                return df[required_cols].dropna()

            except Exception as e:
                retries += 1
                print(f"❌ Failed to fetch '{ticker}' (Attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    # Invece di crashare tutto, restituiamo DataFrame vuoto se fallisce
                    print(f"⚠️ Skipping update for now due to fetch errors.")
                    return pd.DataFrame()
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
            
            # Start date: giorno dopo l'ultimo dato
            start_update_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            # FIX PRINCIPALE: End date deve includere oggi, quindi mettiamo domani
            # yfinance 'end' is exclusive
            end_update_date = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            print(f"📅 Last data point is {last_date.strftime('%Y-%m-%d')}. Fetching new data...")
            
            # Controllo preventivo: Se la start date è nel futuro (o oggi e mercato non ancora chiuso), attenzione
            if start_update_date >= end_update_date:
                print("✅ Data is already up to date.")
                return existing_df

            new_df = self.fetch_historical_data(
                ticker=ticker,
                start=start_update_date,
                end=end_update_date
            )
            
            if not new_df.empty:
                # Rimuove duplicati basandosi sull'indice
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                self.save_data(combined_df)
                return combined_df
            else:
                print("✅ No new data found (Market likely closed or holiday). Using existing data.")
                return existing_df

        except FileNotFoundError:
            print("📥 No existing data found. Fetching full history...")
            
            # Anche qui applichiamo la logica dell'end date inclusiva
            end_date = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            full_df = self.fetch_historical_data(
                ticker=ticker,
                start=Config.START_DATE,
                end=end_date 
            )
            if not full_df.empty:
                self.save_data(full_df)
                return full_df
            else:
                raise ValueError("Could not fetch initial historical data.")
