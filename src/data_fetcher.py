# src/data_fetcher.py

"""
Data fetching module for Kriterion Quant Trading System
Handles all data acquisition from Yahoo Finance
"""

import pandas as pd
import yfinance as yf
import time
import os

from config import Config

class DataFetcher:
    """Class to handle data fetching from Yahoo Finance"""

    def __init__(self, data_path: str):
        """
        Initialize the data fetcher.

        Parameters
        ----------
        data_path : str
            The path to the directory where data should be saved/loaded.
        """
        # API key non è più necessaria con Yahoo Finance
        self.data_path = data_path

    def fetch_historical_data(
        self,
        ticker: str = None,
        start_date: str = None,
        end_date: str = None,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data from Yahoo Finance.

        Parameters
        ----------
        ticker : str, optional
            Stock ticker symbol. Defaults to Config.TICKER.
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format. Defaults to Config.START_DATE.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format. Defaults to Config.END_DATE.
        max_retries : int, optional
            Maximum number of retry attempts for network issues.

        Returns
        -------
        pd.DataFrame
            DataFrame with OHLC data and volume.
        """
        ticker = ticker or Config.TICKER
        start_date = start_date or Config.START_DATE
        end_date = end_date or Config.END_DATE

        print(f"📡 Fetching data for {ticker} from {start_date} to {end_date} via Yahoo Finance...")

        retries = 0
        backoff_factor = 2
        
        while retries < max_retries:
            try:
                # Usa yfinance per scaricare i dati
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if df.empty:
                    raise ValueError(f"No data returned for {ticker}. It might be delisted or the ticker is incorrect.")
                
                # Gestisce il caso in cui yfinance restituisca un MultiIndex nelle colonne
                if isinstance(df.columns, pd.MultiIndex):
                    # Appiattiamo le colonne prendendo solo il primo livello (es. 'Open', 'Close')
                    df.columns = df.columns.get_level_values(0)

                # Ora possiamo processare le colonne in sicurezza
                df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
                
                # Per gli indici come il VIX, 'adj_close' è uguale a 'close'.
                # Per mantenere la compatibilità, ci assicuriamo che la colonna 'close' sia quella di riferimento.
                if 'adj_close' in df.columns:
                    df.rename(columns={'adj_close': 'close'}, inplace=True)

                # Seleziona e valida le colonne richieste dal sistema
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                
                # Controlla che tutte le colonne necessarie esistano dopo la pulizia
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns after download for {ticker}")
                
                df = df[required_cols]
                
                # Rimuovi eventuali righe con dati mancanti
                df.dropna(inplace=True)
                
                print(f"✅ Successfully fetched {len(df)} days of data")
                return df

            except Exception as e:
                retries += 1
                print(f"❌ Error fetching data: {e}. Retrying ({retries}/{max_retries})...")
                if retries < max_retries:
                    time.sleep(backoff_factor * retries)
                else:
                    raise Exception(f"Failed to fetch data for {ticker} after {max_retries} retries.")
        
        raise Exception("Failed to fetch data due to an unknown issue.")

    def save_data(self, df: pd.DataFrame) -> str:
        """
        Save DataFrame to 'historical_data.csv' inside the data_path directory.
        """
        filename = 'historical_data.csv'
        filepath = os.path.join(self.data_path, filename)
        
        df.to_csv(filepath)
        print(f"💾 Data saved to {filepath}")
        return filepath

    def load_data(self) -> pd.DataFrame:
        """
        Load DataFrame from 'historical_data.csv' inside the data_path directory.
        """
        filename = 'historical_data.csv'
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True) # yfinance usa 'Date' come index name
        df.index.name = 'date' # Standardizziamo il nome dell'indice
        print(f"📂 Loaded {len(df)} days of data from {filepath}")
        return df

    def update_latest_data(self, ticker: str = None) -> pd.DataFrame:
        """
        Update data with the latest available information.
        """
        ticker = ticker or Config.TICKER
        
        try:
            existing_df = self.load_data()
            last_date = existing_df.index[-1].strftime('%Y-%m-%d')
            print(f"📅 Last data point: {last_date}")
            
            # Fetch only new data
            new_df = self.fetch_historical_data(
                ticker=ticker,
                start_date=last_date,
                end_date=Config.END_DATE
            )
            
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)
        
        except FileNotFoundError:
            print("📥 No existing data found. Fetching full history...")
            combined_df = self.fetch_historical_data(ticker=ticker)
        
        # Save updated data
        self.save_data(combined_df)
        return combined_df
