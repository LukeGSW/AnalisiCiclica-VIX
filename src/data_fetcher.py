# src/data_fetcher.py

"""
Data fetching module for Kriterion Quant Trading System
Handles all data acquisition from Yahoo Finance
"""

import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime

from config import Config

def _get_active_vix_future_ticker() -> str:
    """
    Calculates the ticker for the current or next active VIX future contract.
    VIX futures tickers use CBOE month codes. This is a robust way to get
    the front-month contract when the continuous ticker 'VX=F' is unreliable.
    """
    now = datetime.now()
    month = now.month
    year = now.year

    # CBOE month codes
    month_codes = "FGHJKMNQUVXZ"
    
    # Heuristic: VIX futures expire mid-month. To be safe, if we are past
    # the 10th of the month, we look for the next month's contract to ensure
    # we're fetching the most liquid one (the front-month).
    if now.day > 10:
        month += 1
        if month > 12:
            month = 1
            year += 1
            
    code = month_codes[month - 1]
    ticker = f"VX{code}{str(year)[-2:]}=F"
    return ticker


class DataFetcher:
    """Class to handle data fetching from Yahoo Finance"""

    def __init__(self, data_path: str):
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
        """
        ticker = ticker or Config.TICKER
        start_date = start_date or Config.START_DATE
        end_date = end_date or Config.END_DATE
        
        # === LOGICA DI RISOLUZIONE DINAMICA DEL TICKER ===
        # Se si richiede il future VIX continuo, lo sostituiamo con il contratto attivo.
        # Questo bypassa l'inaffidabilità del ticker generico 'VX=F'.
        effective_ticker = ticker
        if ticker == 'VX=F':
            effective_ticker = _get_active_vix_future_ticker()
            print(f"ℹ️ Continuous VIX future ('VX=F') requested.")
            print(f"   Dynamically resolving to active contract: {effective_ticker}")
        # =================================================

        print(f"📡 Fetching data for {effective_ticker} from {start_date} to {end_date} via Yahoo Finance...")

        retries = 0
        backoff_factor = 2
        
        while retries < max_retries:
            try:
                asset = yf.Ticker(effective_ticker)
                df = asset.history(start=start_date, end=end_date)

                if df.empty:
                    # Se il contratto dinamico fallisce, proviamo con quello del mese successivo come ultima spiaggia
                    if ticker == 'VX=F' and retries == 1:
                         print("   Active contract fetch failed, trying next month's contract...")
                         current_month_code_index = "FGHJKMNQUVXZ".find(effective_ticker[2])
                         next_month_code_index = (current_month_code_index + 1) % 12
                         next_year = int(effective_ticker[3:5])
                         if next_month_code_index == 0: next_year +=1
                         next_code = "FGHJKMNQUVXZ"[next_month_code_index]
                         effective_ticker = f"VX{next_code}{next_year}=F"
                         print(f"   New attempt with: {effective_ticker}")
                         continue # Riprova il ciclo con il nuovo ticker

                    raise ValueError(f"No data returned for {effective_ticker}.")

                df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
                
                if 'adj_close' in df.columns:
                    df.rename(columns={'adj_close': 'close'}, inplace=True)

                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns after download for {effective_ticker}")
                
                df = df[required_cols]
                df.dropna(inplace=True)
                
                print(f"✅ Successfully fetched {len(df)} days of data for {effective_ticker}")
                return df

            except Exception as e:
                retries += 1
                print(f"❌ Error fetching data: {e}. Retrying ({retries}/{max_retries})...")
                if retries < max_retries:
                    time.sleep(backoff_factor * retries)
                else:
                    raise Exception(f"Failed to fetch data for {effective_ticker} after {max_retries} retries.")
        
        raise Exception("Failed to fetch data due to an unknown issue.")

    # Il resto della classe (save_data, load_data, update_latest_data) rimane invariato
    # ... (copia qui il resto dei metodi della tua classe che non ho bisogno di modificare)

    def save_data(self, df: pd.DataFrame) -> str:
        """
        Save DataFrame to 'historical_data.csv' inside the data_path directory.
        """
        filename = 'historical_data.csv'
        filepath = os.path.join(self.data_path, filename)
        
        # Assicura che l'indice si chiami 'date' prima di salvare
        df.index.name = 'date'
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
        
        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        
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
            
            # Utilizziamo il giorno successivo per evitare sovrapposizioni
            start_update_date = (existing_df.index[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            new_df = self.fetch_historical_data(
                ticker=ticker,
                start_date=start_update_date,
                end_date=Config.END_DATE
            )
            
            if not new_df.empty:
                combined_df = pd.concat([existing_df, new_df])
                combined_df.sort_index(inplace=True)
            else:
                print("No new data to combine.")
                combined_df = existing_df
        
        except FileNotFoundError:
            print("📥 No existing data found. Fetching full history...")
            combined_df = self.fetch_historical_data(ticker=ticker)
        
        self.save_data(combined_df)
        return combined_df
