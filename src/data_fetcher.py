# Esempio di implementazione con EODHD in data_fetcher.py
import requests
import pandas as pd
from config import Config # Assicurati di avere la tua API_KEY di EODHD qui

# ... all'interno della classe DataFetcher ...

def fetch_historical_data_eodhd(
    self,
    ticker: str = 'VX.FUT', # Ticker EODHD per il future VIX
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    
    ticker = ticker # Puoi renderlo dinamico se vuoi
    start_date = start_date or Config.START_DATE
    end_date = end_date or Config.END_DATE
    api_key = Config.EODHD_API_KEY # DEVI AGGIUNGERE QUESTO AL TUO CONFIG

    url = f"https://eodhd.com/api/eod/{ticker}?api_token={api_key}&from={start_date}&to={end_date}&period=d&fmt=json"
    
    print(f"📡 Fetching data for {ticker} from EODHD...")

    try:
        response = requests.get(url)
        response.raise_for_status() # Lancia un'eccezione per errori HTTP
        data = response.json()
        
        if not data:
            raise ValueError(f"No data returned from EODHD for {ticker}")
            
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Rinomina le colonne per essere compatibile con il resto del tuo codice
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }, inplace=True)
        
        # Rimuovi la colonna 'adjusted_close' se esiste
        if 'adjusted_close' in df.columns:
            df = df.drop(columns=['adjusted_close'])

        print(f"✅ Successfully fetched {len(df)} days of data from EODHD")
        return df

    except Exception as e:
        print(f"❌ Error fetching data from EODHD: {e}")
        raise
