import yfinance as yf
import pandas as pd
from typing import Optional
from utils.config import DATA_DIR
import os

class DataFetcher:
    """Class to handle fetching and caching of stock data."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache

    def fetch_data(self, ticker: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        cache_path = os.path.join(DATA_DIR, f"{ticker}_{start_date}_{end_date}.csv")
        
        if self.use_cache and os.path.exists(cache_path):
            return pd.read_csv(cache_path, index_col="Date", parse_dates=True)

        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}.")
        
        if self.use_cache:
            data.to_csv(cache_path)
            
        return data
