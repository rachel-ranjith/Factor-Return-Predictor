"""Data fetching with local caching."""

import os
import time
import pandas as pd
import yfinance as yf
from pathlib import Path

from config import config
from data.universe import UNIVERSE


def get_cache_path(ticker: str) -> Path:
    """Return cache file path for a ticker."""
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{ticker}.parquet"


def fetch_universe_batch(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """
    Fetch all tickers in a single batch request.
    """
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    to_fetch = []
    data = {}
    
    for ticker in UNIVERSE:
        cache_path = get_cache_path(ticker)
        if use_cache and cache_path.exists():
            data[ticker] = pd.read_parquet(cache_path)
        else:
            to_fetch.append(ticker)
    
    if not to_fetch:
        print(f"All {len(UNIVERSE)} tickers loaded from cache")
        return data
    
    print(f"Fetching {len(to_fetch)} tickers from Yahoo Finance...")
    
    df = yf.download(
        to_fetch,
        start=config.data.start_date,
        end=config.data.end_date,
        progress=True,
        auto_adjust=True,
        group_by="ticker",
        threads=False,
    )
    
    for ticker in to_fetch:
        try:
            if len(to_fetch) == 1:
                ticker_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                ticker_df = df[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
            
            if ticker_df.dropna().empty:
                print(f"Warning: No data for {ticker}")
                continue
                
            cache_path = get_cache_path(ticker)
            ticker_df.to_parquet(cache_path)
            data[ticker] = ticker_df
            
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")
    
    print(f"Fetched {len(data)}/{len(UNIVERSE)} tickers")
    return data


def fetch_universe(use_cache: bool = True, allow_synthetic: bool = True) -> dict[str, pd.DataFrame]:
    """
    Fetch data for all tickers.
    
    Falls back to synthetic data if Yahoo fails and allow_synthetic=True.
    """
    data = fetch_universe_batch(use_cache=use_cache)
    
    if len(data) == 0 and allow_synthetic:
        print("\nYahoo rate limited. Falling back to synthetic data...")
        from data.synthetic import generate_universe
        data = generate_universe()
    
    return data


def get_aligned_closes(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align close prices across all tickers."""
    closes = pd.DataFrame({
        ticker: df["Close"] for ticker, df in data.items()
    })
    return closes.dropna()


if __name__ == "__main__":
    data = fetch_universe()
    
    if data:
        closes = get_aligned_closes(data)
        print(f"\nAligned data shape: {closes.shape}")
        if not closes.empty:
            print(f"Date range: {closes.index[0]} to {closes.index[-1]}")
    else:
        print("No data fetched")