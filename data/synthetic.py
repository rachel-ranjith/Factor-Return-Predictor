"""Synthetic data generator for testing when Yahoo is rate limited."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import config
from data.universe import UNIVERSE, SECTOR_MAP


def generate_gbm_prices(
    start_price: float = 100.0,
    mu: float = 0.10,  # 10% annual drift
    sigma: float = 0.20,  # 20% annual volatility
    days: int = 1260,  # ~5 years
    seed: int = None,
) -> pd.Series:
    """
    Generate prices using Geometric Brownian Motion.
    
    dS = mu*S*dt + sigma*S*dW
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = 1 / 252  # daily
    
    # Generate random returns
    daily_returns = np.random.normal(
        loc=(mu - 0.5 * sigma**2) * dt,
        scale=sigma * np.sqrt(dt),
        size=days,
    )
    
    # Convert to prices
    prices = start_price * np.exp(np.cumsum(daily_returns))
    prices = np.insert(prices, 0, start_price)[:-1]
    
    return prices


def generate_volume(days: int, base_volume: float = 1e6, seed: int = None) -> np.ndarray:
    """Generate realistic volume with clustering."""
    if seed is not None:
        np.random.seed(seed)
    
    # Log-normal volume with some autocorrelation
    log_vol = np.random.normal(np.log(base_volume), 0.5, days)
    
    # Add some persistence (volume clustering)
    for i in range(1, days):
        log_vol[i] = 0.7 * log_vol[i-1] + 0.3 * log_vol[i]
    
    return np.exp(log_vol)


def generate_ohlcv(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    seed: int = None,
) -> pd.DataFrame:
    """Generate full OHLCV DataFrame for a ticker."""
    # Create date range (business days only)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    days = len(dates)
    
    # Use ticker hash for reproducible but different data per ticker
    if seed is None:
        seed = hash(ticker) % (2**31)
    
    np.random.seed(seed)
    
    # Vary parameters by "sector" for realism
    sector = SECTOR_MAP.get(ticker, "Other")
    sector_params = {
        "Tech": {"mu": 0.15, "sigma": 0.30},
        "Finance": {"mu": 0.08, "sigma": 0.22},
        "Healthcare": {"mu": 0.10, "sigma": 0.20},
        "Consumer": {"mu": 0.07, "sigma": 0.18},
        "Energy": {"mu": 0.05, "sigma": 0.28},
        "Industrial": {"mu": 0.08, "sigma": 0.20},
        "ETF": {"mu": 0.10, "sigma": 0.18},
        "Other": {"mu": 0.08, "sigma": 0.22},
    }
    params = sector_params.get(sector, sector_params["Other"])
    
    # Generate close prices
    close = generate_gbm_prices(
        start_price=np.random.uniform(50, 500),
        mu=params["mu"],
        sigma=params["sigma"],
        days=days,
        seed=seed,
    )
    
    # Generate OHLC from close
    daily_range = np.random.uniform(0.005, 0.025, days)  # 0.5% to 2.5% daily range
    
    high = close * (1 + daily_range * np.random.uniform(0.3, 1.0, days))
    low = close * (1 - daily_range * np.random.uniform(0.3, 1.0, days))
    
    # Open is close of previous day with gap
    gap = np.random.normal(0, 0.003, days)  # small overnight gaps
    open_prices = np.roll(close, 1) * (1 + gap)
    open_prices[0] = close[0] * (1 + gap[0])
    
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))
    
    volume = generate_volume(days, seed=seed + 1)
    
    df = pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)
    
    return df


def generate_universe() -> dict[str, pd.DataFrame]:
    """Generate synthetic data for entire universe."""
    data = {}
    
    start = datetime.combine(config.data.start_date, datetime.min.time())
    end = datetime.combine(config.data.end_date, datetime.min.time())
    
    for ticker in UNIVERSE:
        data[ticker] = generate_ohlcv(ticker, start, end)
    
    print(f"Generated synthetic data for {len(data)} tickers")
    return data


if __name__ == "__main__":
    data = generate_universe()
    
    # Show sample
    ticker = UNIVERSE[0]
    print(f"\n{ticker} sample:")
    print(data[ticker].head(10))
    print(f"\nShape: {data[ticker].shape}")
    print(f"Date range: {data[ticker].index[0]} to {data[ticker].index[-1]}")