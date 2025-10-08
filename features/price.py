"""Price-based features: momentum, volatility, mean reversion."""

import pandas as pd
import numpy as np

from config import config


def momentum(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate momentum as return over window.
    
    momentum_t = (price_t / price_{t-window}) - 1
    """
    return prices.pct_change(window)


def momentum_features(prices: pd.Series) -> pd.DataFrame:
    """Generate momentum features for all configured windows."""
    features = {}
    
    for window in config.features.momentum_windows:
        features[f"mom_{window}d"] = momentum(prices, window)
    
    return pd.DataFrame(features)


def volatility(returns: pd.Series, window: int = None) -> pd.Series:
    """
    Rolling realized volatility (standard deviation of returns).
    
    Annualized by sqrt(252).
    """
    if window is None:
        window = config.features.volatility_window
    
    return returns.rolling(window).std() * np.sqrt(252)


def volatility_features(prices: pd.Series) -> pd.DataFrame:
    """Generate volatility features."""
    returns = prices.pct_change()
    
    return pd.DataFrame({
        "volatility_21d": volatility(returns, 21),
        "volatility_63d": volatility(returns, 63),
    })


def mean_reversion_signal(prices: pd.Series, window: int = None) -> pd.Series:
    """
    Mean reversion signal: z-score of price vs rolling mean.
    
    Negative z-score = price below average = potential buy signal
    Positive z-score = price above average = potential sell signal
    """
    if window is None:
        window = config.features.mean_reversion_window
    
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    
    return (prices - rolling_mean) / rolling_std


def mean_reversion_features(prices: pd.Series) -> pd.DataFrame:
    """Generate mean reversion features."""
    return pd.DataFrame({
        "mr_zscore_21d": mean_reversion_signal(prices, 21),
        "mr_zscore_63d": mean_reversion_signal(prices, 63),
    })


def price_features(prices: pd.Series) -> pd.DataFrame:
    """Combine all price-based features."""
    mom = momentum_features(prices)
    vol = volatility_features(prices)
    mr = mean_reversion_features(prices)
    
    return pd.concat([mom, vol, mr], axis=1)