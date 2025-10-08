"""Volume-based features."""

import pandas as pd
import numpy as np

from config import config


def volume_zscore(volume: pd.Series, window: int = None) -> pd.Series:
    """
    Z-score of volume vs rolling average.
    
    High z-score = unusual volume = potential signal
    """
    if window is None:
        window = config.features.volume_window
    
    rolling_mean = volume.rolling(window).mean()
    rolling_std = volume.rolling(window).std()
    
    return (volume - rolling_mean) / rolling_std


def volume_price_corr(prices: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """
    Rolling correlation between returns and volume.
    
    Positive correlation = volume confirms price moves
    Negative correlation = divergence (potential reversal)
    """
    returns = prices.pct_change()
    
    return returns.rolling(window).corr(volume)


def dollar_volume(prices: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """
    Rolling average dollar volume (liquidity proxy).
    
    Log-transformed for better distribution.
    """
    dv = prices * volume
    return np.log(dv.rolling(window).mean())


def volume_features(prices: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Combine all volume-based features."""
    return pd.DataFrame({
        "volume_zscore": volume_zscore(volume),
        "volume_price_corr": volume_price_corr(prices, volume),
        "log_dollar_volume": dollar_volume(prices, volume),
    })