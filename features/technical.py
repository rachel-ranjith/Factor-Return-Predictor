"""Technical indicators."""

import pandas as pd
import numpy as np

from config import config


def rsi(prices: pd.Series, window: int = None) -> pd.Series:
    """
    Relative Strength Index.
    
    RSI = 100 - (100 / (1 + RS))
    RS = avg gain / avg loss over window
    
    RSI > 70 = overbought
    RSI < 30 = oversold
    """
    if window is None:
        window = config.features.rsi_window
    
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD indicator.
    
    Returns MACD line, signal line, and histogram.
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    })


def bollinger_position(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Position within Bollinger Bands (0 to 1).
    
    0 = at lower band, 1 = at upper band, 0.5 = at middle
    Values outside [0,1] indicate breakout.
    """
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    
    return (prices - lower) / (upper - lower)


def technical_features(prices: pd.Series) -> pd.DataFrame:
    """Combine all technical features."""
    macd_df = macd(prices)
    
    return pd.concat([
        pd.DataFrame({
            "rsi": rsi(prices),
            "bollinger_pos": bollinger_position(prices),
        }),
        macd_df,
    ], axis=1)