"""Feature pipeline: combines all features with proper lagging."""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

from config import config
from features.price import price_features
from features.volume import volume_features
from features.technical import technical_features


def compute_features(prices: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """
    Compute all features for a single ticker.
    
    Returns DataFrame with features as columns, dates as index.
    """
    price_feat = price_features(prices)
    vol_feat = volume_features(prices, volume)
    tech_feat = technical_features(prices)
    
    features = pd.concat([price_feat, vol_feat, tech_feat], axis=1)
    
    return features


def compute_target(prices: pd.Series, horizon: int = None) -> pd.Series:
    """
    Compute forward return as prediction target.
    
    target_t = (price_{t+horizon} / price_t) - 1
    
    Note: This creates lookahead - must be used carefully in backtesting.
    """
    if horizon is None:
        horizon = config.model.target_horizon
    
    return prices.pct_change(horizon).shift(-horizon)


def lag_features(features: pd.DataFrame, lag: int = None) -> pd.DataFrame:
    """
    Lag features to avoid lookahead bias.
    
    At time t, we can only use features computed from data up to t-1.
    """
    if lag is None:
        lag = config.features.feature_lag
    
    return features.shift(lag)


def check_stationarity(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Run ADF test for stationarity.
    
    Returns dict with test statistic, p-value, and whether stationary.
    """
    clean = series.dropna()
    
    if len(clean) < 20:
        return {"statistic": np.nan, "pvalue": np.nan, "stationary": False}
    
    result = adfuller(clean, autolag="AIC")
    
    return {
        "statistic": result[0],
        "pvalue": result[1],
        "stationary": result[1] < significance,
    }


def stationarity_report(features: pd.DataFrame) -> pd.DataFrame:
    """Check stationarity of all features."""
    results = []
    
    for col in features.columns:
        stat = check_stationarity(features[col])
        stat["feature"] = col
        results.append(stat)
    
    return pd.DataFrame(results).set_index("feature")


def build_dataset(
    data: dict[str, pd.DataFrame],
    lag_features_flag: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build full feature matrix and target vector.
    
    Returns:
        X: DataFrame with (date, ticker) MultiIndex and features as columns
        y: Series with same index containing forward returns
    """
    all_features = []
    all_targets = []
    
    for ticker, df in data.items():
        prices = df["Close"]
        volume = df["Volume"]
        
        # Compute features
        features = compute_features(prices, volume)
        
        # Lag to avoid lookahead
        if lag_features_flag:
            features = lag_features(features)
        
        # Compute target
        target = compute_target(prices)
        
        # Add ticker identifier
        features["ticker"] = ticker
        
        # Align and combine
        combined = features.copy()
        combined["target"] = target
        
        all_features.append(combined)
    
    # Stack all tickers
    full_df = pd.concat(all_features)
    
    # Create MultiIndex
    full_df = full_df.reset_index()
    full_df = full_df.rename(columns={"index": "date", "Date": "date"})
    full_df = full_df.set_index(["date", "ticker"])
    
    # Separate features and target
    y = full_df["target"]
    X = full_df.drop(columns=["target"])
    
    # Drop rows with any NaN
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]
    
    return X, y


def get_feature_names() -> list[str]:
    """Return list of all feature names."""
    return [
        # Momentum
        "mom_1d", "mom_5d", "mom_10d", "mom_21d", "mom_63d",
        # Volatility
        "volatility_21d", "volatility_63d",
        # Mean reversion
        "mr_zscore_21d", "mr_zscore_63d",
        # Volume
        "volume_zscore", "volume_price_corr", "log_dollar_volume",
        # Technical
        "rsi", "bollinger_pos", "macd", "macd_signal", "macd_hist",
    ]