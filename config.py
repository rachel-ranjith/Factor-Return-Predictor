from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DataConfig:
    start_date: date = date(2020, 1, 1)
    end_date: date = date(2024, 12, 31)
    cache_dir: str = ".cache"


@dataclass(frozen=True)
class FeatureConfig:
    # momentum lookbacks
    momentum_windows: tuple = (1, 5, 10, 21, 63)  # days
    
    # volatility
    volatility_window: int = 21
    
    # volume
    volume_window: int = 21
    
    # mean reversion
    mean_reversion_window: int = 21
    
    # technical indicators
    rsi_window: int = 14
    
    # lag features to avoid lookahead
    feature_lag: int = 1


@dataclass(frozen=True)
class ModelConfig:
    # target
    target_horizon: int = 5  # predict 5-day forward return
    
    # walk-forward
    train_months: int = 12
    test_months: int = 1
    
    # regularization
    alpha_ridge: float = 1.0
    alpha_lasso: float = 0.01


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()


config = Config()