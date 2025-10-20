"""Model evaluation metrics."""

import pandas as pd
import numpy as np
from scipy import stats


def r_squared(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Out-of-sample R-squared.
    
    R² = 1 - SS_res / SS_tot
    
    Can be negative if model is worse than predicting mean.
    """
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    
    return 1 - ss_res / ss_tot


def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Information Coefficient (IC) = Spearman correlation.
    
    Measures rank correlation between predictions and actuals.
    More robust than Pearson for return prediction.
    """
    return stats.spearmanr(y_true, y_pred)[0]


def hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Directional accuracy: % of times we got the sign right.
    
    Random = 50%, good model > 52-55%.
    """
    correct = (np.sign(y_true) == np.sign(y_pred)).sum()
    return correct / len(y_true)


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error."""
    return (y_true - y_pred).abs().mean()


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute all evaluation metrics."""
    return {
        "r_squared": r_squared(y_true, y_pred),
        "ic": information_coefficient(y_true, y_pred),
        "hit_rate": hit_rate(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "n_samples": len(y_true),
    }


def format_metrics(metrics: dict) -> str:
    """Pretty print metrics."""
    return (
        f"R²: {metrics['r_squared']:.4f} | "
        f"IC: {metrics['ic']:.4f} | "
        f"Hit Rate: {metrics['hit_rate']:.2%} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"N: {metrics['n_samples']}"
    )