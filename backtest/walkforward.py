"""Walk-forward validation for time series."""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass

from config import config
from models.linear import fit_ridge, fit_lasso, fit_ols, ModelResult
from models.evaluation import evaluate_predictions


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""
    predictions: pd.Series
    actuals: pd.Series
    metrics_by_period: pd.DataFrame
    coefficients_by_period: pd.DataFrame
    overall_metrics: dict


def get_walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_months: int = None,
    test_months: int = None,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate train/test splits for walk-forward validation.
    
    Returns list of (train_dates, test_dates) tuples.
    """
    if train_months is None:
        train_months = config.model.train_months
    if test_months is None:
        test_months = config.model.test_months
    
    unique_dates = dates.unique().sort_values()
    min_date = unique_dates.min()
    max_date = unique_dates.max()
    
    splits = []
    
    # Start after initial training period
    current_start = min_date + relativedelta(months=train_months)
    
    while current_start < max_date:
        # Training period: train_months before current_start
        train_end = current_start - relativedelta(days=1)
        train_start = current_start - relativedelta(months=train_months)
        
        # Test period: test_months after current_start
        test_start = current_start
        test_end = current_start + relativedelta(months=test_months) - relativedelta(days=1)
        
        # Get actual dates in our data
        train_mask = (unique_dates >= train_start) & (unique_dates <= train_end)
        test_mask = (unique_dates >= test_start) & (unique_dates <= test_end)
        
        train_dates = unique_dates[train_mask]
        test_dates = unique_dates[test_mask]
        
        if len(train_dates) > 0 and len(test_dates) > 0:
            splits.append((train_dates, test_dates))
        
        # Roll forward
        current_start += relativedelta(months=test_months)
    
    return splits


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "ridge",
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Run walk-forward validation.
    
    model_type: 'ridge', 'lasso', or 'ols'
    """
    # Get date level of MultiIndex
    dates = X.index.get_level_values("date")
    splits = get_walk_forward_splits(dates)
    
    if verbose:
        print(f"Walk-forward: {len(splits)} periods")
        print(f"Train: {config.model.train_months}mo, Test: {config.model.test_months}mo")
    
    all_predictions = []
    all_actuals = []
    period_metrics = []
    period_coefficients = []
    
    for i, (train_dates, test_dates) in enumerate(splits):
        # Split data
        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        # Fit model
        if model_type == "ridge":
            result = fit_ridge(X_train, y_train, X_test, alpha=config.model.alpha_ridge)
        elif model_type == "lasso":
            result = fit_lasso(X_train, y_train, X_test, alpha=config.model.alpha_lasso)
        else:
            result = fit_ols(X_train, y_train, X_test)
        
        # Evaluate
        metrics = evaluate_predictions(y_test, result.predictions)
        metrics["period"] = i
        metrics["test_start"] = test_dates.min()
        metrics["test_end"] = test_dates.max()
        
        period_metrics.append(metrics)
        
        # Store coefficients
        coef_row = result.coefficients.to_dict()
        coef_row["period"] = i
        period_coefficients.append(coef_row)
        
        # Collect predictions
        all_predictions.append(result.predictions)
        all_actuals.append(y_test)
        
        if verbose:
            print(f"  Period {i+1}: {metrics['test_start'].date()} to {metrics['test_end'].date()} | "
                  f"IC: {metrics['ic']:.3f} | Hit: {metrics['hit_rate']:.1%}")
    
    # Combine all predictions
    predictions = pd.concat(all_predictions)
    actuals = pd.concat(all_actuals)
    
    # Overall metrics
    overall = evaluate_predictions(actuals, predictions)
    
    if verbose:
        print(f"\nOverall: IC: {overall['ic']:.4f} | Hit Rate: {overall['hit_rate']:.2%} | R²: {overall['r_squared']:.4f}")
    
    return WalkForwardResult(
        predictions=predictions,
        actuals=actuals,
        metrics_by_period=pd.DataFrame(period_metrics),
        coefficients_by_period=pd.DataFrame(period_coefficients),
        overall_metrics=overall,
    )