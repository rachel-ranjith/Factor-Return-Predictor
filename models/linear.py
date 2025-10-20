"""Linear models for return prediction."""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class ModelResult:
    """Container for model predictions and metadata."""
    predictions: pd.Series
    coefficients: pd.Series
    intercept: float
    model_name: str


def fit_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = 1.0,
) -> ModelResult:
    """
    Fit Ridge regression and predict.
    
    Ridge = OLS + L2 penalty
    Good when features are correlated (shrinks coefficients).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    predictions = pd.Series(
        model.predict(X_test_scaled),
        index=X_test.index,
        name="prediction",
    )
    
    coefficients = pd.Series(
        model.coef_,
        index=X_train.columns,
        name="coefficient",
    )
    
    return ModelResult(
        predictions=predictions,
        coefficients=coefficients,
        intercept=model.intercept_,
        model_name=f"Ridge(alpha={alpha})",
    )


def fit_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = 0.01,
) -> ModelResult:
    """
    Fit Lasso regression and predict.
    
    Lasso = OLS + L1 penalty
    Good for feature selection (drives coefficients to zero).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_scaled, y_train)
    
    predictions = pd.Series(
        model.predict(X_test_scaled),
        index=X_test.index,
        name="prediction",
    )
    
    coefficients = pd.Series(
        model.coef_,
        index=X_train.columns,
        name="coefficient",
    )
    
    return ModelResult(
        predictions=predictions,
        coefficients=coefficients,
        intercept=model.intercept_,
        model_name=f"Lasso(alpha={alpha})",
    )


def fit_ols(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> ModelResult:
    """
    Fit ordinary least squares (no regularization).
    
    Baseline model - prone to overfitting with many features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    predictions = pd.Series(
        model.predict(X_test_scaled),
        index=X_test.index,
        name="prediction",
    )
    
    coefficients = pd.Series(
        model.coef_,
        index=X_train.columns,
        name="coefficient",
    )
    
    return ModelResult(
        predictions=predictions,
        coefficients=coefficients,
        intercept=model.intercept_,
        model_name="OLS",
    )