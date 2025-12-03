#!/usr/bin/env python3
"""
Factor Return Predictor

A walk-forward validated factor model for predicting short-term stock returns.
Demonstrates proper ML methodology for time series: no lookahead bias, 
rolling validation, and regularization.

Usage:
    python main.py                    # Run full pipeline with defaults
    python main.py --model ridge      # Specific model
    python main.py --no-plots         # Skip plot generation
    python main.py --refresh-data     # Force re-download data
"""

import argparse
from pathlib import Path

from data.fetcher import fetch_universe
from data.universe import UNIVERSE
from features.pipeline import build_dataset, stationarity_report
from backtest.walkforward import walk_forward_validate
from backtest.metrics import generate_report
from config import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Factor-based return prediction with walk-forward validation"
    )
    parser.add_argument(
        "--model",
        choices=["ridge", "lasso", "ols", "all"],
        default="all",
        help="Model to run (default: all)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force re-download of market data",
    )
    parser.add_argument(
        "--check-stationarity",
        action="store_true",
        help="Run stationarity tests on features",
    )
    return parser.parse_args()


def print_config():
    """Display current configuration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Universe:        {len(UNIVERSE)} tickers")
    print(f"Date range:      {config.data.start_date} to {config.data.end_date}")
    print(f"Target horizon:  {config.model.target_horizon} days")
    print(f"Train window:    {config.model.train_months} months")
    print(f"Test window:     {config.model.test_months} month")
    print(f"Feature lag:     {config.features.feature_lag} day(s)")
    print(f"Ridge alpha:     {config.model.alpha_ridge}")
    print(f"Lasso alpha:     {config.model.alpha_lasso}")


def main():
    args = parse_args()
    
    print_config()
    
    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    use_cache = not args.refresh_data
    data = fetch_universe(use_cache=use_cache)
    
    if not data:
        print("ERROR: No data available. Exiting.")
        return 1
    
    # Build features
    print("\n" + "=" * 60)
    print("BUILDING FEATURES")
    print("=" * 60)
    
    X, y = build_dataset(data)
    print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Target stats:   mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Optional stationarity check
    if args.check_stationarity:
        print("\n" + "=" * 60)
        print("STATIONARITY TESTS")
        print("=" * 60)
        
        # Run on first ticker's features as sample
        first_ticker = list(data.keys())[0]
        sample_X = X.xs(first_ticker, level="ticker")
        report = stationarity_report(sample_X)
        
        print(f"\nResults for {first_ticker}:")
        print(report.to_string())
        
        stationary_pct = report["stationary"].mean() * 100
        print(f"\n{stationary_pct:.0f}% of features are stationary (ADF p < 0.05)")
    
    # Run models
    models_to_run = ["ridge", "lasso", "ols"] if args.model == "all" else [args.model]
    results = {}
    
    for model_name in models_to_run:
        print("\n" + "=" * 60)
        print(f"{model_name.upper()} REGRESSION")
        print("=" * 60)
        
        result = walk_forward_validate(X, y, model_type=model_name)
        results[model_name] = result
        
        if not args.no_plots:
            generate_report(result, model_name.capitalize())
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"{'Model':<10} {'IC':>10} {'Hit Rate':>12} {'R²':>10}")
        print("-" * 44)
        
        for name, result in results.items():
            m = result.overall_metrics
            print(f"{name:<10} {m['ic']:>10.4f} {m['hit_rate']:>11.2%} {m['r_squared']:>10.4f}")
    
    if not args.no_plots:
        print(f"\nPlots saved to: analysis/plots/")
    
    print("\nDone.")
    return 0


if __name__ == "__main__":
    exit(main())