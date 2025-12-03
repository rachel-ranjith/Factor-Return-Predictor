from data.fetcher import fetch_universe
from features.pipeline import build_dataset
from backtest.walkforward import walk_forward_validate
from backtest.metrics import generate_report

# Load data
data = fetch_universe()
X, y = build_dataset(data)

# Run models and generate reports
print("\n" + "=" * 60)
print("RIDGE REGRESSION")
print("=" * 60)
ridge_result = walk_forward_validate(X, y, model_type="ridge")
generate_report(ridge_result, "Ridge")

print("\n" + "=" * 60)
print("OLS (no regularization)")
print("=" * 60)
ols_result = walk_forward_validate(X, y, model_type="ols")
generate_report(ols_result, "OLS")

print("\nPlots saved to analysis/plots/")