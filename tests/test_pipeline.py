from data.fetcher import fetch_universe
from features.pipeline import build_dataset, stationarity_report

# Load data
data = fetch_universe()

# Build features
X, y = build_dataset(data)

print(f'Feature matrix shape: {X.shape}')
print(f'Target vector shape: {y.shape}')
print(f'\nFeatures: {list(X.columns)}')
print(f'\nSample X:')
print(X.head(10))
print(f'\nTarget stats:')
print(y.describe())