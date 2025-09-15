"""Stock universe definition."""

# Liquid large-caps across sectors
# Chosen for: high volume, tight spreads, diverse sector exposure

UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO", "LLY", "BMY", "AMGN", "GILD",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "HD",
    # Industrial / Energy
    "XOM", "CVX", "CAT", "BA", "UPS", "HON", "GE", "LMT", "UNP", "RTX",
]

SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech", "META": "Tech",
    "NVDA": "Tech", "TSLA": "Tech", "AMD": "Tech", "INTC": "Tech", "CRM": "Tech",
    "JPM": "Finance", "BAC": "Finance", "WFC": "Finance", "GS": "Finance", "MS": "Finance",
    "C": "Finance", "BLK": "Finance", "SCHW": "Finance", "AXP": "Finance", "V": "Finance",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare",
    "ABBV": "Healthcare", "TMO": "Healthcare", "LLY": "Healthcare", "BMY": "Healthcare",
    "AMGN": "Healthcare", "GILD": "Healthcare",
    "WMT": "Consumer", "PG": "Consumer", "KO": "Consumer", "PEP": "Consumer",
    "COST": "Consumer", "MCD": "Consumer", "NKE": "Consumer", "SBUX": "Consumer",
    "TGT": "Consumer", "HD": "Consumer",
    "XOM": "Energy", "CVX": "Energy", "CAT": "Industrial", "BA": "Industrial",
    "UPS": "Industrial", "HON": "Industrial", "GE": "Industrial", "LMT": "Industrial",
    "UNP": "Industrial", "RTX": "Industrial",
}