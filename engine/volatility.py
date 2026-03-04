import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from data.fetcher import get_stock_data


def calculate_volatility(ticker):
    """Calculate annualised historical volatility from 12 months of price data."""

    # Step 1 — Get the price data
    prices = get_stock_data(ticker)

    # Step 2 — Calculate daily log returns
    # Log return = ln(today's price / yesterday's price)
    log_returns = np.log(prices / prices.shift(1))

    # Step 3 — Drop the first NaN (no return for day 1)
    log_returns = log_returns.dropna()

    # Step 4 — Annualise it (252 trading days in a year)
    daily_volatility = log_returns.std()
    annual_volatility = daily_volatility * np.sqrt(252)

    return round(annual_volatility, 4)


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    tickers = ["NVDA", "AAPL", "TSLA"]

    print("--- Historical Volatility (Annualised) ---\n")
    for t in tickers:
        vol = calculate_volatility(t)
        print(f"{t}   :  {vol * 100:.2f}%")

"""
TSLA is 60% volatile annually — meaning the market expects huge swings. 
AAPL at 32% is much calmer. This σ directly feeds into our 
Monte Carlo — wilder stocks = wider simulation spread = more expensive options.
"""