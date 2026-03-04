"""
Black-Scholes says — "Given the stock price, strike, time, rate and volatility — 
here's the mathematically fair price of the option." It's a formula. One shot, 
one answer.
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import norm
from data.fetcher import get_current_price
from engine.volatility import calculate_volatility


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    S     = Current stock price
    K     = Strike price
    T     = Time to expiry in years
    r     = Risk-free rate (e.g. 0.05 for 5%)
    sigma = Volatility (e.g. 0.42 for 42%)
    """

    # Step 1 — Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Step 2 — Price the option
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return round(price, 4)


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    ticker = "NVDA"

    S     = get_current_price(ticker)
    K     = S * 1.05          # Strike = 5% above current price
    T     = 30 / 252          # 30 trading days to expiry
    r     = 0.0525            # 5.25% risk-free rate
    sigma = calculate_volatility(ticker)

    call_price = black_scholes(S, K, T, r, sigma, "call")
    put_price  = black_scholes(S, K, T, r, sigma, "put")

    print(f"--- Black-Scholes Pricing for {ticker} ---\n")
    print(f"Stock Price (S)   : ${S}")
    print(f"Strike Price (K)  : ${K:.2f}")
    print(f"Time to Expiry    : 30 days")
    print(f"Volatility        : {sigma * 100:.2f}%")
    print(f"Risk-Free Rate    : 5.25%")
    print(f"\nCall Price      : ${call_price}")
    print(f"Put Price         : ${put_price}")