import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from engine.black_scholes import black_scholes
from data.fetcher import get_current_price
from engine.volatility import calculate_volatility


def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Delta, Theta, Vega using numerical bumping.
    """

    # ── Delta ─────────────────────────────────────────
    # Bump stock price up and down by $1, see price change
    bump_S  = 1.0
    price_up   = black_scholes(S + bump_S, K, T, r, sigma, option_type)
    price_down = black_scholes(S - bump_S, K, T, r, sigma, option_type)
    delta = (price_up - price_down) / (2 * bump_S)

    # ── Theta ─────────────────────────────────────────
    # Bump time forward by 1 day, see price change
    bump_T  = 1 / 252
    price_today    = black_scholes(S, K, T, r, sigma, option_type)
    price_tomorrow = black_scholes(S, K, T - bump_T, r, sigma, option_type)
    theta = price_tomorrow - price_today   # always negative for long options

    # ── Vega ──────────────────────────────────────────
    # Bump volatility up by 1%, see price change
    bump_v   = 0.01
    price_vol_up   = black_scholes(S, K, T, r, sigma + bump_v, option_type)
    price_vol_down = black_scholes(S, K, T, r, sigma - bump_v, option_type)
    vega = (price_vol_up - price_vol_down) / 2

    return {
        "delta" : round(delta, 4),
        "theta" : round(theta, 4),
        "vega"  : round(vega,  4)
    }


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    ticker = "NVDA"
    S      = get_current_price(ticker)
    K      = S * 1.05
    T      = 30 / 252
    r      = 0.0525
    sigma  = calculate_volatility(ticker)

    greeks = calculate_greeks(S, K, T, r, sigma, "call")

    print(f"--- Greeks for {ticker} CALL Option ---\n")
    print(f"Stock Price (S)  : ${S}")
    print(f"Strike Price (K) : ${K:.2f}")
    print(f"Volatility (σ)   : {sigma * 100:.2f}%\n")
    print(f"Delta  (Δ)  : {greeks['delta']:>8}  ← option gains ${greeks['delta']:.2f} per $1 stock move")
    print(f"Theta  (Θ)  : {greeks['theta']:>8}  ← option loses ${abs(greeks['theta']):.2f} per day")
    print(f"Vega   (ν)  : {greeks['vega']:>8}  ← option gains ${greeks['vega']:.2f} per 1% vol increase")


"""
"Delta of 0.41 tells me I need to short 41 shares of NVDA per 100 contracts to be delta-neutral — that's my hedge."
"""