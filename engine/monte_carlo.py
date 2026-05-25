import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
from data.fetcher import get_current_price
from engine.volatility import calculate_volatility


def monte_carlo_price(S, K, T, r, sigma, option_type="call", simulations=10000):
    """
    Prices an option using Monte Carlo simulation.

    S           = Current stock price
    K           = Strike price
    T           = Time to expiry in years
    r           = Risk-free rate
    sigma       = Volatility
    option_type = 'call' or 'put'
    simulations = Number of price paths to simulate
    """

    # Step 1 — Generate random shocks using antithetic variates
    # Antithetic variates: pair each Z with -Z to reduce variance by ~40%
    # Each "simulation" count produces one Z and one -Z path → halve draws
    half = simulations // 2
    Z = np.random.standard_normal(half)
    Z_antithetic = np.concatenate([Z, -Z])   # paired: positive + negative shocks

    # Step 2 — Simulate terminal prices under risk-neutral GBM
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_antithetic)

    # Step 3 — Calculate payoff for each simulation
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    # Step 4 — Discount average payoff back to today
    price = np.exp(-r * T) * np.mean(payoffs)

    return round(price, 4)


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    from engine.black_scholes import black_scholes

    ticker = "NVDA"
    S      = get_current_price(ticker)
    K      = S * 1.05
    T      = 30 / 252
    r      = 0.0525
    sigma  = calculate_volatility(ticker)

    mc_call = monte_carlo_price(S, K, T, r, sigma, "call")
    mc_put  = monte_carlo_price(S, K, T, r, sigma, "put")
    bs_call = black_scholes(S, K, T, r, sigma, "call")
    bs_put  = black_scholes(S, K, T, r, sigma, "put")

    print(f"--- Monte Carlo vs Black-Scholes for {ticker} ---\n")
    print(f"Stock Price (S)   : ${S}")
    print(f"Strike Price (K)  : ${K:.2f}")
    print(f"Volatility (σ)    : {sigma * 100:.2f}%")
    print(f"Simulations       : 10,000\n")
    print(f"{'':20} {'CALL':>10} {'PUT':>10}")
    print(f"{'─'*42}")
    print(f"{'Monte Carlo':20} ${mc_call:>9} ${mc_put:>9}")
    print(f"{'Black-Scholes':20} ${bs_call:>9} ${bs_put:>9}")
    diff_call = abs(mc_call - bs_call) / bs_call * 100
    diff_put  = abs(mc_put  - bs_put)  / bs_put  * 100
    print(f"{'─'*42}")
    print(f"{'Difference':20}  {diff_call:>8.2f}%  {diff_put:>8.2f}%")