import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.fetcher import get_current_price
from engine.volatility import calculate_volatility
from engine.black_scholes import black_scholes
from engine.monte_carlo import monte_carlo_price
from engine.greeks import calculate_greeks


def generate_report(ticker, K, T, r, option_type="call"):
    """Generates a full options pricing and risk report."""

    # ── Gather all data ───────────────────────────────
    S     = get_current_price(ticker)
    sigma = calculate_volatility(ticker)
    bs    = black_scholes(S, K, T, r, sigma, option_type)
    mc    = monte_carlo_price(S, K, T, r, sigma, option_type)
    greeks = calculate_greeks(S, K, T, r, sigma, option_type)

    # ── Verdict ───────────────────────────────────────
    diff       = mc - bs
    diff_pct   = (diff / bs) * 100
    if abs(diff_pct) < 3:
        convergence = " STRONG  — Models converge within 3%"
    elif abs(diff_pct) < 7:
        convergence = "  MODERATE — Minor divergence detected"
    else:
        convergence = " WEAK    — Models diverge significantly"

    # ── Hedge suggestion ──────────────────────────────
    hedge_shares = abs(round(greeks["delta"] * 100))

    # ── Print report ──────────────────────────────────
    print("\n")
    print("=" * 52)
    print("        OptionSim — Risk & Pricing Report")
    print("=" * 52)

    print(f"\n  Ticker        : {ticker}")
    print(f"  Option Type   : {option_type.upper()}")
    print(f"  Stock Price   : ${S}")
    print(f"  Strike Price  : ${K:.2f}")
    print(f"  Days to Exp.  : {round(T * 252)} days")
    print(f"  Volatility    : {sigma * 100:.2f}%")
    print(f"  Risk-Free Rate: {r * 100:.2f}%")

    print(f"\n{'─' * 52}")
    print(f"  PRICING")
    print(f"{'─' * 52}")
    print(f"  Monte Carlo Price  : ${mc}")
    print(f"  Black-Scholes Price: ${bs}")
    print(f"  Convergence        : {convergence}")

    print(f"\n{'─' * 52}")
    print(f"  GREEKS")
    print(f"{'─' * 52}")
    print(f"  Delta (Δ) : {greeks['delta']:>7}  → ${greeks['delta']:.2f} per $1 stock move")
    print(f"  Theta (Θ) : {greeks['theta']:>7}  → -${abs(greeks['theta']):.2f} lost per day")
    print(f"  Vega  (ν) : {greeks['vega']:>7}  → ${greeks['vega']:.2f} per 1% vol change")

    print(f"\n{'─' * 52}")
    print(f"  RISK & HEDGE")
    print(f"{'─' * 52}")
    print(f"  Delta Hedge : Short {hedge_shares} shares per 100 contracts")
    print(f"  Time Risk   : Losing ${abs(greeks['theta']):.2f}/day from decay")
    print(f"  Vol Risk    : {'Beneficial' if option_type == 'call' else 'Adverse'} — Vega {greeks['vega']}")

    print(f"\n{'=' * 52}\n")


# ── Quick test ────────────────────────────────────────
if __name__ == "__main__":
    ticker      = "NVDA"
    S           = get_current_price(ticker)
    K           = S * 1.05
    T           = 30 / 252
    r           = 0.0525
    option_type = "call"

    generate_report(ticker, K, T, r, option_type)