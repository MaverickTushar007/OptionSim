import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.fetcher import get_current_price
from report.summary import generate_report


def get_user_inputs():
    """Collect inputs from the user interactively."""

    print("\n")
    print("=" * 52)
    print("         OptionSim — Monte Carlo Pricer")
    print("       Built by Tushar Bhatt")
    print("=" * 52)

    # Ticker
    ticker = input("\n  Enter ticker (NVDA / AAPL / TSLA / MSFT): ").upper().strip()

    # Show live price
    print(f"\n  Fetching live price for {ticker}...")
    S = get_current_price(ticker)
    print(f"  Current Price : ${S}")

    # Strike
    strike_input = input(f"\n  Enter strike price (or press Enter for 5% OTM = ${S*1.05:.2f}): ").strip()
    K = float(strike_input) if strike_input else round(S * 1.05, 2)

    # Days to expiry
    days_input = input(f"\n  Enter days to expiry (or press Enter for 30): ").strip()
    days = int(days_input) if days_input else 30
    T = days / 252

    # Option type
    opt_input = input(f"\n  Option type — call or put (or press Enter for call): ").strip().lower()
    option_type = opt_input if opt_input in ["call", "put"] else "call"

    # Risk free rate
    r = 0.0525

    return ticker, K, T, r, option_type


if __name__ == "__main__":
    try:
        ticker, K, T, r, option_type = get_user_inputs()
        generate_report(ticker, K, T, r, option_type)

    except ValueError as e:
        print(f"\n   Invalid input: {e}")
    except Exception as e:
        print(f"\n   Error: {e}")