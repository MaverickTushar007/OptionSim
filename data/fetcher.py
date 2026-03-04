import yfinance as yf

def get_stock_data(ticker):
    """Fetch last 12 months of closing prices for a given ticker."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    closing_prices = df["Close"]
    return closing_prices


def get_current_price(ticker):
    """Fetch the latest live price for a given ticker."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="1d")

    if df.empty:
        raise ValueError(f"Could not fetch current price for: {ticker}")

    current_price = df["Close"].iloc[-1]
    return round(current_price, 2)



if __name__ == "__main__":
    ticker = "NVDA"

    prices = get_stock_data(ticker)
    current = get_current_price(ticker)

    print("")
    print(f"Ticker        : {ticker}")
    print(f"Current Price : ${current}")
    print(f"Data Points   : {len(prices)} days")
    print(f"Date Range    : {prices.index[0].date()} → {prices.index[-1].date()}")

    # ── Check the actual price data ──
    print("\n--- First 5 days ---")
    print(prices.head())

    print("\n--- Last 5 days ---")
    print(prices.tail())

    print("\n--- Basic Stats ---")
    print(f"Highest Price : ${prices.max():.2f}")
    print(f"Lowest Price  : ${prices.min():.2f}")
    print(f"Average Price : ${prices.mean():.2f}")


"""
    fetcher.py
├── get_stock_data()    → downloaded 252 days of NVDA prices
└── get_current_price() → grabbed today's live price $183.92
"""