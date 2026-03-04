# OptionSim — Monte Carlo Options Pricing Engine

A Python-based options pricing engine that uses Monte Carlo simulation 
to price equity options and compute risk metrics (Greeks).

## What it does
- Fetches live stock data for any ticker (NVDA, AAPL, TSLA, MSFT, etc.)
- Calculates historical volatility from 12 months of real market data
- Prices Call and Put options using Monte Carlo simulation (10,000 paths)
- Validates results against Black-Scholes analytical model
- Computes Delta, Theta, Vega with plain-English interpretation
- Generates a structured Risk & Hedge recommendation report

## Tech Stack
- Python 3.x
- NumPy — simulation and math
- SciPy — normal distribution (Black-Scholes)
- yfinance — live market data

## Project Structure
```
OptionSim/
├── data/
│   └── fetcher.py        # Live stock data via yfinance
├── engine/
│   ├── volatility.py     # Historical volatility (σ)
│   ├── black_scholes.py  # BS analytical pricer
│   ├── monte_carlo.py    # MC simulation engine
│   └── greeks.py         # Delta, Theta, Vega
├── report/
│   └── summary.py        # Risk report generator
└── main.py               # Entry point
```

## How to Run

### Install dependencies
pip install numpy yfinance scipy

### Run the pricer
python main.py

## Sample Output
```
OptionSim — Risk & Pricing Report
════════════════════════════════════════════

Ticker        : MSFT
Option Type   : CALL
Stock Price   : $408.31
Strike Price  : $440.00
Volatility    : 26.65%

PRICING
────────────────────────────────────────────
Monte Carlo Price   : $5.0327
Black-Scholes Price : $5.1378
Convergence         : STRONG — Models converge within 3%

GREEKS
────────────────────────────────────────────
Delta (Δ) : 0.2423  → $0.24 per $1 stock move
Theta (Θ) :-0.2155  → -$0.22 lost per day
Vega  (ν) : 0.4400  → $0.44 per 1% vol change

RISK & HEDGE
────────────────────────────────────────────
Delta Hedge : Short 24 shares per 100 contracts
Time Risk   : Losing $0.22/day from decay
```

## Key Concepts
- **Monte Carlo Simulation** — Models 10,000 possible future price paths 
  using Geometric Brownian Motion
- **Black-Scholes** — Analytical benchmark to validate MC results
- **Greeks** — Risk sensitivities calculated via numerical bumping
- **Delta Hedging** — Number of shares needed to neutralise directional risk

## Author
Tushar Bhatt — Built as a quantitative finance project