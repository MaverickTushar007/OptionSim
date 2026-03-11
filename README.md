# 📈 OptionSim Dashboard

> **An interactive Monte Carlo options pricing and risk analysis platform — built with Python, Streamlit, and Plotly.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 🧩 Problem Statement

Options are mispriced every day in the market — but retail investors and junior analysts have no accessible tool to independently verify fair value, understand their risk exposure, or stress-test their positions in real time.

Institutional desks have Bloomberg terminals and proprietary pricing tools costing millions. Everyone else has spreadsheets and gut feel.

**OptionSim Dashboard solves this.**

---

## 💡 Solution

A two-part system built from scratch:

| Part | Description |
|------|-------------|
| **OptionSim Engine** | Python-based Monte Carlo + Black-Scholes pricing engine |
| **OptionSim Dashboard** | Interactive Streamlit web dashboard wrapping the engine |

Together they give any user — retail trader, analyst, or student — a professional-grade options pricing and risk platform in their browser.

---

## ✨ Features

### 📊 Market Snapshot
- Live stock price fetched via Yahoo Finance
- Automatic strike price calculation (customizable % OTM/ITM)
- Historical annualized volatility (252-day)
- Time to expiry in days and years

### 💰 Pricing Engine
- **Monte Carlo simulation** — up to 50,000 simulated price paths using Geometric Brownian Motion
- **Black-Scholes pricing** — analytical benchmark
- **Model convergence** — measures how closely MC tracks BS (typically 97–99%)

### 📉 Simulated Price Paths
- 200 of 10,000 GBM paths plotted interactively
- Mean path highlighted
- Strike and spot price reference lines

### 💸 Payoff Diagram
- Profit/Loss at expiry for every possible stock price
- Profit zone (green) and loss zone (red) filled
- Breakeven point clearly visible

### 😊 Volatility Smile
- Implied volatility plotted across all strikes
- ATM reference line
- Your specific strike highlighted on the curve

### 🎯 Greeks
- **Delta (Δ)** — directional sensitivity per $1 stock move
- **Theta (Θ)** — daily time decay cost
- **Vega (ν)** — sensitivity to 1% volatility change

### ⚠️ Risk Verdict
- Model reliability verdict (converge / diverge)
- Delta hedge recommendation (shares to short per 100 contracts)

### 📊 VaR Backtest — Basel III
- 252 trading days of real returns vs rolling 95% VaR threshold
- Violations marked with ❌ on chart
- **Basel Traffic Light verdict** — Green / Yellow / Red zone
- Violation rate vs expected 5%

---

## 🏦 Real World Use Cases

### Retail Trader
> Ravi wants to buy NVDA calls before earnings. He sets 14 days to expiry, 5% OTM. The dashboard shows fair value $4.20, Delta 0.31 — he needs a $13 move to break even, and Theta is costing $0.18 per day. He decides to pass. The model saved him money.

### Junior Analyst
> A hedge fund analyst needs to sanity-check an options price before a trade goes through. OptionSim returns MC price, BS price, and 99% convergence in under 3 seconds — confirming the desk's model is correct.

### Risk Manager
> A risk manager checks the VaR backtest and sees 14 violations in 192 days — RED zone (7.3% vs expected 5%). She flags the 60-day rolling window as too slow for NVDA's current volatility regime and switches to EWMA. That's exactly what Basel III compliance requires.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| NumPy | Monte Carlo simulation, matrix operations |
| SciPy | Black-Scholes normal distribution |
| yfinance | Live market data + historical prices |
| Streamlit | Web dashboard framework |
| Plotly | Interactive charts |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/MaverickTushar007/OptionSim-Dashboard.git
cd OptionSim-Dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda install -c conda-forge streamlit plotly yfinance numpy scipy
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`

---

## 📁 Project Structure

```
OptionSim-Dashboard/
│
├── app.py                  ← Streamlit dashboard (main file)
├── requirements.txt        ← Dependencies
├── README.md
│
├── data/
│   ├── __init__.py
│   └── fetcher.py          ← Live price + historical data via yfinance
│
└── engine/
    ├── __init__.py
    ├── volatility.py       ← Annualized historical volatility
    ├── black_scholes.py    ← Analytical BS pricing
    ├── monte_carlo.py      ← GBM Monte Carlo simulation
    └── greeks.py           ← Delta, Theta, Vega (numerical bumping)
```

---

## 📐 The Math

### Geometric Brownian Motion (Monte Carlo)
```
S_T = S_0 × exp((r - 0.5σ²)T + σ√T × Z)

Where:
  S_0 = Current stock price
  r   = Risk-free rate (5.25%)
  σ   = Annualized volatility
  T   = Time to expiry (years)
  Z   ~ N(0,1) random shock
```

### Black-Scholes (Benchmark)
```
Call = S·N(d1) - K·e^(-rT)·N(d2)
Put  = K·e^(-rT)·N(-d2) - S·N(-d1)

d1 = [ln(S/K) + (r + 0.5σ²)T] / (σ√T)
d2 = d1 - σ√T
```

### Value at Risk (95%, 1-day)
```
VaR = 5th percentile of rolling 60-day return window
Violation = actual return < VaR threshold
Basel Red Zone = >10 violations in 252 days
```

---

## 📊 Sample Output (NVDA)

```
Current Price:     $183.04
Strike (5% OTM):   $192.19
Volatility:         42.30%

Monte Carlo Price:  $7.4312
Black-Scholes:      $7.3609
Convergence:        99.0% ✅

Delta:   0.4135  → gains $0.41 per $1 move
Theta:  -0.1809  → loses $0.19 per day
Vega:    0.2372  → gains $0.24 per 1% vol change

VaR Backtest: 14 violations / 192 days
Violation Rate: 7.3% (expected 5.0%)
Basel Zone: 🔴 RED
```

---

## 🔗 Related Project

**[OptionSim](https://github.com/MaverickTushar007/OptionSim)** — The terminal-based pricing engine this dashboard is built on top of.

---

## 👤 Author

**Tushar Bhatt**
- GitHub: [@MaverickTushar007](https://github.com/MaverickTushar007)

---

## 📄 License

MIT License — free to use, modify and distribute.
