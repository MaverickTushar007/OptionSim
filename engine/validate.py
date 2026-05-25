"""
validate.py — OptionSim Pricing Engine Validation Suite

Runs three measurements that back up resume claims:
  1. MAPE grid  — MC vs BS across 50 strike/maturity combos  → "<0.5% MAPE"
  2. Convergence — MC std-dev vs 1/√N theory                → "O(1/√N) rate"
  3. Latency     — full Greeks surface timing                → "Xms latency"
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import time
import numpy as np
from engine.black_scholes import black_scholes
from engine.monte_carlo   import monte_carlo_price
from engine.greeks        import calculate_greeks


# ── Fixed market params ───────────────────────────────────────────────────────
S     = 100.0    # spot — normalised so results are clean
r     = 0.0525
sigma = 0.30


# ── 1. MAPE GRID — 50 strike/maturity combinations ───────────────────────────
def run_mape_grid(n_trials=5):
    """
    Grid: 5 maturities × 10 strikes = 50 combos.
    Average each combo over n_trials MC runs (10k sims each) to smooth noise.
    Prints MAPE and worst-case APE.
    """
    maturities = [7/252, 14/252, 30/252, 60/252, 90/252]   # 1w, 2w, 1m, 2m, 3m
    otm_pcts   = [-0.10, -0.05, -0.02, 0.0, 0.02,
                   0.05,  0.08,  0.10, 0.15, 0.20]         # -10% to +20% OTM

    errors = []
    results = []

    for T in maturities:
        for pct in otm_pcts:
            K  = S * (1 + pct)
            bs = black_scholes(S, K, T, r, sigma, "call")

            # average multiple MC runs to reduce sampling noise
            mc_runs = [monte_carlo_price(S, K, T, r, sigma, "call", 10000)
                       for _ in range(n_trials)]
            mc = np.mean(mc_runs)

            # skip deep OTM near-zero prices (APE blows up when BS ≈ 0)
            if bs < 0.01:
                continue

            ape = abs(mc - bs) / bs * 100
            errors.append(ape)
            results.append({
                "T_days": round(T * 252),
                "K":      round(K, 2),
                "BS":     bs,
                "MC":     round(mc, 4),
                "APE%":   round(ape, 3)
            })

    mape   = np.mean(errors)
    worst  = np.max(errors)
    median = np.median(errors)

    print("\n" + "="*60)
    print("  MAPE GRID — MC vs Black-Scholes (50 combos, 5 trials each)")
    print("="*60)
    print(f"  Combos tested  : {len(errors)}")
    print(f"  MAPE           : {mape:.4f}%")
    print(f"  Median APE     : {median:.4f}%")
    print(f"  Worst APE      : {worst:.4f}%")
    print(f"\n  {'Days':>6} {'Strike':>8} {'BS':>8} {'MC':>8} {'APE%':>8}")
    print(f"  {'─'*46}")
    for r_ in results[:10]:   # show first 10 rows
        print(f"  {r_['T_days']:>6} {r_['K']:>8.2f} {r_['BS']:>8.4f} "
              f"{r_['MC']:>8.4f} {r_['APE%']:>7.3f}%")
    print(f"  ... ({len(results)-10} more rows)")

    return mape, worst


# ── 2. CONVERGENCE — MC std-dev vs 1/√N theory ───────────────────────────────
def run_convergence_test():
    """
    At each N, run 50 MC trials and measure std-dev of the price estimate.
    Theory: std-dev ∝ 1/√N. We fit and verify the slope.
    """
    T = 30 / 252
    K = S * 1.05

    sim_counts  = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    n_trials    = 50
    std_devs    = []

    print("\n" + "="*60)
    print("  CONVERGENCE — MC Std-Dev vs 1/√N")
    print("="*60)
    print(f"  {'N sims':>8} {'Std-Dev':>10} {'1/√N scaled':>14} {'Ratio':>8}")
    print(f"  {'─'*44}")

    bs_price = black_scholes(S, K, T, r, sigma, "call")
    ref_std  = None

    for n in sim_counts:
        prices  = [monte_carlo_price(S, K, T, r, sigma, "call", n)
                   for _ in range(n_trials)]
        std     = np.std(prices)
        std_devs.append(std)

        if ref_std is None:
            ref_std  = std
            ref_n    = n

        theoretical = ref_std * np.sqrt(ref_n / n)
        ratio        = std / theoretical if theoretical > 0 else 0
        print(f"  {n:>8,} {std:>10.5f} {theoretical:>14.5f} {ratio:>8.3f}")

    # fit log-log slope — should be close to -0.5 for O(1/√N)
    log_n   = np.log(sim_counts)
    log_std = np.log(std_devs)
    slope, _ = np.polyfit(log_n, log_std, 1)
    print(f"\n  Log-log slope : {slope:.4f}  (theory = -0.500)")
    print(f"  BS benchmark  : {bs_price:.4f}")

    return slope


# ── 3. LATENCY — full Greeks surface timing ───────────────────────────────────
def run_latency_test(n_runs=200):
    """
    Time calculate_greeks() over n_runs calls.
    Reports mean, p50, p95, p99 in milliseconds.
    """
    T = 30 / 252
    K = S * 1.05

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        calculate_greeks(S, K, T, r, sigma, "call")
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)   # convert to ms

    times = np.array(times)

    print("\n" + "="*60)
    print(f"  LATENCY — calculate_greeks() over {n_runs} runs")
    print("="*60)
    print(f"  Mean   : {times.mean():.3f} ms")
    print(f"  Median : {np.median(times):.3f} ms")
    print(f"  p95    : {np.percentile(times, 95):.3f} ms")
    print(f"  p99    : {np.percentile(times, 99):.3f} ms")
    print(f"  Min    : {times.min():.3f} ms")
    print(f"  Max    : {times.max():.3f} ms")

    return times.mean(), np.percentile(times, 95)


# ── 4. ANTITHETIC VARIATES — variance reduction measurement ──────────────────
def run_variance_reduction_test(n_trials=100):
    """
    Compare plain MC vs antithetic MC std-dev at 10k sims.
    Measures actual variance reduction percentage.
    Note: current monte_carlo.py uses antithetic variates.
    Plain MC is re-implemented here inline for comparison.
    """
    T = 30 / 252
    K = S * 1.05
    n = 10000

    # Plain MC (no antithetic)
    plain_prices = []
    for _ in range(n_trials):
        Z  = np.random.standard_normal(n)
        ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        payoffs = np.maximum(ST - K, 0)
        plain_prices.append(np.exp(-r*T) * np.mean(payoffs))

    plain_std = np.std(plain_prices)

    # Antithetic MC (pairs Z and -Z)
    anti_prices = []
    for _ in range(n_trials):
        Z  = np.random.standard_normal(n // 2)
        Z2 = np.concatenate([Z, -Z])
        ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z2)
        payoffs = np.maximum(ST - K, 0)
        anti_prices.append(np.exp(-r*T) * np.mean(payoffs))

    anti_std = np.std(anti_prices)

    reduction = (1 - anti_std / plain_std) * 100

    print("\n" + "="*60)
    print("  ANTITHETIC VARIATES — Variance Reduction")
    print("="*60)
    print(f"  Plain MC std-dev      : {plain_std:.6f}")
    print(f"  Antithetic MC std-dev : {anti_std:.6f}")
    print(f"  Variance reduction    : {reduction:.1f}%")

    return reduction


# ── Runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nOptionSim — Validation Suite")
    print("Running all 4 tests... (takes ~60s)\n")

    mape, worst       = run_mape_grid()
    slope             = run_convergence_test()
    mean_ms, p95_ms   = run_latency_test()
    var_reduction     = run_variance_reduction_test()

    print("\n" + "="*60)
    print("  RESUME BULLET VERIFICATION")
    print("="*60)
    print(f"  MAPE vs BS         : {mape:.3f}%  "
          f"{'✅ <0.5%' if mape < 0.5 else '❌ >0.5% — update bullet'}")
    print(f"  Convergence slope  : {slope:.3f}  "
          f"{'✅ ~-0.5 (O(1/√N))' if abs(slope + 0.5) < 0.05 else '⚠️ check'}")
    print(f"  Greeks latency     : {mean_ms:.2f}ms mean, {p95_ms:.2f}ms p95")
    print(f"  Variance reduction : {var_reduction:.1f}%  "
          f"{'✅ ~40%' if 30 < var_reduction < 55 else '⚠️ actual number above'}")
    print()
