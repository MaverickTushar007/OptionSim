import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from data.fetcher import get_current_price
from engine.volatility import calculate_volatility
from engine.black_scholes import black_scholes
from engine.monte_carlo import monte_carlo_price
from engine.greeks import calculate_greeks

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="OptionSim Dashboard",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .main { background-color: #0e1117; }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
    }
    .metric-label {
        color: #8892b0;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #ccd6f6;
        font-size: 32px;
        font-weight: 700;
        line-height: 1;
    }
    .metric-delta-green {
        color: #64ffda;
        font-size: 13px;
        margin-top: 6px;
    }
    .metric-delta-red {
        color: #ff6b6b;
        font-size: 13px;
        margin-top: 6px;
    }
    .metric-delta-blue {
        color: #57cbff;
        font-size: 13px;
        margin-top: 6px;
    }

    /* Section Headers */
    .section-header {
        color: #ccd6f6;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin: 28px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3561;
    }

    /* Verdict Boxes */
    .verdict-green {
        background: linear-gradient(135deg, #0d2818, #0a3622);
        border: 1px solid #64ffda;
        border-radius: 10px;
        padding: 16px 20px;
        color: #64ffda;
        font-weight: 600;
        margin: 8px 0;
    }
    .verdict-yellow {
        background: linear-gradient(135deg, #2b2000, #332800);
        border: 1px solid #ffd700;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ffd700;
        font-weight: 600;
        margin: 8px 0;
    }
    .verdict-red {
        background: linear-gradient(135deg, #2b0000, #330a0a);
        border: 1px solid #ff6b6b;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ff6b6b;
        font-weight: 600;
        margin: 8px 0;
    }
    .verdict-blue {
        background: linear-gradient(135deg, #001a2b, #002233);
        border: 1px solid #57cbff;
        border-radius: 10px;
        padding: 16px 20px;
        color: #57cbff;
        font-weight: 600;
        margin: 8px 0;
    }

    /* Hero Banner */
    .hero {
        background: linear-gradient(135deg, #0a0e1a, #141929, #0d1b2a);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #64ffda, #57cbff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .hero-sub {
        color: #8892b0;
        font-size: 15px;
    }

    /* Tag */
    .tag {
        display: inline-block;
        background: #1a1f2e;
        border: 1px solid #2d3561;
        border-radius: 20px;
        padding: 4px 12px;
        color: #64ffda;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 12px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0a0e1a;
        border-right: 1px solid #2d3561;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Hero Banner ────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">📈 OptionSim Dashboard</div>
    <div class="hero-sub">Monte Carlo + Black-Scholes Pricing Engine · Real-Time Market Data · Greeks Analysis</div>
    <span class="tag">🐍 Python</span>
    <span class="tag">📊 Plotly</span>
    <span class="tag">🎲 Monte Carlo</span>
    <span class="tag">⚡ Live Data</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameters")
    st.markdown("---")

    ticker = st.selectbox(
        "Ticker",
        ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "GOOGL"],
        help="Select the underlying stock"
    )

    option_type = st.radio(
        "Option Type",
        ["call", "put"],
        horizontal=True
    )

    days = st.slider("Days to Expiry", 7, 90, 30)

    otm_pct = st.slider(
        "Strike (% OTM)",
        min_value=-10,
        max_value=20,
        value=5,
        step=1,
        help="Negative = ITM, Positive = OTM"
    )

    simulations = st.select_slider(
        "Simulations",
        options=[1000, 5000, 10000, 50000],
        value=10000
    )

    r = 0.0525
    T = days / 252

    st.markdown("---")
    run = st.button("🚀 Run Simulation", use_container_width=True)
    st.markdown(f"<div style='color:#8892b0; font-size:11px; text-align:center; margin-top:8px'>Built by Tushar Bhatt</div>", unsafe_allow_html=True)

# ── Helper: metric card ────────────────────────────────
def card(label, value, delta="", delta_type="green"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-delta-" + delta_type + "'>" + delta + "</div>" if delta else ""}
    </div>
    """, unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────
if run:
    with st.spinner("⚡ Fetching live data and running simulations..."):
        S     = get_current_price(ticker)
        sigma = calculate_volatility(ticker)
        K     = round(S * (1 + otm_pct / 100), 2)
        mc    = monte_carlo_price(S, K, T, r, sigma, option_type, simulations)
        bs    = black_scholes(S, K, T, r, sigma, option_type)
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)
        diff  = round(abs(mc - bs) / bs * 100, 2)

    # ── Section 1 — Market Data ────────────────────────
    st.markdown('<div class="section-header">📊 Market Snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Current Price", f"${S:,.2f}", f"{'Call' if option_type == 'call' else 'Put'} Option", "blue")
    with c2:
        card("Strike Price", f"${K:,.2f}", f"{otm_pct:+d}% {'OTM' if otm_pct > 0 else 'ITM'}", "blue")
    with c3:
        card("Annual Volatility", f"{sigma*100:.2f}%",
             "High Vol ⚠️" if sigma > 0.5 else "Normal Vol ✅", 
             "red" if sigma > 0.5 else "green")
    with c4:
        card("Time to Expiry", f"{days}d", f"T = {T:.4f} years", "blue")

    # ── Section 2 — Pricing ───────────────────────────
    st.markdown('<div class="section-header">💰 Pricing Engine</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        card("Monte Carlo Price", f"${mc:.4f}", f"{simulations:,} simulations", "green")
    with p2:
        card("Black-Scholes Price", f"${bs:.4f}", "Analytical benchmark", "blue")
    with p3:
        convergence = 100 - diff
        card("Model Convergence", f"{convergence:.1f}%",
             "✅ Models agree" if diff < 3 else "⚠️ Diverging",
             "green" if diff < 3 else "red")

    # ── Section 3 — Charts Row ────────────────────────
    st.markdown('<div class="section-header">📉 Analysis Charts</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    # Chart 1 — Price Paths
    with col_left:
        np.random.seed(42)
        n_paths = 200
        Z = np.random.standard_normal((n_paths, days))
        paths = np.zeros((n_paths, days + 1))
        paths[:, 0] = S
        for t in range(1, days + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (r - 0.5 * sigma**2) * (1/252) +
                sigma * np.sqrt(1/252) * Z[:, t-1]
            )

        fig_paths = go.Figure()
        for i in range(n_paths):
            fig_paths.add_trace(go.Scatter(
                y=paths[i], mode='lines',
                line=dict(width=0.6, color='rgba(100, 255, 218, 0.12)'),
                showlegend=False, hoverinfo='skip'
            ))

        # Mean path
        mean_path = paths.mean(axis=0)
        fig_paths.add_trace(go.Scatter(
            y=mean_path, mode='lines',
            line=dict(width=2.5, color='#64ffda'),
            name='Mean Path'
        ))

        fig_paths.add_hline(y=K, line_dash="dash", line_color="#ff6b6b",
                            annotation_text=f"Strike ${K}", annotation_font_color="#ff6b6b")
        fig_paths.add_hline(y=S, line_dash="dash", line_color="#57cbff",
                            annotation_text=f"Spot ${S}", annotation_font_color="#57cbff")

        fig_paths.update_layout(
            title=dict(text=f"📉 {ticker} Price Paths ({n_paths} of {simulations:,})", 
                      font=dict(color='#ccd6f6', size=14)),
            xaxis_title="Days", yaxis_title="Price ($)",
            height=380,
            paper_bgcolor='#0e1117', plot_bgcolor='#0a0e1a',
            font=dict(color='#8892b0'),
            xaxis=dict(gridcolor='#1a1f2e', showgrid=True),
            yaxis=dict(gridcolor='#1a1f2e', showgrid=True),
            legend=dict(font=dict(color='#ccd6f6'))
        )
        st.plotly_chart(fig_paths, use_container_width=True)

    # Chart 2 — Payoff Diagram
    with col_right:
        spot_range = np.linspace(S * 0.7, S * 1.3, 300)
        premium = mc

        if option_type == 'call':
            payoff    = np.maximum(spot_range - K, 0)
            intrinsic = np.maximum(spot_range - K, 0)
        else:
            payoff    = np.maximum(K - spot_range, 0)
            intrinsic = np.maximum(K - spot_range, 0)

        pnl = payoff - premium

        fig_payoff = go.Figure()

        # Profit zone fill
        fig_payoff.add_trace(go.Scatter(
            x=spot_range, y=np.maximum(pnl, 0),
            fill='tozeroy', fillcolor='rgba(100, 255, 218, 0.08)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))

        # Loss zone fill
        fig_payoff.add_trace(go.Scatter(
            x=spot_range, y=np.minimum(pnl, 0),
            fill='tozeroy', fillcolor='rgba(255, 107, 107, 0.08)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))

        # P&L line
        fig_payoff.add_trace(go.Scatter(
            x=spot_range, y=pnl,
            mode='lines', name='P&L at Expiry',
            line=dict(color='#64ffda', width=2.5),
            hovertemplate='Spot: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>'
        ))

        # Breakeven line
        fig_payoff.add_hline(y=0, line_color='#8892b0', line_width=1)

        # Current spot
        fig_payoff.add_vline(x=S, line_dash='dash', line_color='#57cbff',
                             annotation_text=f'Spot ${S}',
                             annotation_font_color='#57cbff')

        # Strike
        fig_payoff.add_vline(x=K, line_dash='dash', line_color='#ff6b6b',
                             annotation_text=f'Strike ${K}',
                             annotation_font_color='#ff6b6b')

        fig_payoff.update_layout(
            title=dict(text=f"💰 {option_type.title()} Option Payoff Diagram",
                      font=dict(color='#ccd6f6', size=14)),
            xaxis_title="Stock Price at Expiry ($)",
            yaxis_title="Profit / Loss ($)",
            height=380,
            paper_bgcolor='#0e1117', plot_bgcolor='#0a0e1a',
            font=dict(color='#8892b0'),
            xaxis=dict(gridcolor='#1a1f2e'),
            yaxis=dict(gridcolor='#1a1f2e'),
            legend=dict(font=dict(color='#ccd6f6'))
        )
        st.plotly_chart(fig_payoff, use_container_width=True)

    # Chart 3 — Volatility Smile
    st.markdown('<div class="section-header">😊 Volatility Smile</div>', unsafe_allow_html=True)

    strikes = np.linspace(S * 0.80, S * 1.20, 50)
    base_vol = sigma

    # Stronger smile curve
    moneyness = np.log(strikes / S)
    smile_vol = base_vol + 0.30 * moneyness**2 - 0.08 * moneyness

    # Current strike vol
    current_vol = base_vol + 0.30 * np.log(K/S)**2 - 0.08 * np.log(K/S)

    fig_smile = go.Figure()

    # Fill under smile
    fig_smile.add_trace(go.Scatter(
        x=strikes, y=smile_vol * 100,
        fill='tozeroy', fillcolor='rgba(167, 139, 250, 0.06)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))

    # Smile curve
    fig_smile.add_trace(go.Scatter(
        x=strikes, y=smile_vol * 100,
        mode='lines', name='Implied Vol',
        line=dict(color='#a78bfa', width=2.5),
        hovertemplate='Strike: $%{x:.2f}<br>IV: %{y:.2f}%<extra></extra>'
    ))

    # Current strike dot
    fig_smile.add_trace(go.Scatter(
        x=[K], y=[current_vol * 100],
        mode='markers', name='Your Strike',
        marker=dict(color='#64ffda', size=12,
                    line=dict(color='#0e1117', width=2))
    ))

    fig_smile.add_vline(x=S, line_dash='dash', line_color='#57cbff',
                        annotation_text='ATM',
                        annotation_font_color='#57cbff')

    y_min = float(smile_vol.min() * 100) - 2
    y_max = float(smile_vol.max() * 100) + 2

    fig_smile.update_layout(
        title=dict(text=f"😊 {ticker} Volatility Smile — Implied Vol by Strike",
                   font=dict(color='#ccd6f6', size=14)),
        xaxis_title="Strike Price ($)",
        yaxis_title="Implied Volatility (%)",
        yaxis=dict(range=[y_min, y_max], gridcolor='#1a1f2e'),
        height=350,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0a0e1a',
        font=dict(color='#8892b0'),
        xaxis=dict(gridcolor='#1a1f2e')
    )

    st.plotly_chart(fig_smile, use_container_width=True)

    # ── Section 4 — Greeks ────────────────────────────
    st.markdown('<div class="section-header">🎯 Greeks</div>', unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)

    with g1:
        card("Delta (Δ)", f"{greeks['delta']:.4f}",
             f"↑ ${greeks['delta']:.2f} gain per $1 stock move", "green")

    with g2:
        card("Theta (Θ)", f"{greeks['theta']:.4f}",
             f"↓ ${abs(greeks['theta']):.2f} lost per day", "red")

    with g3:
        card("Vega (ν)", f"{greeks['vega']:.4f}",
             f"↑ ${greeks['vega']:.2f} per 1% vol increase", "green")

    # ── Section 5 — Risk Verdict ──────────────────────
    st.markdown('<div class="section-header">⚠️ Risk Verdict</div>', unsafe_allow_html=True)

    hedge_shares = abs(round(greeks['delta'] * 100))

    v1, v2 = st.columns(2)

    with v1:
        if diff < 3:
            st.markdown(
                '<div class="verdict-green">✅ Models converge within 3% — Pricing is reliable</div>',
                unsafe_allow_html=True
            )
        elif diff < 7:
            st.markdown(
                f'<div class="verdict-yellow">⚠️ Models diverge {diff}% — Use with caution</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="verdict-red">❌ Models diverge {diff}% — Results unreliable</div>',
                unsafe_allow_html=True
            )

    with v2:
        st.markdown(
            f'<div class="verdict-blue">🛡️ Delta Hedge: Short <b>{hedge_shares} shares</b> per 100 contracts to be market neutral</div>',
            unsafe_allow_html=True
        )
else:
    # ── Empty State ───────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px; color: #8892b0;">
        <div style="font-size: 64px; margin-bottom: 16px;">📊</div>
        <div style="font-size: 22px; font-weight: 700; color: #ccd6f6; margin-bottom: 12px;">
            Ready to run simulation
        </div>
        <div style="font-size: 15px; max-width: 400px; margin: 0 auto; line-height: 1.6;">
            Select your parameters in the sidebar and hit 
            <span style="color: #64ffda; font-weight: 600;">Run Simulation</span> 
            to see live pricing, Greeks, payoff diagrams and more.
        </div>
    </div>
    """, unsafe_allow_html=True)