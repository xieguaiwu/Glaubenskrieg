#!/usr/bin/env python3
"""
Quantitative Deep Dive: Monte Carlo Survival Simulation
for LSTM Walk-Forward Strategy

Based on:
- LSTM IC: mean=0.0751, std=0.0444 (window-to-window), CV=0.59
- Ridge IC: 0.065 (walk-forward), held-out 0.044
- sigma_5d = 4.47%, top-20% z-score = 0.70
- Transaction costs: 30bp/period, 50 periods/year
- 402 stocks, S&P 500 universe
"""

import numpy as np
import json
from scipy import stats
from dataclasses import dataclass

# ============================================================================
# Section 1: Grinold-Kahn Expected Return Modeling
# ============================================================================

@dataclass
class StrategyParams:
    ic_mean: float = 0.0751       # LSTM mean rank IC
    ic_std_window: float = 0.0444  # IC std across 11 walk-forward windows
    ic_cv: float = 0.59            # Coefficient of variation
    sigma_5d: float = 0.0447       # Cross-sectional 5-day return dispersion
    z_score: float = 0.70          # Effective z-score for top-20% long-only
    n_stocks: int = 402
    n_periods_per_year: int = 50   # 5-day rebalancing
    cost_per_period: float = 0.0030  # 30bp transaction cost per period
    risk_free_rate: float = 0.03   # Annual risk-free rate
    pairwise_corr: float = 0.30    # Average pairwise correlation
    top_frac: float = 0.20         # Top quintile selection

params = StrategyParams()

# 1.1 Expected 5-day gross return per period
# E[r_5d] = IC * sigma_5d * z_score
expected_5d_gross = params.ic_mean * params.sigma_5d * params.z_score
expected_annual_gross = expected_5d_gross * params.n_periods_per_year

# 1.2 Transaction costs
# With ~100% turnover per period (typical for cross-sectional ranking)
annual_cost = params.cost_per_period * params.n_periods_per_year

# 1.3 Net expected return
expected_annual_net = expected_annual_gross - annual_cost

# 1.4 Portfolio volatility estimation
# Individual stock 5-day vol ≈ sigma_5d
# Portfolio vol: sigma_p = sigma_i * sqrt(1/N + (N-1)/N * rho)
n_selected = int(params.n_stocks * params.top_frac)
sigma_portfolio_5d = params.sigma_5d * np.sqrt(
    1/n_selected + (n_selected - 1)/n_selected * params.pairwise_corr
)
sigma_portfolio_annual = sigma_portfolio_5d * np.sqrt(params.n_periods_per_year)

# 1.5 Information Ratio
# IR = IC * sqrt(breadth)
# Breadth ≈ n_periods * effective_n_independent_bets
# Effective independent bets ≈ n_selected * (1 - avg_pairwise_corr)
effective_breadth = params.n_periods_per_year * n_selected * (1 - params.pairwise_corr)
IR_gross = params.ic_mean * np.sqrt(effective_breadth)

# 1.6 Sharpe Ratio
sharpe_gross = (expected_annual_gross - params.risk_free_rate) / sigma_portfolio_annual
sharpe_net = (expected_annual_net - params.risk_free_rate) / sigma_portfolio_annual

# 1.7 Breakeven IC
ic_breakeven_full_cost = (annual_cost + params.risk_free_rate) / (
    params.sigma_5d * params.z_score * params.n_periods_per_year
)

# 1.8 Sensitivity analysis: IC values and corresponding returns
ic_scenarios = [0.03, 0.044, 0.05, 0.057, 0.065, 0.0751, 0.097, 0.10]
sensitivity = []
for ic in ic_scenarios:
    gross = ic * params.sigma_5d * params.z_score * params.n_periods_per_year
    net = gross - annual_cost
    # Monthly on 10k/50k capital
    monthly_net_10k = net / 12 * 10_000  # CNY
    monthly_net_50k = net / 12 * 50_000  # CNY (assumes CNY-denominated)
    sensitivity.append({
        "ic": ic,
        "gross_annual_pct": round(gross * 100, 2),
        "net_annual_pct": round(net * 100, 2),
        "monthly_expected_10k_cny": round(monthly_net_10k, 1),
        "monthly_expected_50k_cny": round(monthly_net_50k, 1),
    })

print("=" * 70)
print("SECTION 1: GRINOLD-KAHN EXPECTED RETURN MODEL")
print("=" * 70)
print(f"\nParameters:")
print(f"  IC = {params.ic_mean:.4f}, sigma_5d = {params.sigma_5d:.4f}")
print(f"  z-score (top 20%) = {params.z_score:.2f}")
print(f"  Stocks selected = {n_selected}")
print(f"  Annual periods = {params.n_periods_per_year}")
print(f"  Cost per period = {params.cost_per_period*100:.0f}bp")
print(f"  Pairwise correlation = {params.pairwise_corr}")

print(f"\nExpected 5-day gross return: {expected_5d_gross*100:.3f}%")
print(f"Expected annual gross return: {expected_annual_gross*100:.2f}%")
print(f"Annual transaction costs: {annual_cost*100:.2f}%")
print(f"Expected annual net return: {expected_annual_net*100:.2f}%")

print(f"\nPortfolio risk:")
print(f"  5-day portfolio sigma: {sigma_portfolio_5d*100:.2f}%")
print(f"  Annual portfolio sigma: {sigma_portfolio_annual*100:.2f}%")

print(f"\nRisk-adjusted metrics:")
print(f"  Gross Sharpe: {sharpe_gross:.3f}")
print(f"  Net Sharpe: {sharpe_net:.3f}")
print(f"  Information Ratio (gross): {IR_gross:.3f}")
print(f"  Effective breadth: {effective_breadth:.0f}")

print(f"\nBreakeven IC (with 30bp cost + 3% rf): {ic_breakeven_full_cost:.4f}")

print(f"\nSensitivity Analysis (IC → Annual Return):")
print(f"{'IC':>8s}  {'Gross%':>8s}  {'Net%':>8s}  {'Mo.10k':>8s}  {'Mo.50k':>8s}")
for s in sensitivity:
    print(f"{s['ic']:8.4f}  {s['gross_annual_pct']:8.2f}  {s['net_annual_pct']:8.2f}  "
          f"{s['monthly_expected_10k_cny']:8.1f}  {s['monthly_expected_50k_cny']:8.1f}")

# ============================================================================
# Section 2: LSTM Signal Temporal Analysis & Degradation
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 2: LSTM SIGNAL TEMPORAL CHARACTERISTICS & DEGRADATION")
print("=" * 70)

per_window_ic = np.array([0.108, 0.116, 0.171, 0.067, 0.064,
                          0.013, 0.023, 0.107, 0.066, 0.059, 0.033])
windows = np.arange(11)

# 2.1 Linear trend regression
from scipy import stats as sp_stats
slope, intercept, r_value, p_value, std_err = sp_stats.linregress(windows, per_window_ic)
decay_per_window = slope

# 2.2 Rolling statistics
last3 = per_window_ic[-3:]
last5 = per_window_ic[-5:]
first3 = per_window_ic[:3]

# 2.3 IC volatility impact on trading
# What does IC std = 0.0444 mean for period-by-period returns?
# For a single period, IC varies; the expected return varies accordingly
# P(IC < ic_breakeven) — probability of negative-return regime
p_below_breakeven = stats.norm.cdf(ic_breakeven_full_cost,
                                    loc=params.ic_mean,
                                    scale=params.ic_std_window)

# 2.4 Expected frequency of "bad months" (IC < 0.02)
p_ic_neg = stats.norm.cdf(0, loc=params.ic_mean, scale=params.ic_std_window)
p_ic_below_002 = stats.norm.cdf(0.02, loc=params.ic_mean, scale=params.ic_std_window)

# 2.5 Conditional expected return in "good" vs "bad" IC regimes
# Truncated normal expectations
def truncated_mean(mu, sigma, a, b):
    """Mean of truncated normal N(mu,sigma) in [a, b]"""
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    Z = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
    return mu + sigma * (stats.norm.pdf(alpha) - stats.norm.pdf(beta)) / Z

ic_good_mean = truncated_mean(params.ic_mean, params.ic_std_window,
                                ic_breakeven_full_cost, np.inf)
ic_bad_mean = truncated_mean(params.ic_mean, params.ic_std_window,
                              -np.inf, ic_breakeven_full_cost)

# 2.6 Markov regime-switching perspective
# Transition: estimate probability of moving from high-IC to low-IC
# Using the 11-window sequence
high_low_threshold = 0.05
regimes = (per_window_ic > high_low_threshold).astype(int)
transitions = sum(1 for i in range(len(regimes)-1) if regimes[i] != regimes[i+1])
n_high = sum(regimes)
n_low = len(regimes) - n_high

if n_high > 0:
    high_to_low = sum(1 for i in range(len(regimes)-1)
                       if regimes[i]==1 and regimes[i+1]==0)
    p_high_to_low = high_to_low / n_high if n_high > 0 else 0
else:
    p_high_to_low = 0

if n_low > 0:
    low_to_high = sum(1 for i in range(len(regimes)-1)
                       if regimes[i]==0 and regimes[i+1]==1)
    p_low_to_high = low_to_high / n_low if n_low > 0 else 0
else:
    p_low_to_high = 0

print(f"\nWindow-by-window IC:")
for i, ic in enumerate(per_window_ic):
    marker = " ← LAST 3" if i >= 8 else " ← FIRST 3" if i < 3 else ""
    print(f"  W{i:2d}: {ic:.4f}{marker}")

print(f"\nTemporal statistics:")
print(f"  Mean IC: {per_window_ic.mean():.4f}")
print(f"  Std IC: {per_window_ic.std():.4f}")
print(f"  Linear decay slope: {decay_per_window:.4f}/window")
print(f"  R-squared of trend: {r_value**2:.4f}")
print(f"  Trend p-value: {p_value:.4f}")
print(f"  First 3 windows mean: {first3.mean():.4f}")
print(f"  Last 3 windows mean: {last3.mean():.4f}")
print(f"  Last 5 windows mean: {last5.mean():.4f}")
print(f"  Decay from first3 to last3: {(last3.mean() - first3.mean()):.4f}")

print(f"\nIC distribution implications for trading:")
print(f"  P(IC < 0): {p_ic_neg*100:.1f}% — probability of negative-signal period")
print(f"  P(IC < 0.02): {p_ic_below_002*100:.1f}% — prob of near-zero signal")
print(f"  P(IC < breakeven={ic_breakeven_full_cost:.4f}): {p_below_breakeven*100:.1f}%")
print(f"  E[IC | IC > breakeven]: {ic_good_mean:.4f} (good regime)")
print(f"  E[IC | IC < breakeven]: {ic_bad_mean:.4f} (bad regime)")

print(f"\nRegime transition (threshold IC > {high_low_threshold}):")
print(f"  High-IC windows: {n_high}, Low-IC windows: {n_low}")
print(f"  Transitions: {transitions}")
print(f"  P(High→Low): {p_high_to_low:.2f}")
print(f"  P(Low→High): {p_low_to_high:.2f}")

# ============================================================================
# Section 3: Kelly Optimal Position Sizing & Risk Budget
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 3: KELLY OPTIMAL POSITION SIZING & RISK BUDGET")
print("=" * 70)

# 3.1 Full Kelly for long-only strategy
# f* = (mu - rf) / sigma^2  for the excess return above risk-free
mu_excess = expected_annual_gross - params.risk_free_rate  # gross excess return
variance = sigma_portfolio_annual ** 2
f_kelly_full = mu_excess / variance

# But with transaction costs:
mu_excess_net = expected_annual_net - params.risk_free_rate
f_kelly_net = mu_excess_net / variance

# 3.2 Fractional Kelly (more practical)
f_half_kelly = f_kelly_full * 0.5
f_quarter_kelly = f_kelly_full * 0.25

# 3.3 Kelly with parameter uncertainty
# Adjust Kelly downward to account for estimation error in IC
# Using the "uncertainty-adjusted Kelly": f_adj = (mu - rf - lambda*sigma_mu^2) / sigma^2
ic_se = params.ic_std_window / np.sqrt(11)  # standard error of IC estimate
mu_se = ic_se * params.sigma_5d * params.z_score * params.n_periods_per_year
lambda_param = 2  # risk aversion adjustment
f_kelly_robust = (mu_excess - lambda_param * mu_se**2 / variance) / variance

# 3.4 Risk-budget approach: Value at Risk
# For different capital levels
for capital in [10_000, 50_000, 2000]:  # CNY, USD
    var_95_1m = sigma_portfolio_annual / np.sqrt(12) * stats.norm.ppf(0.95) * capital
    var_99_1m = sigma_portfolio_annual / np.sqrt(12) * stats.norm.ppf(0.99) * capital
    cvar_95_1m = sigma_portfolio_annual / np.sqrt(12) * (
        stats.norm.pdf(stats.norm.ppf(0.95)) / 0.05
    ) * capital
    expected_1m_pnl = expected_annual_net / 12 * capital

    print(f"\nCapital: {capital:,.0f}:")
    print(f"  Expected monthly PnL (net): {expected_1m_pnl:+.1f}")
    print(f"  Monthly VaR (95%): {var_95_1m:,.1f}")
    print(f"  Monthly VaR (99%): {var_99_1m:,.1f}")
    print(f"  Monthly CVaR (95%): {cvar_95_1m:,.1f}")
    print(f"  Monthly std dev: {sigma_portfolio_annual/np.sqrt(12)*100:.1f}% "
          f"= {sigma_portfolio_annual/np.sqrt(12)*capital:,.1f}")

print(f"\nKelly fractions:")
print(f"  Full Kelly (gross): {f_kelly_full:.3f} ({f_kelly_full*100:.1f}% of capital)")
print(f"  Full Kelly (net): {f_kelly_net:.3f} ({f_kelly_net*100:.1f}% of capital)")
print(f"  Half Kelly (gross): {f_half_kelly:.3f} ({f_half_kelly*100:.1f}%)")
print(f"  Quarter Kelly (gross): {f_quarter_kelly:.3f} ({f_quarter_kelly*100:.1f}%)")
print(f"  Robust Kelly (lambda=2): {f_kelly_robust:.3f} ({f_kelly_robust*100:.1f}%)")

# 3.5 Optimal Kelly with log-wealth utility directly
# For a long-only strategy with discrete returns:
# E[log(1 + f*R)] is maximized
# Using the annual return distribution parameters
n_sim_kelly = 100_000
rng = np.random.default_rng(42)

# Simulate annual returns: r ~ N(mu_gross, sigma^2)
annual_returns_gross = rng.normal(expected_annual_gross, sigma_portfolio_annual, n_sim_kelly)
annual_returns_net = annual_returns_gross - annual_cost

# Find optimal f by grid search
f_grid = np.linspace(0.01, 3.0, 300)
expected_log_wealth = []
for f in f_grid:
    log_wealth = np.log(1 + f * annual_returns_net)
    expected_log_wealth.append(np.mean(log_wealth))

best_idx = np.argmax(expected_log_wealth)
f_optimal_empirical = f_grid[best_idx]
max_elw = expected_log_wealth[best_idx]

print(f"\nEmpirical Kelly (log-wealth maximization):")
print(f"  Optimal f (net returns): {f_optimal_empirical:.3f}")
print(f"  Max E[log(1+f*R)]: {max_elw:.6f}")

# ============================================================================
# Section 4: Monte Carlo Survival Simulation
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 4: MONTE CARLO SURVIVAL SIMULATION")
print("=" * 70)

N_SIMULATIONS = 20_000
N_MONTHS = 12
rng = np.random.default_rng(12345)

# 4.1 Model the IC process with uncertainty
# We consider two IC models:
#   Model A: IC ~ N(mean_IC, std_IC_window) — stationary, no decay
#   Model B: IC follows a decaying trend (based on the -0.079/window estimate)

# Monthly return: r_monthly = IC_monthly * sigma_5d * z * (50/12) - cost_monthly
# where IC_monthly ~ N(mu_IC, sigma_IC) approximately
# The IC std of 0.044 is window-level; monthly IC variance needs scaling
# For 50 periods/year, each period ~1 week; monthly is ~4 periods
# For simplicity, model monthly IC with same distribution parameters
# (conservative: higher monthly aggregation reduces noise)

sigma_ic_monthly = params.ic_std_window / np.sqrt(params.n_periods_per_year / 12)
# This gives ~0.044 / sqrt(4.17) ~ 0.0215

# However, the window IC varies more slowly. Let's model two levels:
# Level 1: "strategic" IC — the underlying regime, changes slowly (annually)
# Level 2: "tactical" IC noise — monthly variation around the regime

# For conservatism: use the full IC std of 0.044 as monthly variation
# This represents worse-than-expected noise

sigma_monthly_return = (
    params.ic_std_window * params.sigma_5d * params.z_score * params.n_periods_per_year / 12
)

# Annualized cost per month
cost_monthly = annual_cost / 12

# Baseline monthly expected return (net)
mu_monthly_net_base = expected_annual_net / 12

# 4.2 Simulation scenarios
scenarios = {
    "Optimistic (IC=0.097, MetaGate fused)": {
        "ic_mean": 0.097,
        "ic_std": params.ic_std_window,
        "label": "optimistic"
    },
    "Baseline (IC=0.075, LSTM mean)": {
        "ic_mean": params.ic_mean,
        "ic_std": params.ic_std_window,
        "label": "baseline"
    },
    "Recent (IC=0.057, last 3 windows)": {
        "ic_mean": 0.057,
        "ic_std": params.ic_std_window,
        "label": "recent"
    },
    "Held-out (IC=0.044, Ridge held-out)": {
        "ic_mean": 0.044,
        "ic_std": params.ic_std_window,
        "label": "heldout"
    },
    "Decaying (IC starts 0.075, decays -0.0079/month)": {
        "ic_mean": 0.0751,
        "ic_std": params.ic_std_window,
        "decay_per_month": -0.0079 * (12 / params.n_periods_per_year),  # ~-0.0019/month
        "label": "decaying"
    },
}

stop_loss_rules = {
    "No stop loss": {"type": "none"},
    "10% drawdown stop": {"type": "drawdown", "threshold": 0.10},
    "15% drawdown stop": {"type": "drawdown", "threshold": 0.15},
    "20% drawdown stop": {"type": "drawdown", "threshold": 0.20},
    "3 consecutive losing months": {"type": "consecutive_losing", "n": 3},
    "Cumulative -$200 stop (2k account)": {"type": "absolute", "amount": 200, "capital": 2000},
}

def compute_monthly_return(ic, sigma_5d, z, periods_per_year, cost_annual):
    """Convert IC to monthly net return."""
    annual_gross = ic * sigma_5d * z * periods_per_year
    annual_net = annual_gross - cost_annual
    return annual_net / 12

def run_simulation(ic_mean, ic_std, n_months, n_sim, decay_per_month=0.0,
                   initial_capital=2000, stop_rules=None):
    """
    Run Monte Carlo simulation of monthly portfolio returns.

    Parameters:
    - ic_mean: initial mean IC
    - ic_std: standard deviation of IC draws
    - n_months: number of months to simulate
    - n_sim: number of simulation paths
    - decay_per_month: IC decay per month (negative = deteriorating)
    - initial_capital: starting capital
    - stop_rules: list of stop-loss rule dictionaries

    Returns:
    - results: dict with survival statistics
    """
    if stop_rules is None:
        stop_rules = [{"type": "none"}]

    results = {}

    for rule in stop_rules:
        rule_name = rule.get("type", "none")
        if rule_name == "none":
            rule_name = "No stop loss"

        # Pre-generate IC draws
        # IC path: mean decays linearly
        ic_paths = np.zeros((n_sim, n_months))
        base_ic = ic_mean

        # Generate correlated IC draws (AR(1) process with rho=0.3 for realism)
        rho_ic = 0.3
        shocks = rng.normal(0, ic_std, (n_sim, n_months))
        for t in range(n_months):
            if t == 0:
                ic_paths[:, t] = base_ic + shocks[:, t]
            else:
                ic_paths[:, t] = (rho_ic * ic_paths[:, t-1] +
                                  (1 - rho_ic) * (base_ic + decay_per_month * t) +
                                  shocks[:, t] * np.sqrt(1 - rho_ic**2))

        # Compute monthly returns
        monthly_returns = np.zeros((n_sim, n_months))
        for t in range(n_months):
            monthly_returns[:, t] = compute_monthly_return(
                ic_paths[:, t], params.sigma_5d, params.z_score,
                params.n_periods_per_year, annual_cost
            )

        # Track survival
        capital = np.full(n_sim, float(initial_capital))
        peak = np.full(n_sim, float(initial_capital))
        alive = np.ones(n_sim, dtype=bool)
        stop_month = np.full(n_sim, n_months)  # month when stopped (n_months = survived)
        consecutive_losing = np.zeros(n_sim, dtype=int)

        for t in range(n_months):
            # Update capital for alive paths
            pnl = capital * monthly_returns[:, t]
            capital[alive] += pnl[alive]
            peak[alive] = np.maximum(peak[alive], capital[alive])

            # Check stop conditions
            newly_stopped = np.zeros(n_sim, dtype=bool)

            if rule["type"] == "drawdown":
                drawdown = (peak - capital) / peak
                newly_stopped = (drawdown > rule["threshold"]) & alive
            elif rule["type"] == "consecutive_losing":
                is_losing = monthly_returns[:, t] < 0
                consecutive_losing[alive & is_losing] += 1
                consecutive_losing[alive & ~is_losing] = 0
                newly_stopped = (consecutive_losing >= rule["n"]) & alive
            elif rule["type"] == "absolute":
                newly_stopped = (capital < rule["capital"] - rule["amount"]) & alive

            if np.any(newly_stopped):
                stop_month[newly_stopped] = t + 1  # month index (1-based)
                alive[newly_stopped] = False

        # After simulation: compute terminal statistics
        survived = stop_month == n_months
        survival_rate = np.mean(survived)
        terminal_capital = capital
        terminal_pnl = terminal_capital - initial_capital

        # For survived paths
        if np.any(survived):
            mean_terminal = np.mean(terminal_capital[survived])
            mean_pnl = np.mean(terminal_pnl[survived])
            mean_return = np.mean(terminal_pnl[survived]) / initial_capital * 100
        else:
            mean_terminal = np.nan
            mean_pnl = np.nan
            mean_return = np.nan

        # Probability of being profitable at end
        p_profitable = np.mean(terminal_pnl > 0)

        # Expected months survived
        mean_months = np.mean(stop_month)

        # Ruin probability (capital < 50% of initial)
        p_ruin = np.mean(capital < initial_capital * 0.5)

        # Distribution of final capital
        pctiles = [5, 10, 25, 50, 75, 90, 95]
        capital_pctiles = np.percentile(terminal_capital, pctiles)

        results[rule_name] = {
            "survival_rate": round(survival_rate * 100, 2),
            "mean_terminal_capital": round(float(mean_terminal), 1),
            "mean_pnl": round(float(mean_pnl), 1),
            "mean_return_pct": round(float(mean_return), 2) if not np.isnan(mean_return) else "N/A",
            "p_profitable": round(p_profitable * 100, 2),
            "p_ruin": round(p_ruin * 100, 2),
            "mean_months_survived": round(float(mean_months), 1),
            "median_terminal": round(float(capital_pctiles[3]), 1),
            "p5_terminal": round(float(capital_pctiles[0]), 1),
            "p95_terminal": round(float(capital_pctiles[6]), 1),
            "p25_terminal": round(float(capital_pctiles[2]), 1),
            "p75_terminal": round(float(capital_pctiles[4]), 1),
        }

        # Debug: show survival by month
        survival_by_month = []
        for t in range(1, n_months + 1):
            survival_by_month.append(round(np.mean(stop_month >= t) * 100, 1))

    return results

# Run simulations for each scenario and stop rule
all_results = {}

for scenario_name, scenario_params in scenarios.items():
    print(f"\nRunning scenario: {scenario_name}...")
    decay = scenario_params.get("decay_per_month", 0.0)
    results = run_simulation(
        ic_mean=scenario_params["ic_mean"],
        ic_std=scenario_params["ic_std"],
        n_months=N_MONTHS,
        n_sim=N_SIMULATIONS,
        decay_per_month=decay,
        initial_capital=2000,
        stop_rules=list(stop_loss_rules.values())
    )
    all_results[scenario_name] = results

    print(f"  {'Rule':<30s} {'Survival%':>8s} {'Mean PnL':>10s} {'P(Profit)%':>10s} {'P(Ruin)%':>8s} {'P5':>8s} {'Median':>8s} {'P95':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for rule_name, r in results.items():
        mean_pnl_str = f"${r['mean_pnl']:+.0f}" if isinstance(r['mean_pnl'], float) else "N/A"
        print(f"  {rule_name:<30s} {r['survival_rate']:>7.1f}% {mean_pnl_str:>10s} "
              f"{r['p_profitable']:>9.1f}% {r['p_ruin']:>7.1f}% "
              f"${r['p5_terminal']:>7.0f} ${r['median_terminal']:>7.0f} ${r['p95_terminal']:>7.0f}")

# ============================================================================
# Section 5: Capital Requirement Analysis
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 5: CAPITAL REQUIREMENT & BREAKEVEN ANALYSIS")
print("=" * 70)

# Minimum capital needed to survive 12 months with 95% confidence
# Using the monthly VaR framework:
# Required capital = 12 * |monthly expected loss| + z_95 * sigma_monthly * sqrt(12)
for ic_val, label in [(0.0751, "Baseline"), (0.057, "Recent"), (0.044, "Held-out")]:
    mu_annual = ic_val * params.sigma_5d * params.z_score * params.n_periods_per_year
    mu_annual_net = mu_annual - annual_cost
    mu_monthly = mu_annual_net / 12
    sigma_m = sigma_portfolio_annual / np.sqrt(12)

    # Capital needed: |E[loss]| * 12 + z_95 * sigma * sqrt(12)
    expected_12m_loss = -mu_monthly * 12
    risk_buffer = stats.norm.ppf(0.95) * sigma_m * np.sqrt(12)
    min_capital = expected_12m_loss + risk_buffer

    print(f"\n{label} (IC={ic_val:.4f}):")
    print(f"  Monthly expected net return: {mu_monthly*100:+.3f}%")
    print(f"  Expected 12-month return: {mu_annual_net*100:+.2f}%")
    print(f"  Minimum capital for 95% survival: ${min_capital:,.0f}")
    print(f"  Capital utilization (2k account): {min_capital/2000*100:.0f}%")

# ============================================================================
# Section 6: Sharpe Ratio Decomposition
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 6: SHARPE RATIO DECOMPOSITION")
print("=" * 70)

# Sharpe = IC * sqrt(BR) * TC (Grinold-Kahn with Transfer Coefficient)
# Given observed Sharpe, decompose into components
# SR = IC * sqrt(N * periods) * TC

# Using observed annual Sharpe (gross):
observed_sr = sharpe_gross
implied_tc = observed_sr / (params.ic_mean * np.sqrt(params.n_stocks * params.n_periods_per_year))

print(f"\nGrinold-Kahn decomposition:")
print(f"  IC = {params.ic_mean:.4f}")
print(f"  sqrt(Breadth) = sqrt({params.n_stocks} stocks × {params.n_periods_per_year} periods) = {np.sqrt(params.n_stocks * params.n_periods_per_year):.1f}")
print(f"  Theoretical max IR = IC × √BR = {params.ic_mean * np.sqrt(params.n_stocks * params.n_periods_per_year):.3f}")
print(f"  Observed gross Sharpe: {observed_sr:.3f}")
print(f"  Implied Transfer Coefficient: {implied_tc:.4f}")

# With effective breadth (accounting for correlation)
tc_effective = observed_sr / (params.ic_mean * np.sqrt(effective_breadth))
print(f"\nWith correlation-adjusted breadth:")
print(f"  Effective breadth = {effective_breadth:.0f}")
print(f"  sqrt(Effective breadth) = {np.sqrt(effective_breadth):.1f}")
print(f"  Implied TC (correlation-adjusted): {tc_effective:.4f}")
print(f"  Interpretation: TC={tc_effective:.2f} means ~{tc_effective*100:.0f}% of theoretical IC translates to portfolio returns")

# 6.1 What Sharpe would be needed for net profitability?
required_gross_return_for_breakeven = annual_cost + params.risk_free_rate
required_gross_sharpe = (required_gross_return_for_breakeven - params.risk_free_rate) / sigma_portfolio_annual
required_ic_for_breakeven = required_gross_return_for_breakeven / (params.sigma_5d * params.z_score * params.n_periods_per_year)

print(f"\nBreakeven requirements:")
print(f"  Required gross return: {required_gross_return_for_breakeven*100:.2f}%")
print(f"  Required gross Sharpe: {required_gross_sharpe:.3f}")
print(f"  Required IC: {required_ic_for_breakeven:.4f}")
print(f"  Current IC: {params.ic_mean:.4f}")
print(f"  IC gap: {required_ic_for_breakeven - params.ic_mean:.4f}")

# ============================================================================
# Section 7: Expected Max Drawdown (analytic approximation)
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 7: EXPECTED MAXIMUM DRAWDOWN")
print("=" * 70)

# Magdon-Ismail et al. approximation for max drawdown of Brownian motion with drift
# E[MDD] ≈ sigma^2 / (2*mu) * (gamma + log(2*mu*T/sigma^2)) for mu > 0
# where gamma ≈ 0.5772 (Euler's constant)

mu_daily = expected_annual_net / 252  # daily drift (net)
sigma_daily = sigma_portfolio_annual / np.sqrt(252)

if mu_daily > 0:
    gamma_euler = 0.5772156649
    T_days = 252
    expected_mdd_analytic = (sigma_daily**2 / (2 * mu_daily) *
                              (gamma_euler + np.log(2 * mu_daily * T_days / sigma_daily**2)))
else:
    # When drift is negative, use absolute drift for MDD approximation
    # For negative drift: expected MDD ≈ sigma * sqrt(T) * sqrt(pi/8) (random walk)
    expected_mdd_analytic = sigma_portfolio_annual * np.sqrt(np.pi / 8)

print(f"\nAnalytic Expected Maximum Drawdown (1 year):")
print(f"  Daily drift (net): {mu_daily*100:.4f}%")
print(f"  Daily sigma: {sigma_daily*100:.2f}%")
print(f"  Expected MDD: {expected_mdd_analytic*100:.2f}%")
print(f"  On $2,000: ${expected_mdd_analytic * 2000:.0f}")
print(f"  On ¥10,000: ¥{expected_mdd_analytic * 10000:.0f}")
print(f"  On ¥50,000: ¥{expected_mdd_analytic * 50000:.0f}")

# Monte Carlo MDD verification
n_paths_mdd = 100_000
daily_returns_mc = rng.normal(mu_daily, sigma_daily, (n_paths_mdd, 252))
cumulative = np.cumprod(1 + daily_returns_mc, axis=1)
running_max = np.maximum.accumulate(cumulative, axis=1)
drawdowns = (running_max - cumulative) / running_max
max_drawdowns = np.max(drawdowns, axis=1)
mdd_mean_mc = np.mean(max_drawdowns)
mdd_median_mc = np.median(max_drawdowns)
mdd_p95_mc = np.percentile(max_drawdowns, 95)

print(f"\nMonte Carlo MDD (100k paths):")
print(f"  Mean MDD: {mdd_mean_mc*100:.2f}%")
print(f"  Median MDD: {mdd_median_mc*100:.2f}%")
print(f"  95th percentile MDD: {mdd_p95_mc*100:.2f}%")

# ============================================================================
# Summary JSON output
# ============================================================================

output = {
    "grinold_kahn": {
        "expected_5d_gross_pct": round(expected_5d_gross * 100, 4),
        "expected_annual_gross_pct": round(expected_annual_gross * 100, 2),
        "expected_annual_net_pct": round(expected_annual_net * 100, 2),
        "annual_cost_pct": round(annual_cost * 100, 2),
        "portfolio_annual_sigma_pct": round(sigma_portfolio_annual * 100, 2),
        "sharpe_gross": round(sharpe_gross, 3),
        "sharpe_net": round(sharpe_net, 3),
        "ic_breakeven": round(ic_breakeven_full_cost, 4),
        "implied_transfer_coefficient": round(implied_tc, 4),
        "sensitivity": sensitivity,
    },
    "signal_analysis": {
        "ic_mean": params.ic_mean,
        "ic_std_window": params.ic_std_window,
        "decay_per_window": round(decay_per_window, 4),
        "decay_r_squared": round(r_value**2, 4),
        "first3_mean": round(first3.mean(), 4),
        "last3_mean": round(last3.mean(), 4),
        "p_ic_below_breakeven_pct": round(p_below_breakeven * 100, 1),
        "p_high_to_low": round(p_high_to_low, 2),
        "p_low_to_high": round(p_low_to_high, 2),
    },
    "kelly": {
        "f_kelly_full": round(f_kelly_full, 3),
        "f_kelly_net": round(f_kelly_net, 3),
        "f_half_kelly": round(f_half_kelly, 3),
        "f_quarter_kelly": round(f_quarter_kelly, 3),
        "f_robust": round(f_kelly_robust, 3),
        "f_empirical_net": round(f_optimal_empirical, 3),
    },
    "monte_carlo_survival": all_results,
    "drawdown": {
        "expected_mdd_analytic_pct": round(expected_mdd_analytic * 100, 2),
        "mdd_mean_mc_pct": round(mdd_mean_mc * 100, 2),
        "mdd_median_mc_pct": round(mdd_median_mc * 100, 2),
        "mdd_p95_mc_pct": round(mdd_p95_mc * 100, 2),
    },
    "capital_requirements": {
        "baseline_ic_0075": {
            "min_capital_95pct_survival": round(min_capital, 0),
        }
    }
}

output_path = "/home/xieguiawu/Desktop/ML/Glaubenskrieg/analysis/quantitative_results.json"
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n\nResults saved to {output_path}")
print("Done.")
