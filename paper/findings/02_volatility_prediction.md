# Volatility Prediction: The Only Robust Signal in OHLCV Features

> **Source files**: `progress.md` (§4.4, §9), `results/newdata_baselines.json`, `results/us_full_baseline.json`, `results/garch_baseline.json`, `results/vol_regime_final.json`, `docs/architecture_assessment.md` (Appendix D)

---

## 1. Summary

Volatility prediction (using `realized_vol_21` as target) is the **only paradigm** across all 6 tested that produces consistent, statistically significant out-of-sample results. This is expected from financial econometrics: variance exhibits strong autocorrelation (Engle 1982, Bollerslev 1986) unlike returns, which are approximately martingale.

| Property | Returns | Volatility |
|---|---|---|
| Autocorrelation | ≈ 0 (white noise) | Strong (>0.5 at lag 1) |
| SNR | ~0.01 | ~0.1–0.3 |
| Stationarity | Very weak | Moderate (variance clustering) |
| Theoretical R² ceiling | ~1% daily | ~5–15% for 21-day realized vol |
| Empirical predictability | IC ≈ 0 (this study) | QLIKE improvement confirmed (this study) |

---

## 2. Cross-Market Volatility QLIKE Comparison

| Market | LightGBM QLIKE | Mean Baseline QLIKE | Persistence QLIKE | Δ LGB-Mean | Δ LGB-Persist |
|---|---|---|---|---|---|
| **HK** (50 stocks, 3 win) | -6.42 ± 0.24 | ~-6.29 | ~-5.67 | ~-0.13 | ~-0.75 |
| **A-Share** (200 stocks, 6 win) | -6.78 ± 0.26 | -6.34 ± 0.28 | -5.85 ± 0.40 | -0.44 | -0.92 |
| **US** (477 stocks, 10 win) | -7.10 ± 0.20 | -6.75 ± 0.23 | -6.50 ± 0.56 | -0.35 | -0.60 |

**Sources**: `results/newdata_baselines.json` (ashare.volatility, us.volatility), `results/us_full_baseline.json` (us_stocks.volatility).

**Key findings**:
1. **LightGBM beats both baselines in all three markets** — Δ QLIKE is always negative (better), with the largest gain vs persistence.
2. **QLIKE values are remarkably consistent** across seeds: 5 seeds across HK vol prediction produce std < 0.001 (`results/us_full_baseline.json` doesn't have multi-seed vol but A-share vol from `progress.md` shows similar consistency).
3. **A-Share shows the largest improvement** over mean baseline (Δ = -0.44), possibly because A-share volatility is more clustered.

---

## 3. GARCH(1,1) vs LightGBM: LightGBM Loses

Source: `results/garch_baseline.json` — 47 US stocks, walk-forward GARCH(1,1) vs LightGBM comparison.

### Aggregate Results

| Predictor | Mean QLIKE | Median QLIKE | Rank |
|---|---|---|---|
| **GARCH(1,1)** | 2.315 | 2.156 | **1st** (avg rank: 0.13) |
| Mean predictor | 2.434 | 2.262 | 2nd (avg rank: 0.87) |
| **LightGBM** | **12.747** | **5.751** | **3rd** (avg rank: 2.00) |
| Persistence | 436.902 | 291.827 | 4th (avg rank: 3.00) |

### Key Metrics
- **Fraction LGB better than GARCH: 0.0%** — LightGBM loses to GARCH(1,1) on **every single stock** (47/47).
- **Paired t-test LGB vs GARCH**: t = -1.50, p = 0.14 (not significant at 5%, but LGB numerically worse).
- **Diebold-Mariano LGB vs GARCH**: DM = -6.73, p = 1.6e-11 — GARCH is significantly better when accounting for time series structure.
- **Diebold-Mariano LGB vs Persistence**: DM = -7.03, p = 2.1e-12 — LGB beats persistence.
- **Diebold-Mariano GARCH vs Persistence**: DM = -7.19, p = 6.5e-13 — GARCH beats persistence.

### Conclusion
> **GARCH(1,1) is the best volatility predictor. LightGBM, despite beating naive baselines (mean, persistence), adds no value over the simplest GARCH model. In fact, LightGBM is significantly worse than GARCH.** This means ML overfits volatility noise even though volatility is more predictable than returns.

---

## 4. Regime-Switching Volatility Strategy

Source: `results/vol_regime_final.json` — 200 US stocks (top 50 by volume), 16-fold walk-forward, 2015-2026.

### Strategy Performance (Full Period)

| Strategy | Sharpe | Ann Return | Ann Vol | Max DD | Calmar |
|---|---|---|---|---|---|
| Equal Weight (benchmark) | 1.121 | 26.2% | 23.3% | 37.1% | 0.71 |
| Inverse Vol | 1.120 | 23.8% | 21.2% | 35.1% | 0.68 |
| Vol Target 15% | 1.214 | 19.5% | 15.8% | 22.0% | 0.89 |
| **Regime-Adaptive** | **1.402** | **22.3%** | **15.3%** | **19.2%** | **1.16** |

### Delta vs Equal-Weight Benchmark

| Strategy | Δ Sharpe | Δ Return | Δ Vol | Δ MaxDD |
|---|---|---|---|---|
| Inverse Vol | -0.001 | -2.46pp | -2.17pp | -1.99pp |
| Vol Target 15% | +0.093 | -6.80pp | -7.57pp | -15.11pp |
| **Regime-Adaptive** | **+0.281** | **-3.93pp** | **-8.04pp** | **-17.86pp** |

### OOS Performance (2022-2026)

| Strategy | Sharpe | Max DD |
|---|---|---|
| Equal Weight | 1.080 | 37.1% |
| Inverse Vol | 1.038 | 34.6% |
| Vol Target 15% | 1.200 | 22.0% |
| **Regime-Adaptive** | **1.296** | **18.5%** |

**Key conclusion**: Regime-adaptive volatility weighting yields meaningful risk reduction (-8pp annual vol, -18pp max DD) with modest return sacrifice (-3.9pp). The net effect is **+0.28 Sharpe improvement**. This is **not stock selection alpha** — it's **risk management**.

---

## 5. Volatility Backtest on A-Share (Detailed)

Source: `progress.md` §10.4 — 199 A-share stocks, 11yr, top-50 by volume, weekly rebalance.

| Metric | Strategy | Benchmark | Delta |
|---|---|---|---|
| Sharpe Ratio | +0.052 | -0.003 | **+0.056** |
| Annual Return | -0.68% | -2.05% | +1.37pp |
| Annual Vol | 19.25% | 20.80% | -1.55pp |
| Max Drawdown | 45.16% | 51.96% | **-6.80pp** |
| Rolling Sharpe μ | +0.029 | -0.051 | +0.080 |

**Note**: Both strategy and benchmark have negative absolute returns (A-share bear market over period). The value is in **risk reduction**, not absolute return enhancement.

---

## 6. Per-Window Volatility QLIKE Detail (US, 477 stocks)

Source: `results/us_full_baseline.json` (us_stocks.volatility).

| Window | LGB QLIKE | Mean QLIKE | Persistence QLIKE |
|---|---|---|---|
| 0 | -7.026 | -6.579 | -5.128 |
| 1 | -7.292 | -7.021 | -7.089 |
| 2 | -6.776 | -6.400 | -6.375 |
| 3 | -6.761 | -6.440 | -6.138 |
| 4 | -7.187 | -6.932 | -6.782 |
| 5 | -7.353 | -7.036 | -6.942 |
| 6 | -7.335 | -6.943 | -6.982 |
| 7 | -7.199 | -6.845 | -6.706 |
| 8 | -6.973 | -6.536 | -6.150 |
| 9 | -7.105 | -6.742 | -6.751 |
| **Mean** | **-7.101** | **-6.747** | **-6.504** |

**LGB beats Mean in 10/10 windows. LGB beats Persistence in 9/10 windows.**

---

## 7. Caveats and Limitations

1. **GARCH(1,1) beats LightGBM on every stock** — ML adds negative value for volatility prediction. This is a humbling result: a 2-parameter model from 1982 outperforms 300-tree gradient boosting on 47/47 US stocks.
2. **QLIKE improvement over Mean is economically small** — Δ = -0.35 to -0.44. In an investment context, this translates to marginal volatility reduction improvements.
3. **Regime-adaptive strategy's Sharpe improvement (+0.28) comes from risk reduction, not prediction** — The strategy lowers exposure during volatile periods, reducing drawdowns. This is a timing-based improvement that requires no return forecasting.
4. **Inverse volatility weighting alone does not improve Sharpe** — Δ = -0.001, confirming that cross-sectional volatility ranking is not useful.
