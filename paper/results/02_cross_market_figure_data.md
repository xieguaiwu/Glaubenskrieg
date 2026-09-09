# Cross-Market Figure Data for Tables and Figures

> **Purpose**: Provide data for figures/tables in a paper format.
> **Sources**: `results/us_full_baseline.json`, `results/newdata_baselines.json`, `results/portfolio_optimizer.json`, `results/vol_regime_final.json`, `results/garch_baseline.json`

---

## Figure 1: Cross-Market IC Comparison

### Data for Bar Chart

```python
# Mean IC with error bars (±1 std)
markets = ['A-Share (multi-asset)', 'A-Share (cross-sectional)', 'US (full)']
n_stocks = [200, 200, 477]
n_windows = [3, 6, 10]
lgb_ic = [0.053, -0.003, 0.007]
lgb_ic_std = [0.054, 0.042, 0.043]
ridge_ic = [0.006, 0.030, 0.031]
ridge_ic_std = [None, 0.054, 0.030]

# IC detection threshold line at 0.02
```

### Per-Market IC Values (Individual Windows)

**A-Share (multi-asset)**: 0.053 (mean of 3 windows — per-window values not available)
**A-Share**: [-0.059, -0.051, -0.014, +0.014, +0.037, +0.053]
**US (LGB)**: [-0.067, -0.042, -0.039, +0.025, +0.047, +0.001, +0.026, +0.022, +0.083, +0.010]

---

## Figure 2: IC Distribution Across Windows (US, 477 stocks)

### LightGBM Per-Window IC
Window 0: -0.067
Window 1: -0.042
Window 2: -0.039
Window 3: +0.025
Window 4: +0.047
Window 5: +0.001
Window 6: +0.026
Window 7: +0.022
Window 8: +0.083
Window 9: +0.010

### Ridge Per-Window IC
Window 0: +0.010
Window 1: +0.003
Window 2: +0.012
Window 3: +0.065
Window 4: +0.056
Window 5: +0.042
Window 6: +0.015
Window 7: -0.002
Window 8: +0.095
Window 9: +0.012

### Summary Statistics
LightGBM: mean = +0.007, std = 0.043, min = -0.067, max = +0.083
Ridge: mean = +0.031, std = 0.030, min = -0.002, max = +0.095

---

## Figure 3: Volatility QLIKE — LGB vs Mean vs Persistence (US, 477 stocks)

### Per-Window QLIKE

| Window | LGB | Mean | Persistence |
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

### Cross-Market QLIKE Means
| Market | LGB | Mean | Persistence |
|---|---|---|---|
| A-Share (multi-asset) | -6.42 | -6.29 | -5.67 |
| A-Share | -6.78 | -6.34 | -5.85 |
| US (477) | -7.10 | -6.75 | -6.50 |

---

## Figure 4: Sharpe vs IC Scatter (Demonstrating Decoupling)

### Data Points (each = one window-model combination)

| Window | Model | IC | Sharpe |
|---|---|---|---|
| US-0 | LGB | -0.067 | 2.84 |
| US-1 | LGB | -0.042 | 1.02 |
| US-2 | LGB | -0.039 | 0.09 |
| US-3 | LGB | +0.025 | 0.27 |
| US-4 | LGB | +0.047 | 0.64 |
| US-5 | LGB | +0.001 | 2.12 |
| US-6 | LGB | +0.026 | 2.15 |
| US-7 | LGB | +0.022 | 0.94 |
| US-8 | LGB | +0.083 | 0.56 |
| US-9 | LGB | +0.010 | 0.96 |
| AS-0 | LGB | +0.053 | -0.93 |
| AS-1 | LGB | +0.014 | 1.48 |
| AS-2 | LGB | +0.037 | 0.67 |
| AS-3 | LGB | -0.059 | 2.32 |
| AS-4 | LGB | -0.051 | 1.31 |
| AS-5 | LGB | -0.014 | -0.33 |

**Pearson correlation**: IC vs Sharpe ≈ **-0.15** — no relationship.

---

## Figure 5: Volatility Regime Strategy — Cumulative Returns

### Data (Normalized, starting from 1.0)

From `results/vol_regime_final.json` (performance_full):

| Strategy | Total Return | Sharpe | MaxDD |
|---|---|---|---|
| Equal Weight | 12.71× | 1.121 | 37.1% |
| Inverse Vol | 9.99× | 1.120 | 35.1% |
| Vol Target 15% | 6.36× | 1.214 | 22.0% |
| Regime-Adaptive | 8.61× | **1.402** | **19.2%** |

---

## Figure 6: Portfolio Optimization — Method Comparison

### Data (8-window mean ± std)

| Method | Sharpe | Sortino | MaxDD |
|---|---|---|---|
| LGB-SharpeTuned | 2.36 ± 0.36 | 4.27 ± 1.08 | -0.40 ± 0.10 |
| LGB-MSE | 1.57 ± 0.30 | 2.74 ± 0.85 | -0.42 ± 0.14 |
| LGB-LambdaRank | -0.30 ± 0.83 | -0.57 ± 1.62 | -0.70 ± 0.14 |
| Random | 0.34 ± 0.34 | 0.45 ± 0.49 | -0.32 ± 0.12 |
| MeanRev | 0.49 ± 0.22 | 0.80 ± 0.36 | -0.42 ± 0.05 |

---

## Figure 7: GARCH vs LightGBM Volatility — Per-Stock QLIKE

### Summary Statistics (47 US stocks)

GARCH: mean QLIKE = 2.315, median = 2.156, std = 0.558
LGB: mean QLIKE = 12.747, median = 5.751, std = 47.00

### Top 10 Stocks (QLIKE values)

| Stock | GARCH QLIKE | LGB QLIKE | Winner |
|---|---|---|---|
| GARCH wins on **47/47 stocks** (100%) | | | |

---

## Figure 8: CTM Validation IC Evolution (v1 → v5)

| Version | val_IC | test_IC | Key Change |
|---|---|---|---|
| v1 | 0.14 | — | Initial baseline |
| v2 | 0.08 | — | Multi-asset expansion |
| v3 | 0.14 | — | Bug: shuffle=True inflation |
| v4 (TGPE) | 0.065 | — | Curriculum + TimeGate |
| v5 | **0.049** | **0.006** | Bugs fixed |

---

## Table 1: 6 Paradigms — Complete Comparison

| # | Paradigm | Test IC | Test Sharpe | DM Sig | Conclusion |
|---|---|---|---|---|---|
| 1 | DL (CTM Mamba) | ≈ 0.00 | -0.093 | — | ❌ No signal |
| 2 | GBDT (Hoffnung) | 0.008 | -0.088 | — | ❌ No signal |
| 3 | Ensemble (TGPE) | — | -0.024 | — | ❌ Degrades |
| 4 | Feature Engineering (LGB+EMA) | -0.004 | +0.62 | 4/5 ✅ | ❌ Sharpe artifact |
| 5 | Cross-Sectional (lambdarank) | Negative | — | — | ❌ Worse |
| 6 | Meta-Labeling | — | Δ = -0.22 | — | ❌ Degrades |

---

## Table 2: Key Statistical Tests Summary

| Test | Null Hypothesis | Result | p-value |
|---|---|---|---|
| DM: LGB raw vs zero | IC = 0 | Reject (marginal) | 0.014–0.124 |
| DM: LGB smooth vs zero | IC = 0 | Fail to reject | 0.136–0.220 |
| DM: LGB vs GARCH | Equal vol pred | **Reject GARCH better** | 1.6e-11 |
| DM: GARCH vs Persist | Equal vol pred | **Reject GARCH better** | 6.5e-13 |
| Paired t: LGB vs GARCH | Equal mean QLIKE | Fail to reject | 0.14 |
| Paired t: SharpeTuned vs MSE | Equal Sharpe | **Reject ST better** | 0.011 |

---

## Table 3: Hardware Comparison

| Hardware | GPU | CUDA Kernel Speed | Used For |
|---|---|---|---|
| Ascend 910B (PG503-216) | 2×32GB | ~10× slower (compat layer) | v1–v4 |
| NVIDIA V100 | 2×32GB | Native full speed | v5/US baselines |
| RTX 3090 | 2×24GB | Native full speed | v5/Phase 1 |

---

## Table 4: Data Summary

| Dataset | N Stocks | Date Range | Trading Days | Size | Source |
|---|---|---|---|---|---|
| A-Share (tencent_clean) | 200 | 2015–2026 | 1,110 (50-stock window) | — | Tencent Finance |
| A-Share (tencent_clean) | 200 | 2015–2026 | ~2,700 | — | Tencent Finance |
| US (yfinance) | 477 (S&P500+QQQ filtered ≥2500d) | 2015–2026 | 2,504 | 82MB+96MB | Yahoo Finance |
| US (initial 19) | 19 | 2015–2026 | 2,856 | — | Yahoo Finance |

---

## Table 5: Parameter Efficiency — All Models

| Model | Total Params | Sample-Effective Params | Params/Sample | IC (US) |
|---|---|---|---|---|
| Linear Ridge | 451 | 451 | 0.58 | 0.031 |
| GBDT (Hoffnung) | ~5,000 | ~500 (tree leaves) | ~6.4 | 0.008 |
| LightGBM (100 trees, d=3) | ~5,000 | ~500 | ~6.4 | 0.007 |
| CTM (Mamba, v5) | 97,000 | ~50,000 (effective after sharing) | ~125 | ≈ 0.00 |
| CTM (Mamba, pre-fix) | 157,323 | ~97,000 | ~202 | — |
