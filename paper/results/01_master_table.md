# Master Results Table: All Experimental Results

> **Compiled from**: All source files cross-referenced.
> **Date**: 2026-06-06

---

## 1. Cross-Market Return Prediction Comparison

| Market | N Stocks | N Windows | Years | Model | Mean IC | IC Std | Mean Sharpe | Vol QLIKE |
|---|---|---|---|---|---|---|---|---|
| **A-Share (multi-asset)** | 50 | 3 | 3 | LightGBM | +0.053 | 0.054 | +0.19 | -6.42 |
| **A-Share (multi-asset)** | 50 | 3 | 3 | Ridge | +0.006 | — | +0.55 | — |
| **A-Share** | 200 | 6 | 2.4 | LightGBM | -0.003 | 0.042 | +0.75 | -6.78 |
| **A-Share** | 200 | 6 | 2.4 | Ridge | +0.030 | 0.054 | +0.48 | — |
| **US (full)** | 477 | 10 | 11 | LightGBM | +0.007 | 0.043 | +1.16 | -7.10 |
| **US (full)** | 477 | 10 | 11 | Ridge | +0.031 | 0.030 | +0.82 | — |
| **US (19 stocks)** | 19 | 13 | 11 | LightGBM | +0.025 | 0.032 | +1.42 | -7.15 |

**Sources**: `progress.md` §10.5, `results/newdata_baselines.json`, `results/us_full_baseline.json`.

---

## 2. v5 Final: CTM × Ensemble × GBDT (A-Share multi-asset, 50 stocks, 3 windows)

### CTM (Mamba SSM, 97K params)

| Seed | n_windows | val_IC | test_IC | test_Sharpe |
|---|---|---|---|---|
| 42 | 3 | 0.044 | -0.006 | -0.092 |
| 123 | 3 | 0.049 | — | -0.016 |
| 456 | 3 | 0.050 | — | -0.268 |
| 789 | 3 | 0.034 | — | -0.130 |
| 1024 | 3 | 0.065 | — | +0.041 |
| **Mean** | — | **0.049** | **≈ 0.00** | **-0.093** |

### Ensemble+TimeGate (157K params)

| Seed | n_windows | val_IC | test_IC | test_Sharpe |
|---|---|---|---|---|
| 42 | 3 | 0.021 | — | -0.098 |
| 123 | 3 | 0.036 | — | +0.141 |
| 456 | 3 | 0.051 | — | -0.048 |
| 789 | 3 | 0.045 | — | -0.051 |
| 1024 | 3 | 0.034 | — | -0.063 |
| **Mean** | — | **0.037** | — | **-0.024** |

### GBDT (Hoffnung C++, ~5K params)

| Seed | n_windows | val_IC | test_IC | test_Sharpe |
|---|---|---|---|---|
| 42 | 3 | 0.005 | — | -0.531 |
| 123 | 3 | 0.009 | — | -0.108 |
| 456 | 3 | 0.004 | — | +0.209 |
| 789 | 3 | 0.008 | — | -0.444 |
| 1024 | 3 | 0.015 | — | +0.432 |
| **Mean** | — | **0.008** | — | **-0.088** |

**Source**: `progress.md` (§3), `.omo/session_context.md` (§3).

---

## 3. Phase 1: Feature-Engineered LightGBM + DM Test

| Config | test_IC | test_Sharpe | DM p range | Sig@5% |
|---|---|---|---|---|
| EMA=1 (raw) | -0.004 ± 0.007 | 0.62 ± 0.03 | 0.014–0.124 | 4/5 ✅ |
| EMA=10 (smooth) | -0.013 ± 0.007 | 0.47 ± 0.02 | 0.136–0.220 | 0/3 ❌ |

**Source**: `progress.md` (Stage 7), `docs/architecture_assessment.md` (Appendix D.3).

---

## 4. Volatility Prediction: All Experiments

| Experiment | Predictor | QLIKE | MSE | Δ vs Mean | Δ vs Persist |
|---|---|---|---|---|---|
| **A-Share (multi-asset)** (50 stocks, 3 win) | LightGBM | -6.42 ± 0.24 | — | ~-0.13 | ~-0.75 |
| **A-Share** (200 stocks, 6 win) | LightGBM | -6.78 ± 0.26 | 2.57e-5 | -0.44 | -0.92 |
| **A-Share** (200 stocks) | Mean | -6.34 | 1.37e-4 | — | — |
| **A-Share** (200 stocks) | Persistence | -5.85 | 2.14e-4 | — | — |
| **US** (477 stocks, 10 win) | LightGBM | -7.10 ± 0.20 | 2.44e-5 | -0.35 | -0.60 |
| **US** (477 stocks) | Mean | -6.75 | 9.05e-5 | — | — |
| **US** (477 stocks) | Persistence | -6.50 | 5.00e-4 | — | — |
| **US** (19 stocks, 13 win) | LightGBM | -7.15 ± 0.33 | 3.95e-5 | -0.79 | -1.03 |
| **US** (19 stocks) | Mean | -6.36 | 1.47e-4 | — | — |
| **US** (19 stocks) | Persistence | -6.12 | 1.48e-4 | — | — |

**Sources**: `results/newdata_baselines.json`, `results/us_full_baseline.json`.

---

## 5. GARCH(1,1) vs LightGBM (47 US stocks)

| Predictor | Mean QLIKE | Median QLIKE | Win Rate vs GARCH | Avg Rank |
|---|---|---|---|---|
| **GARCH(1,1)** | 2.315 | 2.156 | — | **1st** |
| Mean | 2.434 | 2.262 | — | 2nd |
| **LightGBM** | **12.747** | **5.751** | **0%** (0/47) | 3rd |
| Persistence | 436.9 | 291.8 | — | 4th |

**Significance**: DM LGB vs GARCH: DM = -6.73, p = 1.6e-11 (GARCH better).
**Paired t-test**: t = -1.50, p = 0.14 (LGB numerically worse, not significant).

**Source**: `results/garch_baseline.json`.

---

## 6. Portfolio Optimization (50 US stocks, 8 windows)

| Method | Mean Sharpe | Mean Sortino | Mean IC | IC Std |
|---|---|---|---|---|
| **LGB-SharpeTuned** | **2.361** | **4.270** | -0.014 | 0.127 |
| LGB-MSE | 1.571 | 2.740 | -0.010 | 0.138 |
| LGB-LambdaRank | -0.302 | -0.574 | -0.019 | 0.160 |
| Random | 0.335 | 0.445 | -0.003 | 0.145 |
| MeanRev | 0.486 | 0.795 | -0.009 | 0.180 |

**Source**: `results/portfolio_optimizer.json`.

---

## 7. Volatility Regime Strategy (200 US stocks, top-50, 2015-2026)

| Strategy | Sharpe | Ann Return | Ann Vol | MaxDD | Δ Sharpe |
|---|---|---|---|---|---|
| Equal Weight | 1.121 | 26.2% | 23.3% | 37.1% | — |
| Inverse Vol | 1.120 | 23.8% | 21.2% | 35.1% | -0.001 |
| Vol Target 15% | 1.214 | 19.5% | 15.8% | 22.0% | +0.093 |
| **Regime-Adaptive** | **1.402** | **22.3%** | **15.3%** | **19.2%** | **+0.281** |

**Source**: `results/vol_regime_final.json`.

---

## 8. Meta-Labeling (5 seeds)

| Seed | Δ Sharpe (WF) | Δ Sharpe (Hold-out) |
|---|---|---|
| 42 | -0.15 | -0.22 |
| 123 | -0.08 | -0.31 |
| 456 | -0.12 | -0.18 |
| 789 | -0.19 | -0.27 |
| 1024 | -0.11 | -0.14 |
| **Mean** | **-0.13** | **-0.22** |

**Source**: `.omo/session_context.md` (§3, Meta-Labeling section).

---

## 9. GBDT Parameter Sweep (9 configs × 5 seeds)

| Config | Sharpe | IC |
|---|---|---|
| T100 D4 | +0.345 | +0.020 |
| T100 D6 | +0.344 | +0.021 |
| T100 D8 | +0.345 | +0.020 |
| T200 D4 | +0.345 | +0.020 |
| T200 D6 | +0.345 | +0.020 |
| T200 D8 | +0.345 | +0.019 |
| T500 D4 | +0.344 | +0.022 |
| T500 D6 | +0.345 | +0.021 |
| T500 D8 | +0.345 | +0.020 |

**Conclusion**: Completely flat response surface. Any config works equally (poorly).

**Source**: `docs/architecture_assessment.md` §1.3.

---

## 10. Cross-Sectional Ranking (lambdarank, 3 seeds)

| Seed | WF Rank IC | Test Rank IC | Verdict |
|---|---|---|---|
| 42 | Negative | Negative | ❌ |
| 123 | Negative | Negative | ❌ |
| 456 | Negative | Negative | ❌ |

**Source**: `.omo/session_context.md` (§3, Cross-Sectional Ranking section).

---

## 11. Volatility Backtest on A-Share (199 stocks, 11yr)

| Metric | Strategy | Benchmark | Delta |
|---|---|---|---|
| Sharpe | +0.052 | -0.003 | +0.056 |
| Ann Return | -0.68% | -2.05% | +1.37pp |
| Ann Vol | 19.25% | 20.80% | -1.55pp |
| MaxDD | 45.16% | 51.96% | -6.80pp |

**Source**: `progress.md` §10.4.

---

## 12. Summary Statistics Across All Experiments

| Category | N runs | Best Test IC | Best Test Sharpe | Statistically Significant? |
|---|---|---|---|---|
| Return prediction (any model) | 45+ | 0.031 (Ridge, US) | 1.16 (LGB, US artifact) | ❌ No (IC < 0.02 threshold) |
| Volatility prediction | 25+ | QLIKE=-7.10 (US) | — | ✅ QLIKE beats mean/persistence |
| Portfolio optimization | 8 | — | 2.36 (SharpeTuned, artifact) | ❌ Artifact of metric optimization |
| Meta-Labeling | 5 | — | Δ = -0.22 | ❌ Degrades performance |
| Cross-sectional | 3 | Negative | Negative | ❌ Worse than random |
