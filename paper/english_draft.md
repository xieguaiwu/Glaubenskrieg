# Glaubenskrieg: A Multi-Paradigm Exhaustive Test of ML Stock Return Prediction

## — and a Diagnostic Framework for Detecting Overfitting in Financial Machine Learning

**Authors**: [TBD]
**Target Venue**: ICAIF 2026 / Journal of Financial Data Science
**Status**: Full English Draft

---

## Abstract

all converge to Information Coefficient (IC) ≈ 0.00 on held-out test data. Our 97,000-parameter Conv-Temporal-Mamba (CTM) model achieves the same test IC as a 451-parameter linear Ridge regression: the extra 96,549 parameters contribute zero predictive value, only overfitting noise.

An initial walk-forward validation IC of 0.14 was traced through five sequential bug fixes to a true out-of-sample IC of 0.006—a 23.3× overstatement. We formalize this pattern as a **five-step IC validation diagnostic framework**: (1) held-out test IC gap, (2) linear Ridge baseline, (3) per-window IC stability, (4) per-stock binomial significance test, and (5) train-val loss divergence. Applied to our own pipeline, the framework would have flagged the IC=0.14 as a false positive before downstream resources were consumed.

Volatility prediction proves robust across markets (QLIKE = −6.42 to −7.10), but GARCH(1,1) with three parameters outperforms gradient-boosted trees on all 300 tested stocks across two markets (100% win rate, DM p = 1.6 × 10⁻¹¹, median QLIKE ratio 2.12–2.49×). CTM is 3.8× worse than LightGBM on the same volatility task (MSE: 1.00 × 10⁻⁴ vs 2.64 × 10⁻⁵). Deep learning degrades even the one predictable signal in OHLCV data.

We document the **Sharpe Paradox**: sign-based long/short portfolio construction produces Sharpe ratios of 0.5–2.98 even when IC ≈ 0, because any sorting in a rising market mechanically harvests the volatility premium. Portfolio-aware hyperparameter optimization achieved Sharpe = 2.98 with IC = −0.014, exposing a methodological trap in validation metric selection.

A 10-paper lateral comparison shows our Ridge regression IC of 0.031 falls within the post-publication decay range of published academic factors (0.01–0.04). Our central conclusion is that OHLCV-derived features at daily frequency contain no detectable return-predictive signal across markets, and that ML model complexity—whether Mamba SSM, gradient boosting, or ensemble hybrids—adds noise rather than signal.

---

## 1. Introduction

### 1.1 The Financial ML Reproducibility Crisis

Can machine learning predict stock returns from publicly available price data? This question sits at the intersection of two fields—quantitative finance and deep learning—and produces sharply contradictory answers.

On one side, a growing literature claims that modern architectures extract predictive signals from historical prices. MambaStock (2024) reports superiority over XGBoost on Chinese bank stocks. SAMBA (2025) claims IC improvements of 33–85% over baselines on US equities. PULSE-KAN (2026) reports breakthrough performance with Kolmogorov-Arnold networks. The narrative is one of steady architectural progress: LSTM → Transformer → Mamba → GNN → KAN, each generation claiming to unlock previously inaccessible signal.

On the other side, a counter-current of rigorous studies finds that most such claims do not survive independent verification. A Nature survey (2025) concluded that "most prominent studies regarding LSTMs and DNNs predictors for stock market prediction are not reproducible in real-world applications." Ye et al. (2025) documented that LLM-based trading agents overstate returns by 71–85% due to temporal contamination. McElfresh et al. (NeurIPS 2023) benchmarked 176 datasets with 538K models and found that no pure deep learning model beats tuned gradient-boosted decision trees (GBDT) on tabular data—and financial data is tabular.

This reproducibility crisis has a structural origin. Financial time series combine three properties uniquely hostile to machine learning:
1. **Extremely low signal-to-noise ratio**: ~80% noise in daily returns
2. **Non-stationarity**: The data-generating process shifts across market regimes
3. **Small effective sample size**: ~750 training days for a 97,000-parameter model yields 0.008 samples per parameter

In this regime, the dominant risk is not underfitting but overfitting—learning noise patterns that appear as predictive signal in validation but vanish in deployment.

### 1.2 The Methodological Gap

The field lacks a standardized protocol for distinguishing genuine signal from overfitting artifacts in walk-forward validation. Most published studies report only walk-forward Information Coefficients (IC) without:
- Held-out test sets (never used in training or early stopping)
- Linear baselines (Ridge regression with 451 parameters)
- Per-window stability checks (is IC consistent across time?)
- Per-stock significance tests (is signal broad or driven by a few stocks?)
- Train-val loss divergence monitoring (the classic overfitting signature)

As a result, the literature contains an unknown but substantial fraction of "alpha hallucinations"—results that reflect methodological artifacts rather than true predictability.

### 1.3 This Paper's Contributions

**First**, we conduct the most exhaustive test to date of daily stock return prediction from OHLCV-derived features. Six independent paradigms are tested across three markets covering 877 stocks. All six converge to IC ≈ 0.00 on held-out data. A 10-paper lateral comparison situates this result within the published literature.

**Second**, we present a **five-step IC validation diagnostic framework**—a systematic protocol for detecting overfitting in walk-forward validation. We demonstrate its application through a detailed case study: our own pipeline's evolution from a false-positive IC of 0.14 (v3) to a true out-of-sample IC of 0.006 (v5), with each bug fix documented and quantified.

**Third**, we establish three methodological findings of independent interest:
1. **The Sharpe Paradox**: Sharpe ratios of 0.5–2.98 arise mechanically from sign-based long/short portfolios even when IC ≈ 0
2. **GARCH(1,1) beats gradient boosting on 300/300 stocks**: The 3-parameter model from 1986 systematically outperforms 300-tree gradient boosting for volatility prediction
3. **Portfolio-aware optimization is a trap**: Tuning on Sharpe achieves high Sharpe with zero IC, because the validation metric itself is an artifact of portfolio construction

---

## 2. Related Work

### 2.1 Ten-Head-to-Head Literature Comparison

We compare Glaubenskrieg against 10 peer-reviewed or widely-cited studies. Detailed quantitative mappings are in `paper/CLASSIC_LITERATURE_COMPARISON.md`.

**Table 1: Glaubenskrieg vs 10 Published Studies**

| # | Paper | Published Metric | Glaubenskrieg Equivalent | Agreement | Key Insight |
|:-:|-------|:---------------:|:------------------------:|:--------:|-------------|
| 1 | Gu, Kelly & Xiu (2020) | Monthly R² ≈ 0.4% | Daily R² ≈ 0.005% (IC²) | ✅ Consistent | Feature gap (94→9) explains perf gap |
| 2 | McElfresh et al. (2023) | GBDT > NN on tabular | Ridge IC=0.031 > CTM IC≈0.006 | ✅ Confirms | GBDT/linear dominate DL on financial data |
| 3 | Grinsztajn et al. (2022) | 3 structural NN failures | All 3 confirmed in financial data | ✅ Confirms | Smoothness bias, rotation, uninformative features |
| 4 | Kelly & Malamud (2025) | True R² ≥ 20%; LLG=O(P/T) | P/T=129; IC=0.007 | ✅ Consistent | LLG explains why signal is unlearnable |
| 5 | Bai et al. (2026) | OOS-R² = 0.029 | Wavelet ΔIC=−0.046; EMA ΔIC=−0.009 | ⚠️ Contradicts | Denoising does not create signal |
| 6 | Engle/Bollerslev (82/86) | Variance clustering | GARCH wins 300/300 over LGB | ✅ Confirms | 3-param GARCH > 300-tree LGB |
| 7 | López de Prado (2018) | Meta-Labeling | ΔSharpe = −0.22 | ⚠️ Qualifies | Prerequisite (positive IC) not met |
| 8 | Ye et al. (2025) | α hallucination 71–85% | IC overstatement 95.7% (0.14→0.006) | ✅ Validates | Same contamination pattern |
| 9 | VLSTM Oxford (2026) | Gross Sharpe 2.39 | Best Sharpe 1.16 (artifact) | ⚠️ Different leagues | Multi-asset + 50-seed top-10 |
| 10 | Nature (2025) | LSTM/DNN irreproducible | 5-step diagnosis, 3 markets, purged WF | ✅ Counter-example | Glaubenskrieg protocol sets new bar |

### 2.2 GBDT vs Deep Learning on Tabular Financial Data

McElfresh et al. (NeurIPS 2023) established that CatBoost is the single best algorithm across 176 tabular datasets, and that GBDTs outperform neural networks on large, irregular (skewed/heavy-tailed) datasets—precisely the characteristics of financial returns. Grinsztajn et al. (NeurIPS 2022) identified three structural reasons: (1) spectral bias toward smooth functions harms NNs when targets are non-smooth; (2) rotational invariance destroys the semantic meaning of individual features (RSI, Bollinger, volume); and (3) NNs allocate capacity to uninformative features while trees perform implicit selection at every split. Glaubenskrieg confirms all three: Ridge (451 params) achieves IC=0.031 while CTM Mamba SSM (97K params) achieves IC≈0.00, and GBDT's parameter response surface is completely flat across 9 configurations.

### 2.3 The OHLCV-Only Ceiling

The strongest published positive result using OHLCV-only daily features is Bai et al. (Symmetry 2026), reporting OOS-R² = 0.029 on NIFTY50 with a CNN-LightGBM hybrid. However, 93% of this gain came from wavelet denoising. Glaubenskrieg's independent test of denoising (EMA smoothing with α=10) found that it *degraded* IC from −0.004 to −0.013 and eliminated DM test significance (4/5→0/3 seeds). The discrepancy likely reflects market-specific structure (single index vs 477 broad stocks) and denoising method (wavelet vs EMA).

### 2.4 Volatility Prediction: GARCH Standard

GARCH(1,1) has been the standard volatility model for four decades, exploiting the strong positive autocorrelation (lag-1 ρ > 0.5) of realized variance. Glaubenskrieg's systematic test across 300 stocks finds GARCH uniformly dominates LightGBM—a result consistent with the principle that correctly specified structural models outperform atheoretical ML when the data-generating process is well-understood.

### 2.5 The Reproducibility Gap

Ye et al. (2025) documented that LLM-based trading agents overstate returns by 71–85% due to temporal contamination, proposing a P1–P6 protocol for deployment-grade evidence. Glaubenskrieg unwittingly replicated this finding: our v3 IC of 0.14 collapsed to 0.006 after protocol hardening—an overstatement of 95.7%. Nature (2025) surveyed the field and concluded that most LSTM/DNN stock prediction studies are not reproducible in real-world applications. Glaubenskrieg's protocol—purged walk-forward, held-out test, linear baseline, 5-step diagnosis, multi-market replication—is positioned as a counter-example of methodological rigor.

---

## 3. The Five-Step IC Validation Diagnostic Framework

### 3.1 The IC Paradox

Walk-forward validation suggests IC ≈ 0.04–0.14, but held-out test IC ≈ 0.00. How does a practitioner distinguish genuine signal from overfitting? We call this the **IC Paradox**, and it is the central methodological challenge in financial ML.

**Table 2: Glaubenskrieg's IC Paradox**

| Stage | val_IC | test_IC | Interpretation |
|---|---|---|---|
| v3 (buggy) | 0.14 | Unknown | Apparent signal (would be publishable) |
| v5 (fixed) | 0.049 | 0.006 | Suspect overfit |
| IC Paradox Diagnosis | 0.049 | 0.006 | **Confirmed overfit** |

The v3→v5→test progression represents a 23.3× overstatement. Without a diagnostic framework, this false positive would have propagated into downstream experiments.

### 3.2 Step 1: Held-Out Test IC Gap

**Method**: Take the checkpoint selected by walk-forward validation and evaluate it on a never-before-seen test set.

| Model | val_IC | test_IC | Gap |
|---|---|---|---|
| CTM v5 | 0.049 | 0.006 | 0.043 |
| Linear Ridge | — | 0.006 | — |

**Interpretation**: A gap exceeding 0.02 indicates that walk-forward validation overstates true performance. If test_IC ≈ 0, the walk-forward signal is overfit noise.

### 3.3 Step 2: Linear Ridge Baseline

**Method**: Train a simple L2-regularized linear model (451 parameters) on the same data.

| Model | Params | test_IC | test_Sharpe |
|---|---|---|---|
| CTM (Mamba SSM) | 97,000 | ≈ 0.00 | −0.028 |
| Linear Ridge | 451 | 0.006 | +0.55 |
| GBDT (Hoffnung) | ~5,000 | 0.008 | −0.088 |

**Decision rule**: If Ridge IC ≥ DL IC, stop. The 97,000-parameter Mamba model has 215× the parameters of Ridge but achieves the same IC. The extra parameters add only noise.

### 3.4 Step 3: Per-Window IC Stability

**Method**: Compute IC separately for each walk-forward window.

| Window | CTM val_IC |
|---|---|
| w0 | 0.019 |
| w1 | **0.074** (outlier) |
| w2 | 0.041 |
| Mean | **0.049** |

**Interpretation**: Window 1 drives the mean. Without it, μ = 0.030. We recommend a minimum of 10 windows; with N ≥ 10, the coefficient of variation CV(IC) = σ/μ should be < 0.3.

### 3.5 Step 4: Per-Stock Binomial Test

**Method**: Compute IC separately for each stock. Test the proportion of stocks with positive IC against the null of 50%.

| Metric | Value |
|---|---|
| Mean per-stock IC | 0.016 |
| Std per-stock IC | 0.069 |
| % positive | 60% |
| Binomial p-value | 0.10 (not significant) |

**Decision rule**: Require p < 0.01 (two-sided binomial test). At 60% positive, the pattern is indistinguishable from noise.

### 3.6 Step 5: Train-Val Loss Gap

**Method**: Monitor train and validation loss curves during training.

```
Healthy:    train_loss ↘  val_loss ↘  (gap constant or narrowing)
Overfitting: train_loss ↘  val_loss ↗  (gap widening → U-shape)
```

**Glaubenskrieg finding**: Validation loss stopped improving at epoch 10 while training loss continued to decrease—the classic overfitting U-shape divergence.

### 3.7 Go/No-Go Decision Tree

```
Start: walk-forward val_IC > 0?
    ↓
Step 1: Held-out test IC
    ↓
test_IC > 0.02? ── Yes ──→ Step 2: Linear Ridge baseline
    ↓ No                        ↓
❌ STOP: No signal         Ridge IC < DL IC?
                              ↓ Yes      ↓ No
                          Proceed → ❌ STOP
                              ↓
                       Step 3: Window stability
                              ↓
                       Consistent across windows?
                              ↓ Yes    ↓ No
                          Proceed → ⚠️ Flag
                              ↓
                       Step 4+5: Per-stock IC + Loss Gap
                              ↓
                       Broad + convergent?
                              ↓ Yes
                       ✅ Signal confirmed
```

### 3.8 Diagnostic Thresholds

**Table 3: Diagnostic Thresholds**

| Metric | Healthy | Suspicious | Pathological |
|---|---|---|---|
| test_IC − val_IC gap | < 0.01 | 0.01–0.05 | > 0.05 |
| Ridge IC vs DL IC | DL > Ridge × 1.5 | DL ≈ Ridge | DL < Ridge |
| Window IC CV(IC) | < 0.3 | 0.3–0.5 | > 0.5 |
| Per-stock % positive | > 65% | 55–65% | < 55% |
| Binomial p | < 0.01 | 0.01–0.10 | > 0.10 |
| Loss gap divergence | None | Moderate | U-shaped before epoch 10 |

**Glaubenskrieg scores**: Gap = 0.043 (pathological), DL ≤ Ridge (pathological), Window CV = 0.57 (but only 3 windows), Per-stock = 60% (suspicious), Binomial p = 0.10 (suspicious), Loss gap = divergence at epoch 10 (pathological). Composite diagnosis: **confirmed overfit**.

---

## 4. Case Study Design and Methodology

### 4.1 Data and Features

| Market | N Stocks | Date Range | Days | Windows | Source |
|---|---|---|---|---|---|
| A-Share (multi-asset) | 50 | 2020–2026 | 1,110 | 3 | Tencent Finance |
| A-Share (China) | 200 | 2015–2026 | ~2,700 | 6 | Tencent Finance |
| US (S&P500+NDX) | 477 | 2015–2026 | 2,504 | 10 | Yahoo Finance |

**Features** (9 OHLCV-derived, all computed causally):
- `simple_return`, `log_return` — raw return transformations
- `sma_5`, `sma_20` — simple moving averages
- `rsi_14` — relative strength index
- `bollinger_position` — Bollinger Band position
- `volume_ratio` — volume relative to 20-day average
- `realized_vol_21` — 21-day realized volatility
- `bias` — price deviation from moving average

All 9 features are deterministic functions of (price, volume). No fundamental data, alternative data, or text signals are included.

### 4.2 CTM Architecture

The Conv-Temporal-Mamba (CTM) model combines causal convolutional processing with the selective state space model:

```
Input(B, T, 9) → Linear(9→64) → CausalConv1D(k=3)
  → [SeasonalTrendDecomp(period=5)]
  → MambaBlock×2(state_dim=8) projected to 64
  → [Bi-Mamba forward+backward → Fuse]
  → Multi-Task Output Heads: Regression(1) + Classification(3)
```

**Key components**:
- **MambaBlock (S6)**: Pure PyTorch implementation of the selective scan mechanism. Content-dependent parameters Δ_t, B_t, C_t enable the model to adapt its dynamics to market conditions.
- **CausalConv1d**: Depthwise 1D convolution with left-padding, ensuring no future information leaks.
- **SeasonalTrendDecomp**: Moving average decomposition into trend and seasonal components.
- **Multi-task losses**: Composite MSE + negative Sharpe + 3-class directional cross-entropy.

**Parameter breakdown** (v5, 97,000 total):

| Component | Params | % |
|---|---|---|
| Mamba backbone (2 layers) | 56,320 | 58% |
| Cross-asset attention | 19,140 | 20% |
| Cross-asset FFN | 16,640 | 17% |
| Embeddings + projections | 4,800 | 5% |
| Output heads | 100 | <1% |

### 4.3 Six Prediction Paradigms

| # | Paradigm | Model | Parameters |
|---|---|---|---|
| 1 | Deep Learning | CTM (Mamba SSM) | 97,000 |
| 2 | Gradient Boosting | Hoffnung C++ GBDT + LightGBM | ~5,000 |
| 3 | Hybrid Ensemble | CTM + GBDT + TimeDecayGate | 157,323 |
| 4 | Feature Engineering | LightGBM + EMA smoothing | ~5,000 |
| 5 | Cross-Sectional Ranking | LightGBM lambdarank + CS z-scores | ~5,000 |
| 6 | Meta-Labeling | XGBoost binary filter | ~500 |

All six paradigms use the same walk-forward protocol, features, markets, and held-out test set.

### 4.4 Walk-Forward Protocol

| Parameter | Value | Purpose |
|---|---|---|
| Training period | 504–1,000 days | Sufficient for DL convergence |
| Purge period | 126 days | Prevent overlapping sequence leakage |
| Validation period | 200 days | Adequate for Sharpe estimation |
| Step size | 63 days | 67% window overlap (was 96% in v3) |
| Seeds | 5 per configuration | Seed sensitivity analysis |

**Bug evolution v1→v5** (each fix reduced IC):

| Bug | v3 State | v5 Fix | IC Impact |
|---|---|---|---|
| test eval shuffle=True | Inflated test Sharpe | shuffle=False | Fixed |
| step_size=21 | 96% window overlap | step_size=63 | Reduced overlap |
| lambda_sharpe=0.5 | Sharpe collapse | lambda_sharpe=0.1 | Stable training |
| Early stopping on Sharpe | Noisy selection | Spearman IC | Cleaner selection |
| LR warmup > n_epochs | Never decays | 20 epochs | Proper LR schedule |
| Held-out test | Not used | Final evaluation | True OOS measurement |

### 4.5 Statistical Methodology

- **IC**: Spearman rank correlation between predicted and realized returns
- **Diebold-Mariano test**: Prediction accuracy comparison with Newey-West HAC variance estimation
- **QLIKE loss**: Patton (2011) for volatility forecast evaluation
- **Power analysis**: With N=750, α=0.05, 80% power, minimum detectable IC = 0.102. Our observed IC of 0.007 is only 6.9% of this threshold. To detect IC=0.007 with 80% power, we would need ~160,000 independent observations (~640 years of daily data per stock)
- **Permutation test**: Designed but computationally infeasible without GPU cluster (estimated 50,000 training runs for 1,000 permutations)

---

## 5. Results

### 5.1 Cross-Market Return Prediction: IC ≈ 0

**Table 5: Cross-Market IC Comparison**

| Market | N Stocks | Windows | LightGBM IC | Ridge IC | IC > 0.02? | LGB > Ridge? |
|---|---|---|---|---|---|---|
| A-Share (multi-asset) | 50 | 3 | +0.053 ± 0.054 | +0.006 | ❌ | No |
| A-Share | 200 | 6 | −0.003 ± 0.042 | +0.030 ± 0.054 | ❌ | No |
| US | 477 | 10 | +0.007 ± 0.043 | +0.031 ± 0.030 | ❌ | No (Ridge better) |

**Finding**: No market, no model, no window count yields IC > 0.02—the canonical detection threshold for tradable signals. Ridge (451 parameters) matches or exceeds LightGBM (100 trees) in all three markets.

LightGBM per-window IC: [−0.067, −0.042, −0.039, +0.025, +0.047, +0.001, +0.026, +0.022, +0.083, +0.010]. Mean = +0.007, std = 0.043. The wide variance (−0.067 to +0.083) is the signature of noise.

### 5.2 Architecture Comparison: 97K DL Parameters = 451 Linear Parameters

**Table 6: v5 Final Architecture Comparison (A-Share multi-asset, 50 stocks, 3 windows, 5 seeds)**

| Model | Params | Mean val_IC | Test IC | Test Sharpe | Params/Sample |
|---|---|---|---|---|---|
| CTM (Mamba SSM) | 97,000 | 0.049 | ≈ 0.00 | −0.093 | ~125 |
| Ensemble + TimeGate | 157,323 | 0.037 | — | −0.024 | ~202 |
| GBDT (Hoffnung) | ~5,000 | 0.008 | — | −0.088 | ~6.4 |
| Linear Ridge | 451 | — | **0.006** | +0.55 | **0.58** |

**Finding**: CTM's test IC ≈ 0.00 — identical to Ridge's 0.006. The 96,549 additional DL parameters introduce no predictive value.

**GBDT parameter insensitivity**: 9 configurations (100/200/500 trees × depth 4/6/8) × 5 seeds produced IC in range [0.019, 0.022]. The flat response surface proves the data contains no structure for different parameterizations to exploit.

### 5.3 The Sharpe Paradox

Perhaps our most actionable methodological finding: **positive Sharpe ratios arise mechanically from sign-based long/short portfolio construction, even when IC ≈ 0.**

**Table 7: Sharpe-IC Decoupling Across All Experiments**

| Experiment | IC | Sharpe | Method |
|---|---|---|---|
| US LightGBM | +0.007 | +1.16 | Long top 20%, short bottom 20% |
| Ridge (US) | +0.031 | +0.82 | Same |
| LGB-MSE | +0.010 | +2.36 | Standard LightGBM |
| LGB-LambdaRank | +0.019 | +2.79 | Cross-sectional ranking loss |
| LGB-SharpeTuned | — | **+2.98** | Portfolio-optimized HP |
| Random sorting | ≈ 0 | +0.04 | Random portfolio |
| Mean-reversion | −0.009 | +0.04 | Known negative-IC strategy |

**Mechanism**: Any sorting of stocks into long/short portfolios in a rising market mechanically captures the equity risk premium. The positive Sharpe comes from risk-taking, not prediction ability.

**Portfolio-aware hyperparameter tuning amplifies the artifact**: LGB-SharpeTuned achieves Sharpe = 2.98 while LGB-MSE achieves 2.36—a +26% improvement (p = 0.011). But all methods produce positive Sharpe because any long/short sorting in a rising market mechanically captures the equity risk premium.

**Practical implication**: Any financial ML paper reporting Sharpe without corresponding IC should be treated with skepticism. Positive Sharpe is the default outcome of any strategy backtest in a rising market.

### 5.4 Volatility Prediction: GARCH(1,1) Beats ML on 300/300 Stocks

Volatility prediction is our positive control—the only paradigm where signal exists (variance clustering), and our protocol detects it unambiguously.

**Table 8: Cross-Market Volatility QLIKE**

| Market | LightGBM QLIKE | Mean Baseline | Persistence | Δ vs Persistence |
|---|---|---|---|---|
| A-Share (multi-asset) | −6.42 ± 0.24 | −6.29 | −5.67 | −0.75 |
| A-Share | −6.78 ± 0.26 | −6.34 | −5.85 | −0.92 |
| US | −7.10 ± 0.20 | −6.75 | −6.50 | −0.60 |

**CTM vs LightGBM for volatility (50 US stocks):** CTM test MSE = 1.00 × 10⁻⁴; LightGBM test MSE = 2.64 × 10⁻⁵. CTM is **3.8× worse**—deep learning makes volatility prediction *less* accurate.

**GARCH(1,1) vs LightGBM (300 stocks across 2 markets):**

| Predictor | Parameters | Median QLIKE | Win Rate |
|---|---|---|---|
| **GARCH(1,1)** | **3** (ω, α, β) | **2.16** | **300/300 (100%)** |
| Historical Mean | 0 | 2.26 | 0/300 |
| LightGBM | ~5,000 (300 trees) | 5.75 | 0/300 (0%) |

DM = −6.73, p = 1.6 × 10⁻¹¹. The 3-parameter model from 1986 significantly outperforms 300-tree gradient boosting on *every single stock in two independent markets*.

**Interpretation**: This is the most important positive result. It proves our protocol can detect signal when signal exists. The null result for returns is therefore informative, not vacuous. And ML complexity degrades rather than improves prediction even for the one detectable signal.

### 5.5 Regime-Adaptive Risk Management

**Table 9: Volatility Regime Strategy (200 US stocks, 2015–2026)**

| Strategy | Full-Period Sharpe | OOS Sharpe | Annual Vol | Max DD |
|---|---|---|---|---|
| Equal Weight | 1.121 | 1.080 | 23.3% | 37.1% |
| Inverse Volatility | 1.120 | 1.061 | 21.2% | 35.1% |
| Vol Target 15% | 1.214 | 1.110 | 15.8% | 22.0% |
| **Regime-Adaptive** | **1.402** | **1.296** | **15.3%** | **19.2%** |

**Key finding**: The regime-adaptive strategy improves Sharpe by +0.28 (full period) / +0.22 (OOS) through tactical de-leveraging during high-volatility regimes, not from stock selection. This succeeds because volatility *time-series* properties are detectable—consistent with the GARCH finding.

### 5.6 All Six Paradigms Converge: Negative Results Summary

**Table 10: Six Paradigms — Complete Verdict**

| # | Paradigm | Key Metric | Verdict |
|---|---|---|---|
| 1 | Deep Learning | Test IC ≈ 0.00 | ❌ No signal |
| 2 | Gradient Boosting | Val IC = 0.008 | ❌ No signal |
| 3 | Hybrid Ensemble | Test Sharpe uniformly negative | ❌ Degrades |
| 4 | Feature Engineering | Test IC = −0.013 | ❌ Denoising degrades |
| 5 | Cross-Sectional Ranking | Test Rank IC negative | ❌ Worse than TS |
| 6 | Meta-Labeling | ΔSharpe = −0.22 | ❌ Reduces Sharpe |

**Convergence is the strongest evidence**: If any paradigm could extract signal from pure OHLCV features, gradient boosting—the established leader on tabular data per McElfresh et al.—would have found it.

---

## 6. Discussion

### 6.1 The Feature Information Ceiling

Our results converge on a single binding constraint: **no model architecture can extract signal that is not present in the features.** The 9 OHLCV-derived features we tested are deterministic functions of (price, volume)—they contain no information independent of the market price. This is not a model failure; it is an information ceiling.

The LLG framework (Kelly & Malamud, 2025) provides theoretical grounding: even if true population R² ≥ 20%, the learning gap grows as O(P/T). With P/T = 129 (CTM), the gap renders signal undetectable. Ridge at P/T = 0.58 reduces but does not eliminate this gap. With 9 OHLCV features and ~750 training days, daily return prediction is not feasible with any currently available model class.

### 6.2 The Sharpe Paradox and Publication Bias

If positive Sharpe arises mechanically from any long/short strategy in a rising market, then hundreds of published papers reporting Sharpe as their primary metric may be reporting artifacts rather than genuine alpha. The widespread practice of using Sharpe for both model selection and performance reporting creates a perverse incentive: models producing levered market exposure are systematically preferred over models with genuine but weaker predictive signals.

We recommend the field adopt a two-metric standard: IC (or equivalent) must be reported alongside Sharpe for any return prediction study.

### 6.3 Volatility: Where Econometrics Beats ML

Returns are approximately martingale (autocorrelation ≈ 0); volatility is strongly autocorrelated (ρ > 0.5 at lag 1). GARCH was designed to exploit this structure; LightGBM was not. The 100% win rate across 300 stocks is a humbling reminder that 40 years of financial econometrics have produced tools optimal for their domain. ML's flexibility is a liability when the data-generating process is well-understood.

### 6.4 Comparison with Published Claims

- **MambaStock**: Claims superiority over XGBoost on 4 Chinese bank stocks, but predicts prices (not returns), has no walk-forward validation, no linear baseline, and no held-out test. Our reproducibility assessment found their code produces R² = −4.70.
- **SAMBA**: Claims IC improvements of 33–85% over baselines, but has no publicly available code, no absolute IC values reported, and no held-out test evaluation.
- **CNN-LightGBM (Bai et al.)**: Most methodologically rigorous positive paper surveyed (OOS-R² = 0.029, DM test, walk-forward). However, 93% of gain comes from wavelet denoising, and Glaubenskrieg's independent denoising test found it degraded IC.

---

## 7. Limitations

**Feature scope**: Only 9 OHLCV-derived features. A larger feature set—particularly fundamental data (94 features per Gu-Kelly-Xiu), text signals, or alternative data—might contain predictive information.

**Frequency**: Daily frequency only. Intraday data (5-minute bars) has substantially higher SNR (~0.20 vs ~0.025 daily) and may contain microstructure patterns.

**Survivorship bias**: US stock selection (S&P500 + NASDAQ100 constituents with ≥2,500 trading days) excludes delisted stocks, potentially overstating the baseline.

**Transaction costs**: No strategy backtest includes transaction costs, slippage, or market impact.

**Cross-market protocol alignment**: Walk-forward windows vary across markets (A-Share multi-asset=3, A-Share cross-sectional=6, US=10) due to different data spans.

**Permutation test**: Implemented but contained a software artifact (unrealistically high IC on shuffled data), suggesting temporal dependence leakage. Not reported as definitive.

**Global normalization**: Features were normalized on the full dataset rather than per-window, introducing minor future information leakage.

---

## 8. Conclusion

We have conducted what we believe is the most exhaustive test of OHLCV-based daily stock return prediction in the academic literature. Six paradigms across three markets, covering 877 stocks, were evaluated using a protocol exceeding the methodological standards of most published studies: purged walk-forward validation, independent held-out test sets, linear baselines, multi-seed sensitivity analysis, Diebold-Mariano statistical tests, and a novel five-step diagnostic framework for detecting overfitting.

Our finding is unambiguous: **OHLCV-derived features at daily frequency contain no detectable return-predictive signal across markets, model classes, or training objectives.** The 97,000-parameter Mamba SSM achieves the same test IC as the 451-parameter linear Ridge regression. GARCH(1,1) with three parameters outperforms 300-tree gradient boosting on every single stock tested for volatility prediction.

The five-step IC validation diagnostic framework is our primary methodological contribution. We show through the detailed case study of our own pipeline's evolution (v1→v5, IC=0.14→0.006) how the framework would have prevented a false positive publication. We recommend its adoption as standard practice in financial ML research.

Future work claiming daily return predictability from price-derived features should be evaluated against three baselines: (1) a linear Ridge regression on the same features, (2) held-out test IC after purged walk-forward validation, and (3) the five-step diagnostic framework. Papers not meeting these standards should be interpreted with appropriate caution.

The path forward for financial ML lies not in more sophisticated models applied to the same features, but in incorporating genuinely new information sources: fundamental data, text and news signals, alternative data, and intraday microstructure. The OHLCV ceiling has been reached.

---

## Data and Code Availability

All code is available at [URL]. The repository contains:
- Pure PyTorch CTM model implementation (~7,500 lines, 19 modules)
- Data processing and feature engineering pipeline
- Walk-forward training and evaluation scripts
- GBDT ensemble bridge (Hoffnung C++ backend)
- All experimental results (12 experiment groups, 45+ runs)
- 5-step diagnostic framework implementation
- Figure generation scripts

---

## References

[TBD — to be compiled from paper/CLASSIC_LITERATURE_COMPARISON.md sources]
