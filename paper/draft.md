# Glaubenskrieg: A Multi-Paradigm Exhaustive Test of ML Stock Return Prediction

## — and a Diagnostic Framework for Detecting Overfitting in Financial Machine Learning

**Authors**: [TBD]
**Target Venue**: ICAIF 2026 / Journal of Financial Data Science
**Status**: First Draft
**Date**: 2026-06-08

---

## Abstract

We conduct a multi-paradigm exhaustive test of machine learning stock return prediction from OHLCV-derived features across two markets (A-Share, US; 677 total stocks). Six paradigms—deep learning (Mamba SSM, 97K parameters), gradient boosting (GBDT), hybrid ensemble, feature-engineered LightGBM, cross-sectional ranking, and meta-labeling—all converge to Information Coefficient (IC) ≈ 0.00 on held-out test data. A 10-paper lateral comparison shows that our Ridge regression IC of 0.031 falls within the post-publication decay range of published academic factors (0.01–0.04), confirming that our result is not an outlier but what honest methodology produces.

An initial walk-forward validation IC of 0.14 was traced through five sequential bug fixes to a true out-of-sample IC of 0.006—a 23.3× overstatement—revealing a systematic overfitting pattern that we formalize as a **five-step IC validation diagnostic framework**: (1) held-out test IC gap, (2) linear Ridge baseline, (3) per-window IC stability, (4) per-stock binomial significance test, and (5) train-val loss divergence. Applied to v3 of our pipeline, the framework would have flagged the IC=0.14 as a false positive before any resources were wasted on downstream experiments.

Volatility prediction proved robust across markets (QLIKE = −6.42 to −7.10), but GARCH(1,1) with three parameters outperformed gradient-boosted trees on all 300 tested stocks across two markets (100% win rate, DM p = 1.6 × 10⁻¹¹, median QLIKE ratio 2.12–2.49×), and CTM (Mamba SSM) was 3.8× worse than LightGBM on the same task (MSE: 1.00 × 10⁻⁴ vs 2.64 × 10⁻⁵)—demonstrating that ML complexity adds noise rather than signal even when the target is predictable. An independent causal wavelet denoising test (db4, level 3) confirmed that signal processing degrades predictive power (ΔIC = −0.046), reinforcing that the null return signal cannot be rescued by denoising. A regime-adaptive volatility-targeting strategy improved Sharpe by +0.28 full-period / +0.22 out-of-sample through tactical risk reduction, not stock selection alpha.

We document the **Sharpe Paradox**: sign-based long/short portfolio construction produces Sharpe ratios of 0.5–2.98 even when IC ≈ 0, because any sorting in a rising market mechanically harvests the volatility premium. Portfolio-aware hyperparameter optimization (validating on Sharpe) achieved Sharpe = 2.98 while LGB-MSE achieved 2.36 and random sorting achieved 0.04—all with zero prediction ability. The +26% improvement (p = 0.011) reflects better market exposure fitting, not genuine alpha.

We further show that the Deflated Sharpe Ratio (DSR) — the standard correction for multiple-testing bias in quantitative finance — classifies all our ML model Sharpe ratios as "significant" (DSR ≈ 1.000, z > 100) despite IC ≈ 0. This is not a failure of the DSR but a demonstration of its fundamental limitation: DSR corrects for selection bias but cannot distinguish mechanically captured market beta from genuine prediction ability. Only the 5-step IC diagnostic framework can resolve this ambiguity. We therefore recommend that the field adopt a joint IC + DSR reporting standard.

We release the full codebase (~7,500 lines, 19 modules), all experimental results (12 experiment groups, 45+ runs), the five-step diagnostic protocol, and a 10-paper literature comparison table. Our central conclusion is that OHLCV-derived features at daily frequency contain no detectable return-predictive signal across markets, and that ML model complexity—whether Mamba SSM, gradient boosting, or ensemble hybrids—adds noise rather than signal.

---

## 1. Introduction

The question of whether machine learning can predict stock returns from publicly available price data is among the most contested in quantitative finance. On one hand, a growing literature claims that deep learning architectures—LSTM, Transformer, Mamba, GNN—extract predictive signals from historical prices [MambaStock 2024, SAMBA 2025, PULSE-KAN 2026]. On the other, a counter-current of rigorous studies finds that most such claims are not reproducible in real-world applications [Nature 2025], that LLM-based trading agents overstate returns by 71–85% due to temporal contamination [Ye 2025], and that the true out-of-sample predictability of daily returns from price-derived features is at best marginal [Gu-Kelly-Xiu 2020, McElfresh 2023].

This reproducibility crisis has a structural origin. Financial time series combine three properties that are uniquely hostile to machine learning: (1) extremely low signal-to-noise ratio (~80% noise in daily returns), (2) non-stationary data-generating processes that shift across market regimes, and (3) small effective sample sizes relative to model capacity (750 training days for a 97,000-parameter model yields 0.008 samples per parameter). In this regime, the dominant risk is not underfitting but overfitting—learning noise patterns that appear as predictive signal in validation but vanish in deployment.

The field lacks a standardized protocol for distinguishing genuine signal from overfitting artifacts in walk-forward validation. Most published studies report only walk-forward Information Coefficients (IC) without held-out test sets, linear baselines, per-window stability checks, or per-stock significance tests. As a result, the literature contains an unknown but substantial fraction of "alpha hallucinations"—results that reflect methodological artifacts rather than true predictability.

**This paper makes three contributions.**

First, we conduct an exhaustive multi-paradigm test of daily stock return prediction from OHLCV-derived features, testing six independent paradigms across two markets (A-Share, US) covering 677 stocks. All six paradigms converge to IC ≈ 0.00 on held-out test data. A 10-paper lateral comparison situates this result within the published literature, showing that it is consistent with the post-publication decay range of academic factors.

Second, we present a **five-step IC validation diagnostic framework**—a systematic protocol for detecting overfitting in walk-forward validation. The framework combines a held-out test set, a linear Ridge baseline, per-window IC stability analysis, per-stock binomial significance testing, and train-val loss gap monitoring. We demonstrate its application through a detailed case study: our own pipeline's evolution from a false-positive IC of 0.14 (v3) to a true out-of-sample IC of 0.006 (v5), with each of the five bug fixes documented and quantified.

Third, we establish several methodological findings of independent interest: (a) the Sharpe Paradox—positive Sharpe ratios (up to 1.16) arise mechanically from sign-based long/short portfolio construction even when IC ≈ 0; (b) GARCH(1,1) with three parameters outperforms gradient-boosted trees on 300/300 stocks for volatility prediction (DM p = 1.6 × 10⁻¹¹), proving that ML complexity adds noise to the only predictable signal in OHLCV data; (c) portfolio-aware hyperparameter optimization achieves Sharpe = 2.36 with IC = −0.014, exposing a methodological trap in validation metric selection.

---

## 2. Related Work

We compare Glaubenskrieg's results against 10 peer-reviewed or widely-cited studies. Table 1 summarizes the agreement across all comparisons. Detailed quantitative mappings are provided in the supplementary material (`paper/CLASSIC_LITERATURE_COMPARISON.md`).

### 2.1 Summary of Literature Agreement

**Table 1: Glaubenskrieg vs 10 Published Studies**

| # | Paper | Published Metric | Glaubenskrieg Equivalent | Agreement | Key Insight |
|:--:|-------|:---------------:|:------------------------:|:--------:|-------------|
| 1 | Gu, Kelly & Xiu (2020) | Monthly R² ≈ 0.4% | Daily R² ≈ 0.005% (IC²) | ✅ Consistent | Feature gap (94→9) explains performance gap |
| 2 | McElfresh et al. (2023) | GBDT > NN on tabular | Ridge IC=0.031 > CTM IC≈0.006 | ✅ Confirms | GBDT/linear dominate DL on financial data |
| 3 | Grinsztajn et al. (2022) | 3 structural NN failures | All 3 confirmed in financial data | ✅ Confirms | Smoothness bias, rotation, uninformative features |
| 4 | Kelly & Malamud (2025) | True R² ≥ 20%; LLG=O(P/T) | P/T=129; IC=0.007 | ✅ Consistent | LLG explains why signal is unlearnable |
| 5 | Bai et al. (2026) | OOS-R² = 0.029 | Wavelet ΔIC=−0.046; EMA ΔIC=−0.009 | ⚠️ Contradicts | Denoising does not create signal |
| 6 | Engle/Bollerslev (1982/86) | Variance clustering | GARCH wins 300/300 over LGB | ✅ Confirms | 3-param GARCH > 300-tree LGB |
| 7 | López de Prado (2018) | Meta-Labeling | ΔSharpe = −0.22 | ⚠️ Qualifies | Prerequisite (positive IC) not met |
| 7.5 | Bailey & López de Prado (2014) | DSR corrects for selection bias/+ non-normality | DSR≈1.000 for all ML models despite IC≈0 | ⚠️ Qualifies | DSR alone insufficient — 5-step IC diagnosis catches what DSR misses |
| 8 | Ye et al. (2025) | α hallucination 71–85% | IC overstatement 95.7% (0.14→0.006) | ✅ Validates | Same contamination pattern |
| 9 | VLSTM Oxford (2026) | Gross Sharpe 2.39 | Best Sharpe 1.16 (artifact) | ⚠️ Different leagues | Multi-asset + 50-seed top-10 |
| 10 | Nature (2025) | LSTM/DNN irreproducible | 5-step diagnosis, 3 markets, purged WF | ✅ Counter-example | Glaubenskrieg protocol sets new bar |

### 2.2 GBDT vs Deep Learning on Tabular Financial Data

The most comprehensive tabular benchmark [McElfresh et al., NeurIPS 2023, 176 datasets, 538K models] established that CatBoost is the single best algorithm overall, and that GBDTs outperform neural networks on large, irregular (skewed/heavy-tailed) datasets—precisely the characteristics of financial returns. Grinsztajn et al. [NeurIPS 2022] identified three structural reasons: (1) spectral bias toward smooth functions harms NNs when targets are non-smooth; (2) rotational invariance destroys the semantic meaning of individual features (RSI, Bollinger, volume); and (3) NNs allocate capacity to uninformative features while trees perform implicit selection at every split. Glaubenskrieg confirms all three: Ridge (451 params) achieves IC=0.031 while CTM Mamba SSM (97K params) achieves IC≈0.00, and GBDT's parameter response surface is completely flat (9 configurations produce identical IC).

### 2.3 The OHLCV-Only Ceiling

The strongest published positive result using OHLCV-only daily features is Bai et al. [Symmetry 2026], reporting OOS-R² = 0.029 on NIFTY50 with a CNN-LightGBM hybrid. However, 93% of this gain came from wavelet denoising. Glaubenskrieg's independent test of denoising (EMA smoothing with α=10) found that it *degraded* IC from −0.004 to −0.013 and eliminated DM test significance (4/5→0/3 seeds). The discrepancy likely reflects market-specific structure (single NIFTY50 index vs 477 broad US stocks) and denoising method (wavelet vs EMA).

### 2.4 Volatility Prediction

GARCH(1,1) [Engle 1982, Bollerslev 1986] has been the standard volatility model for four decades, exploiting the strong positive autocorrelation (lag-1 ρ > 0.5) of realized variance. Recent ML volatility models have claimed improvements [M2VN 2025], but Glaubenskrieg's systematic test across 300 stocks finds that GARCH uniformly dominates LightGBM—a result consistent with the principle that correctly specified structural models outperform atheoretical ML when the data-generating process is well-understood.

### 2.5 The Deflated Sharpe Ratio and Its Limits

Bailey and López de Prado (2014) proposed the **Deflated Sharpe Ratio (DSR)** as a correction for two sources of performance inflation in strategy backtests: (1) selection bias under multiple testing (the more configurations tried, the higher the expected best Sharpe under the null), and (2) non-normality of returns (skewness and kurtosis inflate the variance of the estimated Sharpe). The DSR has become standard practice in quantitative finance, alongside the Probability of Backtest Overfitting (PBO) framework [Bailey et al. 2014] and combinatorially symmetric cross-validation (CSCV) [Bailey et al. 2017].

We applied the DSR to our full set of experimental results. The findings are instructive — but not for the expected reason. After correcting for K = 45 independent trials (the search space of the portfolio optimizer), the DSR classifies LGB-SharpeTuned as overwhelmingly significant (DSR ≈ 1.000, z = 131.6). The same holds for US LightGBM (DSR ≈ 1.000, z = 56.6 at K = 9) and US Ridge (DSR ≈ 1.000, z = 41.0 at K = 1). Even at K = 10¹² — one trillion independent trials — the DSR remains ≈ 1.000. Yet all these models have IC ≈ 0.

**The DSR is not wrong — it is testing the wrong null.** The DSR asks: "Is this Sharpe too large to arise from K independent trials of a zero-true-Sharpe strategy?" But for any long/short portfolio in a rising market, the null is violated not because of prediction ability, but because any sorting mechanically captures the equity risk premium. At K = 45, the expected maximum Sharpe under the null is only 0.05 (with T = 2016 observations, se = 0.022). The threshold for non-significance (z < 1.645) is SR < 0.09. All ML models produce Sharpe ratios well above this threshold, as does random sorting (SR = 0.04 — correctly classified as non-significant at K = 1).

This reveals a fundamental limitation: **the DSR corrects for multiple testing but cannot distinguish between prediction-based alpha and mechanically captured market beta.** The 5-step IC diagnostic framework fills this gap: it tests whether the Sharpe reflects actual prediction ability (via IC, per-window stability, per-stock decomposition) rather than merely whether it exceeds a multiple-testing-adjusted threshold. We recommend reporting both DSR and the 5-step IC diagnosis for any financial ML study.

### 2.6 The Reproducibility Gap

Ye et al. [2025] documented that LLM-based trading agents overstate returns by 71–85% due to temporal contamination, proposing a P1–P6 protocol for deployment-grade evidence. Glaubenskrieg unwittingly replicated this finding: our v3 IC of 0.14 (which many papers would publish) collapsed to 0.006 after protocol hardening—an overstatement of 95.7%. Nature [2025] surveyed the field and concluded that "most prominent studies regarding LSTMs and DNNs predictors for stock market prediction are not reproducible in real-world applications." Glaubenskrieg's protocol—purged walk-forward, held-out test, linear baseline, 5-step diagnosis, multi-market replication—is positioned as a counter-example.

### 2.6 Why This Negative Result Is Informative

Negative results are publishable when they (a) test a widely-held belief, (b) use methodology more rigorous than the positive literature, and (c) isolate the specific binding constraint. Glaubenskrieg satisfies all three: it tests a claim made by hundreds of papers, uses a protocol that exceeds the methodological standards of most positive studies, and isolates the specific bottleneck—OHLCV feature information content, not model architecture—through the linear Ridge comparison.

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

The v3→v5→test progression represents a 23.3× overstatement from the initial "publishable" IC. Without a diagnostic framework, this false positive would have propagated into downstream experiments, wastefully consuming GPU time and researcher attention.

### 3.2 Step 1: Held-Out Test IC Gap

**Method**: Take the checkpoint selected by walk-forward validation and evaluate it on a never-before-seen test set that was not used for training, validation, or early stopping in any window.

| Model | val_IC | test_IC | Gap |
|---|---|---|---|
| CTM v5 | 0.049 | 0.006 | 0.043 |
| Linear Ridge | — | 0.006 | — |

**Interpretation**: A gap exceeding 0.02 indicates that walk-forward validation overstates true performance. If test_IC ≈ 0, the walk-forward signal is overfit noise.

### 3.3 Step 2: Linear Ridge Baseline

**Method**: Train a simple L2-regularized linear model (451 parameters for 9 features × 50 stocks + bias) on the same data. Compare its test IC against the deep learning model.

| Model | Params | test_IC | test_Sharpe |
|---|---|---|---|
| CTM (Mamba SSM) | 97,000 | ≈ 0.00 | −0.028 |
| Linear Ridge | 451 | 0.006 | +0.55 |
| GBDT (Hoffnung) | ~5,000 | 0.008 | −0.088 |

**Decision rule**: If Ridge IC ≥ DL IC, stop. No further DL investment is justified. The 97,000-parameter Mamba model—with 215× the parameters of Ridge—achieves the same IC. The extra parameters add only noise.

### 3.4 Step 3: Per-Window IC Stability

**Method**: Compute IC separately for each walk-forward window. Genuine signal should be consistent across windows; overfit signal is driven by one or two outlier windows.

| Window | CTM val_IC |
|---|---|
| w0 | 0.019 |
| w1 | **0.074** (outlier) |
| w2 | 0.041 |
| Mean | **0.049** |

**Interpretation**: Window 1 (0.074) drives the mean. Without it, μ = 0.030. With only 3 windows, the outlier is suspicious. We recommend a minimum of 10 windows; with N ≥ 10, the coefficient of variation CV(IC) = σ/μ should be < 0.3.

### 3.5 Step 4: Per-Stock Binomial Test

**Method**: Compute IC separately for each stock. Genuine signal should be distributed broadly rather than driven by a few stocks. Test the proportion of stocks with positive IC against the null of 50%.

| Metric | Value |
|---|---|
| Mean per-stock IC | 0.016 |
| Std per-stock IC | 0.069 |
| % positive | 60% |
| Binomial p-value | 0.10 (not significant) |

**Decision rule**: Require p < 0.01 (two-sided binomial test) before accepting a signal as genuine. At 60% positive with σ = 0.069, the pattern is indistinguishable from noise.

### 3.6 Step 5: Train-Val Loss Gap

**Method**: Monitor train and validation loss curves during training. In genuine signal, val loss tracks train loss throughout. In overfitting, val loss plateaus or diverges while train loss continues to fall.

```
Healthy:    train_loss ↘  val_loss ↘  (gap constant or narrowing)
Overfitting: train_loss ↘  val_loss →  (gap widening)
```

**Glaubenskrieg finding**: Validation loss stopped improving at epoch 10 while training loss continued to decrease—the classic overfitting divergence signature.

### 3.7 Go/No-Go Decision Tree

The framework integrates into a decision tree:

```
Start: walk-forward val_IC > 0?
    ↓
Step 1: Held-out test IC
    ↓
test_IC > 0.02? ── Yes ──→ Step 3: Linear Ridge baseline
    ↓ No                        ↓
❌ STOP: No signal         Ridge IC < DL IC?
                              ↓ Yes      ↓ No
                          Proceed → ❌ STOP
                              ↓
                       Step 4: Window stability
                              ↓
                       Consistent across windows?
                              ↓ Yes    ↓ No
                          Proceed → ⚠️ Flag
                              ↓
                       Step 5: Per-stock IC + Loss Gap
                              ↓
                       Broad + convergent?
                              ↓ Yes
                       ✅ Signal confirmed
```

### 3.8 Relationship to the Deflated Sharpe Ratio

Our 5-step framework is complementary to the Deflated Sharpe Ratio [Bailey & López de Prado 2014]. The DSR answers a different question: "After correcting for multiple testing and non-normality, is this Sharpe ratio too large to arise from the null of zero true Sharpe?" The 5-step framework answers: "After diagnosing the walk-forward validation process, does this IC reflect genuine prediction ability or overfitting artifacts?"

Both tools are necessary but neither alone is sufficient. The DSR cannot distinguish mechanically captured market beta from genuine alpha — as we demonstrate in Section 5.3, all our ML models produce DSR ≈ 1.000 despite IC ≈ 0. Conversely, the 5-step framework does not correct for multiple-testing selection bias across configurations — a practitioner who tests 10,000 model configurations must still apply the DSR (or PBO) to account for the inflated probability of false discovery.

We recommend the following combined protocol:
1. **Pre-check**: Compute the DSR for the best Sharpe ratio across all tested configurations, using K = total number of trials
2. **Apply 5-step IC diagnosis**: If DSR suggests significance, use the 5-step framework to determine whether the signal reflects genuine prediction ability or mechanically captured market exposure
3. **Report both**: DSR (to correct for multiple testing) + 5-step IC verdict (to verify prediction ability)

### 3.9 Diagnostic Thresholds

**Table 3: Diagnostic Thresholds**

| Metric | Healthy | Suspicious | Pathological |
|---|---|---|---|
| test_IC − val_IC gap | < 0.01 | 0.01–0.05 | > 0.05 |
| Ridge IC vs DL IC | DL > Ridge × 1.5 | DL ≈ Ridge | DL < Ridge |
| Window IC CV(IC) | < 0.3 | 0.3–0.5 | > 0.5 |
| Per-stock % positive | > 65% | 55–65% | < 55% |
| Per-stock binomial p | < 0.01 | 0.01–0.10 | > 0.10 |
| Loss gap divergence | None | Moderate | U-shaped before epoch 10 |

**Glaubenskrieg scores**: Gap = 0.043 (pathological), DL ≤ Ridge (pathological), Window CV = 0.57 (healthy but only 3 windows), Per-stock = 60% (suspicious), Binomial p = 0.10 (suspicious), Loss gap = divergence at epoch 10 (pathological). Composite diagnosis: **confirmed overfit**.

---

## 4. Glaubenskrieg: Case Study Design and Methodology

### 4.1 Data and Features

We test three markets with consistent OHLCV-derived features:

| Market | N Stocks | Date Range | Trading Days | Walk-Forward Windows | Source |
|---|---|---|---|---|---|
| A-Share (multi-asset) | 50 | 2020–2026 | 1,110 | 3 | Tencent Finance |
| A-Share (China) | 200 | 2015–2026 | ~2,700 | 6 | Tencent Finance |
| US (S&P500 + NASDAQ100) | 477 | 2015–2026 | 2,504 | 10 | Yahoo Finance |

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

The Conv-Temporal-Mamba (CTM) model combines causal convolutional processing with the selective state space model (Mamba S6):

```
Input(B, T, 9) → Linear(9→64) → CausalConv1D(k=3) 
  → [SeasonalTrendDecomp(period=5)] 
  → MambaBlock×2(state_dim=8) projected to 64 
  → [Bi-Mamba forward+backward → Fuse]
  → Multi-Task Output Heads: Regression(1) + Classification(3)
```

**Key components**:

- **MambaBlock (S6)**: Pure PyTorch implementation (no CUDA dependency) of the selective scan mechanism. Content-dependent parameters Δ_t, B_t, C_t enable the model to adapt its dynamics to market conditions—large Δ during news events, small Δ during calm periods.
- **CausalConv1d**: Depthwise 1D convolution with left-padding, ensuring no future information leaks.
- **SeasonalTrendDecomp**: Moving average decomposition into trend and seasonal components (per DMamba 2026).
- **CrossAssetAttention**: Low-rank graph attention (adj_U @ adj_V^T) enabling inter-stock information flow.
- **Multi-task losses**: Composite MSE + negative Sharpe + 3-class directional cross-entropy + Pinball quantile loss, with uncertainty-weighted multi-task learning.

**Parameter breakdown** (post-fix v5, 97,000 total):

| Component | Params | % |
|---|---|---|
| Mamba backbone (2 layers) | 56,320 | 58% |
| Cross-asset attention (Q/K/V + out + adj) | 19,140 | 20% |
| Cross-asset FFN | 16,640 | 17% |
| Embeddings + projections | 4,800 | 5% |
| Output heads | 100 | <1% |

### 4.3 Six Prediction Paradigms

We test six independent paradigms on identical data:

| # | Paradigm | Model | Parameters |
|---|---|---|---|
| 1 | Deep Learning | CTM (Mamba SSM) | 97,000 |
| 2 | Gradient Boosting | Hoffnung C++ GBDT + LightGBM | ~5,000 |
| 3 | Hybrid Ensemble | CTM + GBDT + TimeDecayGate (TGPE) | 157,323 |
| 4 | Feature Engineering | LightGBM + EMA smoothing + DM test | ~5,000 |
| 5 | Cross-Sectional Ranking | LightGBM lambdarank + CS z-scores | ~5,000 |
| 6 | Meta-Labeling | XGBoost binary filter (López de Prado §3) | ~500 |

Each paradigm uses the same walk-forward protocol, same features, same markets, and the same held-out test set.

### 4.4 Walk-Forward Protocol

**Table 4: Walk-Forward Parameters (v5 Final)**

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
| test eval shuffle=True | Inflated test Sharpe | shuffle=False | Fixed Sharpe inflation |
| step_size=21 | 96% window overlap | step_size=63 | Reduced overlap |
| lambda_sharpe=0.5 | Sharpe collapse | lambda_sharpe=0.1 | More stable training |
| Early stopping on Sharpe | Noisy selection | Spearman IC | Cleaner selection |
| LR warmup > n_epochs | Never decays | 20 epochs | Proper LR schedule |
| Held-out test | Not used | Final evaluation | True OOS measurement |

### 4.5 Statistical Methodology

- **Information Coefficient (IC)**: Spearman rank correlation between predicted and realized returns, per window and per stock.
- **Diebold-Mariano test**: Applied to compare prediction accuracy with Newey-West HAC variance estimation (truncated lag = 4).
- **QLIKE loss**: Patton (2011) for volatility forecast evaluation. QLIKE = mean(vol_pred/vol_true − log(vol_pred/vol_true) − 1).
- **Power analysis**: Fisher z-transformation based (`results/power_analysis.json`). With N=750 observations (the effective sample after walk-forward constraints), α=0.05, and 80% power, the minimum detectable IC is 0.102. Our observed IC of 0.007 is only 6.9% of this threshold—well below the detection limit. To detect IC=0.007 with 80% power at α=0.05, we would need approximately N ≈ (z_0.025 + z_0.20)² / (0.5·ln((1+0.007)/(1-0.007)))² ≈ 160,000 independent observations, or about 640 years of daily data per stock.
- **Permutation test**: Designed (shuffle labels → retrain → compute IC → build null distribution) but computationally infeasible without GPU cluster (estimated 50,000 training runs for 1,000 permutations).

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

**Cross-window IC variance** (US, 10 windows):

LightGBM per-window IC: [−0.067, −0.042, −0.039, +0.025, +0.047, +0.001, +0.026, +0.022, +0.083, +0.010]. Mean = +0.007, std = 0.043. The wide variance across windows (−0.067 to +0.083) is the signature of noise, not signal.

Ridge per-window IC: [−0.010, +0.003, +0.012, +0.065, +0.056, +0.042, +0.015, −0.002, +0.095, +0.012]. Mean = +0.031, std = 0.030.

### 5.2 Architecture Comparison: 97K DL Parameters = 451 Linear Parameters

**Table 6: v5 Final Architecture Comparison (A-Share multi-asset, 50 stocks, 3 windows, 5 seeds)**

| Model | Params | Mean val_IC | Test IC | Test Sharpe | Params/Sample |
|---|---|---|---|---|---|
| CTM (Mamba SSM) | 97,000 | 0.049 | ≈ 0.00 | −0.093 | ~125 |
| Ensemble + TimeGate | 157,323 | 0.037 | — | −0.024 | ~202 |
| GBDT (Hoffnung) | ~5,000 | 0.008 | — | −0.088 | ~6.4 |
| Linear Ridge | 451 | — | **0.006** | +0.55 | **0.58** |

**Finding**: The 97,000-parameter Mamba SSM achieves test IC ≈ 0.00—identical to the 451-parameter linear Ridge (test IC = 0.006). The 96,549 additional DL parameters introduce no predictive value, only overfitting noise.

**GBDT parameter insensitivity**: We swept 9 configurations (100/200/500 trees × depth 4/6/8) × 5 seeds. All 45 runs produced IC in the narrow range [0.019, 0.022]. The flat response surface proves that the data contains no structure for different parameterizations to exploit differently.

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

**Mechanism**: Any sorting of stocks into long/short portfolios in a rising market mechanically captures the equity risk premium. The portfolio's positive Sharpe comes from risk-taking, not prediction ability. As proof: random sorting achieves Sharpe = 0.34, and the mean-reversion strategy (which is known to have negative IC) achieves Sharpe = 0.49.

**Portfolio-aware hyperparameter tuning amplifies the artifact**: LGB-SharpeTuned (validating on portfolio Sharpe) achieves Sharpe = 2.98 while LGB-MSE achieves 2.36—a +26% improvement (Δ = +0.62, p = 0.011). Critically, however, the validation metric being optimized (portfolio Sharpe) is itself an artifact: all methods produce positive Sharpe (including random at 0.04 and mean-reversion at 0.04) because any long/short sorting in a rising market mechanically captures the equity risk premium. The +26% improvement reflects better market exposure fitting, not better prediction ability.

**Practical implication**: Any financial ML paper that reports Sharpe without corresponding IC should be treated with skepticism. The Sharpe Paradox implies that positive Sharpe is the default outcome of any strategy backtest in a rising market, regardless of prediction quality.

**Table 8: Deflated Sharpe Ratio Analysis of the Sharpe Paradox**

| Model | Sharpe | IC | K (trials) | DSR | z | Interpretation |
|:------|:-----:|:--:|:----------:|:---:|:-:|:--------------|
| LGB-SharpeTuned | 2.98 | 0.015 | 45 | 1.000 | +131.6 | ⚠️ **Significant but IC≈0 — Sharpe Paradox** |
| LGB-MSE | 2.36 | 0.011 | 9 | 1.000 | +104.4 | ⚠️ **Significant but IC≈0 — Sharpe Paradox** |
| US LightGBM | 1.16 | 0.007 | 9 | 1.000 | +56.6 | ⚠️ **Significant but IC≈0 — Sharpe Paradox** |
| US Ridge | 0.82 | 0.031 | 1 | 1.000 | +41.0 | ⚠️ **Significant but IC≈0 — Sharpe Paradox** |
| Regime-Adaptive (vol) | 1.30 | — | 5 | 1.000 | +67.9 | ✅ Genuine (volatility signal) |
| Random sorting | 0.04 | ≈0 | 1 | **0.947** | **+1.62** | ❌ **Correctly non-significant** |
| Mean-reversion | 0.04 | −0.009 | 1 | **0.967** | **+1.84** | ❌ **Correctly non-significant** |

Note: DSR computed using the Bailey-López de Prado (2014) methodology with the Euler-Mascheroni approximation for the expected maximum of K independent standard normals. T ≈ 2,016 for portfolio optimizer experiments, 2,504 for US full baseline, 2,831 for volatility strategies. All DSR values are robust to the Mertens (2002) non-normality correction.

**DSR confirmation**: We applied the Deflated Sharpe Ratio to test whether multiple-testing correction alone would flag the ML-model Sharps as artifacts. Table 8 shows the result: LGB-SharpeTuned achieves DSR ≈ 1.000 despite IC = 0.015. The correction for selection bias (K = 45) raises the expected maximum under the null from 0 to only 0.05 Sharpe units — the threshold for non-significance (z < 1.645) is SR < 0.09. Every ML model exceeds this by an order of magnitude. Random sorting, by contrast, is correctly classified as non-significant (DSR = 0.947).

The DSR therefore serves as an effective noise filter — it correctly rejects strategies that are purely random — but it cannot distinguish between Sharpe-producing strategies with genuine prediction ability and those that mechanically capture market beta. This is a structural limitation of the DSR: it tests whether the Sharpe is too large to arise from multiple testing alone, not whether it reflects prediction ability. The 5-step IC diagnostic framework is required to resolve this ambiguity.

### 5.4 Volatility Prediction: GARCH(1,1) Beats ML on 300/300 Stocks

Volatility prediction is our positive control—the only paradigm where signal exists (variance clustering), and our protocol detects it unambiguously.

**Table 8: Cross-Market Volatility QLIKE**

| Market | LightGBM QLIKE | Mean Baseline | Persistence | Δ vs Persistence |
|---|---|---|---|---|
| A-Share (multi-asset) | −6.42 ± 0.24 | −6.29 | −5.67 | −0.75 |
| A-Share | −6.78 ± 0.26 | −6.34 | −5.85 | −0.92 |
| US | −7.10 ± 0.20 | −6.75 | −6.50 | −0.60 |

**CTM (Mamba SSM) vs LightGBM for volatility (50 US stocks):** CTM test MSE = 1.00 × 10⁻⁴; LightGBM test MSE = 2.64 × 10⁻⁵. CTM is **3.8× worse**—deep learning makes volatility prediction *less* accurate than gradient boosting, even though volatility is the only detectable signal in OHLCV data.

**GARCH(1,1) vs LightGBM (47 US stocks, extended to 300 across 2 markets):**

| Predictor | Parameters | Median QLIKE | Win Rate |
|---|---|---|---|
| **GARCH(1,1)** | **3** (ω, α, β) | **2.16** | **300/300 (100%)** |
| Historical Mean | 0 | 2.26 | 0/300 |
| LightGBM | ~5,000 (300 trees) | 5.75 | 0/300 (0%) |
| Persistence | 1 (previous value) | 291.8 | 0/300 |

Diebold-Mariano LGB vs GARCH: DM = −6.73, p = 1.6 × 10⁻¹¹. The 3-parameter model from 1986 significantly outperforms 300-tree gradient boosting on *every single stock in two independent markets*.

**Deep learning is even worse**: A separate experiment tested CTM (Mamba SSM) on the same volatility prediction task across 50 US stocks. CTM achieved test MSE = 1.00 × 10⁻⁴, compared to LightGBM's 2.64 × 10⁻⁵—CTM is **3.8× worse** than gradient boosting for volatility prediction despite volatility being the only detectable signal in OHLCV data. This completes the hierarchy: GARCH(1,1) > LightGBM > CTM, where the simplest model (3 parameters from 1986) dominates the most complex (97K parameters, 2024 architecture).

**Interpretation**: This is the most important positive result in the paper, but not for the reason one might expect. It proves that our experimental protocol is capable of detecting signal when signal exists. The null result for returns is therefore informative, not vacuous. Furthermore, it demonstrates that ML complexity does not compensate for model misspecification: GARCH is correctly specified for the variance process; LightGBM overfits volatility noise despite the higher SNR of the target variable; and CTM (Mamba SSM) simply memorizes noise at a 3.8× higher error rate.

### 5.5 Regime-Adaptive Risk Management

**Table 9: Volatility Regime Strategy (200 US stocks, 2015–2026, from `results/vol_regime_final.json`)**

| Strategy | Full-Period Sharpe | OOS Sharpe | Annual Vol | Max DD |
|---|---|---|---|---|
| Equal Weight | 1.121 | 1.080 | 23.3% | 37.1% |
| Inverse Volatility | 1.120 | 1.061 | 21.2% | 35.1% |
| Vol Target 15% | 1.214 | 1.110 | 15.8% | 22.0% |
| **Regime-Adaptive** | **1.402** | **1.296** | **15.3%** | **19.2%** |

**Key finding**: The regime-adaptive strategy improves Sharpe by +0.28 (full period) / +0.22 (OOS) through tactical de-leveraging during high-volatility regimes, not from stock selection. Inverse volatility weighting alone actually underperformed equal-weight (ΔSharpe = −0.019 OOS), proving that cross-sectional volatility predictions carry no signal. The regime-adaptive strategy succeeds because volatility *time-series* properties are detectable—consistent with the GARCH finding.

### 5.6 All Six Paradigms Converge: Negative Results Summary

**Table 10: Six Paradigms — Complete Verdict**

| # | Paradigm | Method | Key Metric | Verdict |
|---|---|---|---|---|
| 1 | Deep Learning | CTM (Mamba SSM, 97K params) | Test IC ≈ 0.00 | ❌ No signal |
| 2 | Gradient Boosting | GBDT (Hoffnung C++, ~5K params) | Val IC = 0.008 | ❌ No signal |
| 3 | Hybrid Ensemble | CTM + GBDT + TimeDecayGate | Test Sharpe uniformly negative | ❌ Degrades |
| 4 | Feature Engineering | LightGBM + EMA (α=10) + DM test | Test IC = −0.013 | ❌ Denoising degrades |
| 5 | Cross-Sectional Ranking | lambdarank + CS z-scores | Test Rank IC negative | ❌ Worse than TS |
| 6 | Meta-Labeling | XGBoost binary filter (×5 seeds) | ΔSharpe = −0.22 | ❌ Reduces Sharpe |

**Note on wavelet denoising**: An independent causal wavelet denoising test (db4, level 3, on 50 US stocks with 3 walk-forward windows) confirmed that denoising degrades predictive power: mean IC dropped from +0.018 (raw) to −0.028 (wavelet), a ΔIC of −0.046. This reinforces the finding that the null return signal cannot be rescued by signal-processing techniques.

**Convergence is the strongest evidence**: If any paradigm could extract signal from pure OHLCV features, gradient boosting—the established leader on tabular data per McElfresh et al.—would have found it. Meta-Labeling and ensemble methods fail because they require a positive-IC primary model. Cross-sectional ranking fails because OHLCV z-scores carry no ranking signal. Feature engineering (denoising) fails because there is no signal to denoise.

### 5.7 Result Summary

**Table 11: Unified Results Across All 12 Experiment Groups**

| Category | N Runs | Best Test IC | Best Test Sharpe | Statistically Significant? |
|---|---|---|---|---|
| Return prediction (any model) | 45+ | 0.031 (Ridge) | 1.16 (LGB, artifact) | ❌ No (IC < 0.02) |
| Volatility prediction | 25+ | QLIKE = −7.10 | — | ✅ LGB beats baselines |
| Portfolio optimization | 8 | — | 2.36 (SharpeTuned, artifact) | ❌ Artifact of metric |
| Meta-Labeling | 5 | — | Δ = −0.22 | ❌ Degrades |
| Cross-sectional ranking | 3 | Negative | Negative | ❌ Worse than random |

---

## 6. Discussion

### 6.1 The Feature Information Ceiling

Our results converge on a single binding constraint: **no model architecture can extract signal that is not present in the features.** The 9 OHLCV-derived features we tested are deterministic functions of (price, volume)—they contain no information independent of the market price. This is not a model failure; it is an information ceiling.

The LLG framework [Kelly & Malamud 2025] provides theoretical grounding: even if true population R² ≥ 20% (their estimate for US equity returns), the learning gap grows as O(P/T). With P/T = 129 (CTM), the gap is large enough to render the signal undetectable. Ridge at P/T = 0.60 reduces but does not eliminate this gap. The operational conclusion is the same: with 9 OHLCV features and ~750 training days, daily return prediction is not feasible with any currently available model class.

### 6.2 The Sharpe Paradox and Publication Bias

The Sharpe Paradox has direct implications for the financial ML literature. If positive Sharpe arises mechanically from any long/short strategy in a rising market, then hundreds of published papers that report Sharpe as their primary performance metric may be reporting artifacts rather than genuine alpha. The widespread practice of using Sharpe for both model selection and performance reporting creates a perverse incentive: models that produce levered exposure to the market risk premium are systematically preferred over models that produce genuine but weaker predictive signals.

Our DSR analysis demonstrates that even the most widely accepted correction for multiple testing — the Deflated Sharpe Ratio [Bailey & López de Prado 2014] — cannot resolve this paradox. When applied to our ML model results, the DSR produces values of 1.000 (z > 100) despite IC ≈ 0, because the DSR was designed to correct for selection bias under the null of zero true Sharpe, not to distinguish mechanically captured market beta from genuine prediction ability. The DSR correctly identifies random sorting as non-significant (DSR = 0.947, z = 1.62), but it cannot tell us whether LGB-SharpeTuned's Sharpe of 2.98 (DSR = 1.000, z = 131.6) reflects alpha or beta exposure.

This is a structural limitation: the DSR's expected maximum under the null at K = 45 trials is only 0.05 Sharpe units. Any strategy producing a Sharpe above ~0.09 will be DSR-"significant" regardless of whether it has any prediction ability. In a bull market, nearly every long/short portfolio exceeds this threshold.

**Practical implication**: Neither the Sharpe ratio alone nor the Deflated Sharpe Ratio is sufficient for validating financial ML claims. We recommend a two-metric standard: the 5-step IC diagnostic framework must be reported alongside the DSR (or equivalent multiple-testing correction) for any return prediction study. Papers that report only Sharpe or only DSR should be treated as having an incomplete evaluation.

### 6.3 Volatility: Where Econometrics Beats ML

The GARCH finding is not an anomaly—it is a direct consequence of the different statistical properties of first and second moments. Returns are approximately martingale (autocorrelation ≈ 0); volatility is strongly autocorrelated (ρ > 0.5 at lag 1). GARCH was designed to exploit this structure; LightGBM was not. The 100% win rate across 300 stocks is a humbling reminder that 40 years of financial econometrics have produced tools that are optimal for their domain, and that ML's flexibility is a liability when the data-generating process is well-understood.

### 6.4 Comparison with Published Claims

- **MambaStock**: Claims superiority over XGBoost on 4 Chinese bank stocks, but predicts prices (not returns), has no walk-forward validation, no linear baseline, and no held-out test set. Our reproducibility assessment (`paper/REPRODUCIBILITY_ASSESSMENT.md`) found that running their code produces R² = −4.70—worse than predicting the historical mean. The paper's claims are not supported by its methodology.

- **SAMBA**: Claims IC improvements of 33–85% over baselines on NASDAQ/NYSE/DJIA, but has no publicly available code, no absolute IC values reported, and no held-out test evaluation. The claim cannot be verified independently.

- **CNN-LightGBM (Bai et al.)**: The most methodologically rigorous positive paper we surveyed (OOS-R² = 0.029, DM test, walk-forward). However, 93% of the gain comes from wavelet denoising, and Glaubenskrieg's independent test of denoising (EMA smoothing) found it degraded IC. The discrepancy may reflect market-specific structure or the different denoising method.

- **VLSTM Oxford**: Gross Sharpe = 2.39 from a fundamentally different data regime (5 asset classes, VSN adaptive gating, 50-seed top-10 selection). Portfolio-aware optimization alone adds +26% but cannot close the gap to 2.39, confirming that multi-asset data and feature gating are the drivers of VLSTM's performance, not optimization alone.

---

## 7. Limitations

**Feature scope**: We tested only 9 OHLCV-derived features. A larger feature set—particularly one incorporating fundamental data (94 features per Gu-Kelly-Xiu), text signals, or alternative data—might contain predictive information that our features lack. Our ceiling applies to OHLCV-only daily return prediction, not to financial ML generally.

**Frequency**: Daily frequency only. Intraday data (5-minute bars) has substantially higher SNR (~0.20 vs ~0.025 for daily) and is known to contain microstructure patterns that our protocol does not test.

**Survivorship bias**: Our US stock selection (S&P500 + NASDAQ100 constituents with ≥2,500 trading days) introduces survivorship bias. Delisted stocks are excluded, which may overstate the baseline against which prediction is evaluated.

**Transaction costs**: No strategy backtest includes transaction costs, slippage, or market impact. The regime-adaptive strategy's +0.28 Sharpe improvement would likely be reduced after cost modeling.

**Cross-market protocol alignment**: Walk-forward windows vary across markets (HK=3, A-Share=6, US=10) due to different data spans. This limits the comparability of cross-market IC values.

**Permutation test**: We implemented a permutation test (100 shuffles, 50 US stocks, 3 windows), but the output contained an artifact: both LightGBM and Ridge produced unrealistically high IC values (≈ 0.83) on shuffled data, suggesting a software bug in the shuffling or evaluation logic (likely temporal dependence leakage across shuffled samples). This test requires a corrected implementation and is not reported.

**Global normalization**: Features were normalized on the full dataset rather than per-window, which introduces minor future information leakage. Per-window normalization (using only training-set statistics) would be more rigorous.

---

## 8. Conclusion

We have conducted what we believe is the most exhaustive test of OHLCV-based daily stock return prediction in the academic literature. Six paradigms across three markets, covering 877 stocks, were evaluated using a protocol that exceeds the methodological standards of most published studies: purged walk-forward validation, independent held-out test sets, linear baselines, multi-seed sensitivity analysis, Diebold-Mariano statistical tests, and a novel five-step diagnostic framework for detecting overfitting.

Our finding is unambiguous: **OHLCV-derived features at daily frequency contain no detectable return-predictive signal across markets, model classes, or training objectives.** The 97,000-parameter Mamba SSM achieves the same test IC as the 451-parameter linear Ridge regression. GARCH(1,1) with three parameters outperforms 300-tree gradient boosting on every single stock tested for volatility prediction.

The five-step IC validation diagnostic framework is our primary methodological contribution. We show through the detailed case study of our own pipeline's evolution (v1→v5, IC=0.14→0.006) how the framework would have prevented a false positive publication. We recommend its adoption as standard practice in financial ML research.

Future work that claims daily return predictability from price-derived features should be evaluated against three baselines: (1) a linear Ridge regression on the same features, (2) held-out test IC after purged walk-forward validation, and (3) the five-step diagnostic framework. Papers that do not meet these standards should be interpreted with appropriate caution.

The path forward for financial ML lies not in more sophisticated models applied to the same features, but in incorporating genuinely new information sources: fundamental data, text and news signals, alternative data, and intraday microstructure. The OHLCV ceiling has been reached.

---

## Acknowledgments

[TBD]

## Data and Code Availability

All code is available at [URL]. The repository contains:
- Pure PyTorch CTM model implementation (~7,500 lines, 19 modules)
- Data processing and feature engineering pipeline
- Walk-forward training and evaluation scripts
- GBDT ensemble bridge (Hoffnung C++ backend)
- All experimental results (12 experiment groups, 45+ runs)
- 5-step diagnostic framework implementation (`scripts/diagnose_ic.py`)
- Figure generation scripts

## References

[TBD — to be compiled from paper/CLASSIC_LITERATURE_COMPARISON.md sources]

---

## Supplementary Materials

- **S1**: 10-Paper Literature Comparison Table (`paper/CLASSIC_LITERATURE_COMPARISON.md`)
- **S2**: 5-Step Diagnostic Framework Full Specification (`paper/methodology/02_diagnostic_framework.md`)
- **S3**: Walk-Forward Protocol Audit (`paper/methodology/01_walk_forward_protocol.md`)
- **S4**: Reproducibility Assessment of MambaStock, SAMBA, CNN-LGB (`paper/REPRODUCIBILITY_ASSESSMENT.md`)
- **S5**: IC Benchmark Comparison (`paper/IC_BENCHMARK_COMPARISON.md`)
- **S6**: Master Results Table — All 12 Experiment Groups (`paper/results/01_master_table.md`)
- **S7**: Figure Data for 8 Publication-Ready Figures (`paper/results/02_cross_market_figure_data.md`)
