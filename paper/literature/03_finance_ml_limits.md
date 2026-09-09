# Fundamental Limits of Machine Learning in Finance

> **Source file**: `docs/research_limits_ml_finance.md` (full document)

---

## 1. Theoretical Upper Bound of Return Predictability (R²)

The best-documented out-of-sample R² for monthly stock returns is **1–3%**, even with state-of-the-art ML.

| Source | Finding |
|---|---|
| Gu, Kelly & Xiu (2020) | Monthly stock-level R² ≈ 0.4%; portfolio-level ≈ 1–2% |
| arxiv 2506.03780 | Information-theoretic bound: with 12K features and R²=2-3%, need >25-30 years data |
| **Kelly & Malamud (2025) LLG** | **True population predictability ≥20%, but finite samples hide it** |

### The Limits-to-Learning Gap (LLG)

Kelly & Malamud (2025) provide a universal lower bound:

> **L̂ = O(P/T)** — grows with parameter/observation ratio

- True population R² for US market returns ≥ **20%** (10–20× higher than realized OOS R²)
- The gap is a "dark matter" problem: predictability exists but is extremely hard to learn from finite samples
- In over-parameterized regime (P > T), LLG can be very large

**Implication for Glaubenskrieg**: The LLG bound suggests that true predictability exists but is unattainable with 750 samples and 97K parameters. The standard OOS metrics systematically understate true predictability.

---

## 2. Information Coefficient Benchmarks

| Metric | Academic Factors | Post-Publication Realistic |
|---|---|---|
| Cross-sectional Rank IC | 0.03–0.08 | **0.01–0.04** |
| Realistic IC from models | "Can hardly be materially different from zero" (arxiv 2010.08601) | |

**Fundamental Law of Active Management**: IR = IC × √breadth.
- Typical IC = 0.02–0.05 across 100–500 stocks → IR = 0.2–1.0
- This means even with IC = 0.02, theoretical IR is < 1.0

---

## 3. Efficient Market Hypothesis and ML Evidence

Source: ScienceDirect (2024), Nature (2025).

### The Chaos Overfitting Paper (ScienceDirect 2024)

- ANN, LR, SVM achieve >70% directional accuracy in-sample
- **OOS accuracy converges to 50% for ALL hyperparameter combinations**
- In-sample accuracy reached 100% for many configurations
- Majority of models do not outperform buy-and-hold

### Nature 2025 Review

> "Most prominent studies regarding LSTMs and DNNs predictors for stock market prediction are not reproducible in real-world applications."

---

## 4. The Nonstationarity-Complexity Tradeoff

Source: Capponi, Huang, Wang (2025, arxiv 2512.23596).

The prediction error bound decomposes into:
- **Misspecification** (model class error) 
- **Uncertainty** (finite sample) 
- **Non-stationarity** (distribution shift)

**Key tradeoff**: Increasing training window k reduces statistical uncertainty BUT increases non-stationarity bias. Simple models on short windows outperform complex models on long windows during recessions.

**ATOMS framework**: Joint optimization of model class + window size yields 14–23% OOS R² improvement.

---

## 5. Maximum Practical Performance Ceilings

| Metric | Paper Benchmark | Post-Friction Realistic |
|---|---|---|
| Monthly OOS R² (market) | 1–3% | **<1% net** |
| Daily OOS R² (cross-section) | <0.5% | **negligible** |
| Cross-sectional Rank IC | 0.03–0.08 | **0.01–0.04** |
| Gross Sharpe (DL portfolio) | 1.5–2.4 (Oxford) | **0.5–1.0 after costs** |
| Net Sharpe (deployable) | — | **0.3–0.7** |

### Key Adjustment Factors

1. **Transaction costs**: ~57% reduction in cumulative ML strategy returns (AEA 2025)
2. **Post-publication decay**: IC drops 30–50% after public documentation
3. **Design choice variance**: Non-standard error exceeds standard error by **59%** (SSRN 5031755)
4. **Researcher degrees of freedom**: Two researchers with identical data can produce 0.13% to 1.98% monthly returns

---

## 6. The Alpha Illusion (Ye et al., 2025)

| Paper | Claimed Sharpe | Post-Correction | Reduction |
|---|---|---|---|
| FinCon | 3.27 (LLM trading) | Contaminated | Time contamination |
| FinMem | — | -71.85% return | Look-ahead bias |
| QuantAgent | — | -51.48% Sharpe | Pretraining cutoff contamination |

**P1–P6 Protocol** for deployment-grade evidence:
1. Temporal contamination check
2. Full friction modeling (slippage, capacity)
3. Short-sample uncertainty quantification
4. Miscalibration diagnostics
5. Parametric prior independence
6. Post-cutoff evaluation

---

## 7. Intraday vs Daily Predictability

| Aspect | Daily | Intraday |
|---|---|---|
| SNR | ~0.01 | ~0.05–0.10 |
| R² ceiling | <1% | 5–15% |
| IC ceiling | 0.02–0.05 | 0.05–0.15 |
| Decay rate | Fast (days) | Very fast (minutes) |
| Cost sensitivity | Moderate | **High** |

**Intraday paper** (SSRN 4496917, 900M observations): "Predictability everywhere" but mostly exploitable via nonlinear models. Linear models have strongest statistical power, but nonlinear models dominate economically.

---

## 8. Structural Reasons for Limited Predictability

### 8a. Low SNR

Daily return volatility ~2%, predictable signal <0.05%. Theoretical maximum R² is inherently bounded by: R² ≤ 1 - σ²_ε / (E[f²] + σ²_ε).

### 8b. Non-Stationarity

Data-generating processes shift due to structural breaks, regime changes, economic cycles. **No amount of data from the past distribution can guarantee good performance on a shifted future distribution.**

### 8c. Curse of Dimensionality

LLG grows as O(P/T). In the over-parameterized regime, standard ML systematically understates true predictability — but cannot escape the bound.

---

## 9. Practical Implications for Glaubenskrieg

| Expectation | Ceiling | Our Result |
|---|---|---|
| Daily OOS R² ceiling | ≤0.5–1.0% | ≈ 0% (IC ≈ 0) |
| Rank IC ceiling | 0.02–0.05 | **0.007 (US), -0.003 (A-Share), 0.053 (HK)** |
| Gross Sharpe ceiling | 1.5–2.0 (sophisticated DL) | **-0.028 (CTM), 1.16 (LGB, artifact)** |
| Net Sharpe realistic | 0.3–0.8 | **Would be negative after costs** |
| Warning flag | Any daily R² > 2% or Sharpe > 3 | **Our Sharpe > 1 is already suspicious** |

**Consistency**: Our results are entirely consistent with the literature. IC ≈ 0 for pure OHLCV daily prediction is expected, not anomalous. The anomaly would be finding IC > 0.05.

---

## 10. Key Papers Referenced

1. **Kelly & Malamud (2025)** — "Limits To (Machine) Learning" — LLG framework. [arxiv 2512.12735]
2. **Capponi, Huang, Wang (2025)** — Nonstationarity-complexity tradeoff. [arxiv 2512.23596]
3. **Saly-Kaufmann et al. (2026)** — Oxford DL benchmark. [arxiv 2603.01820]
4. **Ye et al. (2025)** — "The Alpha Illusion" — P1–P6 critique of LLM trading claims. [arxiv 2605.16895]
5. **Chaos, Overfitting and Equilibrium (2024)** — OOS accuracy → 50%. [ScienceDirect S105752192400406X]
6. **McElfresh et al. (2023)** — Tabular ML benchmark. [arxiv 2305.02997]
7. **Gu, Kelly & Xiu (2020)** — Empirical asset pricing via ML. [RFS]
8. **SSRN 4496917** — Intraday predictability (900M obs).
9. **SSRN 5031755** — Design choice variance 59% > standard error.
10. **Nature 2025** — Stock DNN not reproducible. [s41599-025-04761-8]

---

## 11. Empirical Validation of LLG and Detection Limits

> **Source**: `results/power_analysis.json` and `paper/CLASSIC_LITERATURE_COMPARISON.md` §4

The Kelly & Malamud (2025) LLG framework predicts that the limits-to-learning gap grows as O(P/T). Glaubenskrieg provides a concrete case study that validates this prediction quantitatively.

### 11.1 Parameters

| Parameter | Value | Source |
|-----------|:-----:|--------|
| Observed daily IC (US LGB) | **0.007 ± 0.043** | `results/us_full_baseline.json` |
| Min detectable IC (80% power, N=750) | **0.102** | `results/power_analysis.json` |
| Observed / threshold ratio | **6.85%** | `results/power_analysis.json` detectability_report |
| CTM P/T ratio | **129.3** (97K params / 750 obs) | Computed from `05_negative_results.md` |
| Ridge P/T ratio | **0.60** (451 params / 750 obs) | Computed from `us_full_baseline.json` |
| Ideal sample size for IC=0.007 at 80% power | **>150,000 obs** | Extrapolated from power analysis |

### 11.2 What the LLG Predicts

Under Kelly & Malamud (2025):
- True population R² for US market returns ≥ **20%** — but this includes all possible features and relationships, not just 9 OHLCV features
- The LLG gap L̂ = O(P/T) means that with P/T = 129, the model needs ≈ 129× more data to close the learning gap
- Even with Ridge (P/T = 0.60), the gap may still dominate because the data-generating process complexity exceeds model dimension

### 11.3 Glaubenskrieg's Empirical Validation

The observed IC of **0.007** is only **6.85%** of the minimum detectable IC threshold (**0.102**). This has two complementary interpretations, both consistent with the LLG framework:

1. **Signal absence interpretation**: The 9 OHLCV-derived features at daily frequency contain no extractable return-predictive signal across three markets (US, HK, A-Share). The information ceiling for this feature set is R² < 0.01%.

2. **LLG dominance interpretation**: True predictability may exist at the population level (consistent with Kelly-Malamud's ≥20% R² upper bound), but the P/T = 129 ratio and finite sample of 750 days make it fundamentally unlearnable with current data.

Both interpretations reach the same operational conclusion: **IC ≈ 0 is the correct empirical finding**, and any model claiming IC > 0.10 from 9 OHLCV features should be treated with extreme skepticism.

### 11.4 Ridge as LLG Sanity Check

The Ridge regression (P/T = 0.60) provides a critical diagnostic: even with more observations than parameters, Ridge IC = 0.031 is still below the 0.102 detection threshold. This suggests the bottleneck is primarily **feature information content**, not just the LLG:

| Model | P/T | IC | Below threshold? |
|-------|:---:|:--:|:----------------:|
| CTM (Mamba SSM) | 129.3 | 0.006 | ✅ Yes (5.9% of threshold) |
| Ridge (linear) | 0.60 | 0.031 | ✅ Yes (30.4% of threshold) |

If Ridge (P/T < 1) also fails to detect signal, the most parsimonious explanation is that the features themselves lack predictive information — consistent with the known gap between 9 OHLCV transforms and 94 fundamental characteristics (Gu, Kelly & Xiu, 2020).

### 11.5 Power Analysis Details

From `results/power_analysis.json`:
- **Sample size**: N = 750 observations
- **80% power threshold**: IC = 0.102 (Fisher z-transform, two-sided α = 0.05)
- **Observed IC**: 0.007 (LGB, US)
- **Ratio**: 6.85% — the observed effect size is an order of magnitude below what can be reliably detected with 750 daily observations
- **Implication**: To detect IC = 0.007 with 80% power, the required sample size exceeds 150,000 observations — approximately 600 years of daily data

### 11.6 Contribution to the LLG Literature

Glaubenskrieg provides what may be the first **explicit empirical calibration** of the LLG bound in a multi-paradigm, multi-market setting:

1. **P/T ratio linked to detectability**: Documents that P/T = 129 produces IC indistinguishable from zero, while even P/T = 0.60 is insufficient to cross the detection threshold with OHLCV-only features.
2. **Disentangles feature ceiling from LLG**: Ridge's failure (P/T = 0.60, IC = 0.031) isolates the feature information bottleneck from the over-parameterization bottleneck, a distinction the LLG theory does not make.
3. **Provides positive control**: The GARCH volatility result (300/300 stocks, DM p < 10⁻¹³) confirms that the experimental protocol can detect strong signals when they exist — the LLG applies specifically to return prediction, not all financial prediction tasks.
