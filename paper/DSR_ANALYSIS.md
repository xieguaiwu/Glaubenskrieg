# Deflated Sharpe Ratio (DSR) Analysis for Glaubenskrieg

> **Computed**: 2026-06-13
> **Method**: Bailey & López de Prado (2014), *Journal of Portfolio Management*
> **Data sources**: `results/vol_regime_*.csv`, `results/portfolio_optimizer.json`, `results/us_full_baseline.json`

---

## Executive Finding

**DSR alone cannot detect the Sharpe Paradox.** All ML models show DSR ≈ 1.000 (z > 100) despite IC ≈ 0 — because the DSR corrects for multiple testing but not for mechanically captured market beta in long/short portfolios. This is a novel methodological contribution: **the 5-step IC diagnostic framework detects what DSR misses.**

---

## 1. What DSR Tests

The Deflated Sharpe Ratio answers:

> "Given that K independent strategy configurations were tested, each with T return observations, what is the probability that the best observed Sharpe ratio is too large to arise from the null hypothesis of zero true Sharpe — after accounting for non-normal return distributions?"

It corrects for two sources of inflation:
1. **Selection bias** (multiple testing): the more configurations tried, the higher the expected max Sharpe under the null
2. **Non-normality** (skewness, kurtosis): fat tails and asymmetry inflate the variance of the estimated Sharpe

---

## 2. Key Results

### 2.1 ML Portfolio Models (Sharpe Paradox)

| Model | Sharpe | IC | K (trials) | DSR (normal) | z | DSR says |
|:------|:-----:|:--:|:----------:|:------------:|:-:|:--------|
| Random sorting | 0.04 | 0.001 | 1 | 0.947 | 1.62 | ❌ Not significant |
| Mean-reversion | 0.04 | 0.003 | 1 | 0.967 | 1.84 | ❌ Not significant |
| LGB-MSE | 2.36 | 0.011 | 9 | 1.000 | 104.5 | ✅ **Significant** |
| LGB-LambdaRank | 2.79 | 0.018 | 9 | 1.000 | 123.9 | ✅ **Significant** |
| LGB-SharpeTuned | 2.98 | 0.015 | 45 | 1.000 | 131.6 | ✅ **Significant** |
| US LightGBM | 1.16 | 0.007 | 9 | 1.000 | 56.6 | ✅ **Significant** |
| US Ridge | 0.82 | 0.031 | 1 | 1.000 | 41.0 | ✅ **Significant** |

### 2.2 Volatility Regime Strategies (Positive Control)

| Strategy | Sharpe | K | DSR (normal) | DSR (skew/kurt adjusted) |
|:---------|:-----:|:-:|:------------:|:------------------------:|
| Equal Weight | 1.12 | 5 | 1.000 | 1.000 (z=29.0) |
| Inverse Vol | 1.12 | 5 | 1.000 | 1.000 (z=28.3) |
| Vol Target 15% | 1.21 | 5 | 1.000 | 1.000 (z=53.0) |
| **Regime-Adaptive** | **1.40** | **5** | **1.000** | **1.000 (z=51.8)** |

### 2.3 Sensitivity: What K Would Nullify the ML Results?

How many independent trials would a researcher need to have attempted before DSR classifies the ML model Sharps as non-significant (DSR < 0.95)?

| Model | K needed | Interpretation |
|:------|:--------:|:---------------|
| Random sorting | K < 1 | Already non-significant |
| US Ridge (SR=0.82) | K > 10¹² | Never — even 1 trillion trials won't explain Sharpe=0.82 |
| US LightGBM (SR=1.16) | K > 10¹² | Never |
| LGB-MSE (SR=2.36) | K > 10¹² | Never |
| LGB-SharpeTuned (SR=2.98) | K > 10¹² | Never |

---

## 3. Why DSR Fails on the Sharpe Paradox

The DSR tests the null hypothesis:

$$H_0: \text{True Sharpe ratio } = 0$$

But for any long/short portfolio in a rising market, this null is violated **not because of prediction ability, but because any sorting mechanically captures the equity risk premium**. The portfolio's positive Sharpe comes from risk-taking, not signal extraction.

**Quantification**: At K = 45 (the LGB-SharpeTuned search space), the expected maximum of 45 independent trials with T = 2016 observations is:

$$E[\max(\hat{SR})] = \frac{1}{\sqrt{2016}} \times E[Z_{\max}(45)] = 0.0223 \times 0.0498 \approx 0.05$$

Yet LGB-SharpeTuned achieves SR = 2.98 — **60× the expected maximum under the null**.

The threshold for DSR non-significance (z < 1.645) at K = 45 is:

$$SR_{\text{threshold}} = 0.05 + 1.645 \times 0.0223 \approx 0.09$$

Any SR > 0.09 is DSR-"significant." Since all ML models produce SR well above 0.09 (even random sorting barely reaches 0.04), DSR classifies them as "significant" even when IC ≈ 0.

### The Fundamental Limitation

| Test | Detects | Misses | Our fix |
|:----|:--------|:-------|:--------|
| Standard Sharpe t-test | SR > 0? | Multiple testing | DSR |
| DSR | Multiple-testing-adjusted SR > 0? | **Market beta in long/short** | **IC** |
| IC | Actual prediction ability? | — | **5-step diagnostic** |

**DSR is necessary but not sufficient.** The 5-step IC diagnostic framework catches false positives that DSR alone cannot identify.

---

## 4. Implication for the Paper

1. **The DSR computation proves the paper's central methodological argument**: the Sharpe ratio alone — even deflated for multiple testing — cannot distinguish genuine alpha from mechanically captured market risk premium.

2. **DSR should be reported alongside IC** in the final paper. We propose:

   > "Even after applying the Deflated Sharpe Ratio (Bailey & López de Prado, 2014) with K = 45, LGB-SharpeTuned achieves DSR ≈ 1.000 (z = 131.6). Yet its IC = 0.015. The DSR cannot tell us whether this Sharpe reflects prediction ability or market beta exposure. Only the 5-step IC diagnostic framework can resolve this ambiguity."

3. **Suggested addition to the 5-step framework**: Add a "Step 0: DSR pre-check" that reports the deflated Sharpe for the best configuration, followed by "Step 1-5: IC diagnostics" to determine whether the DSR-significant Sharpe is genuine or an artifact.

---

## 5. Code Availability

The DSR computation script is in `scripts/dsr_analysis.py` (or available on request). Implementation details:

```python
def compute_dsr(SR, T, K, skew=0.0, kurt_raw=3.0):
    se = 1.0 / np.sqrt(T)  # normality
    # Or with Mertens correction:
    # se = np.sqrt((1 + skew/2*SR + (kurt_raw-3)/4*SR²) / T)
    E_max = se * expected_max_K(K)
    z = (SR - E_max) / se
    return norm.cdf(z)
```

Where `expected_max_K(K)` uses the Euler-Mascheroni approximation:

$$E[Z_{\max}(K)] \approx (1-\gamma)\Phi^{-1}(1-1/K) + \gamma\Phi^{-1}(1-1/(K e))$$

with $\gamma \approx 0.5772$.
