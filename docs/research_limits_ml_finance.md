# Research: Fundamental Limits of Machine Learning in Finance

## Summary

Machine learning applied to stock return prediction faces hard statistical and economic ceilings. The best out-of-sample monthly R² for aggregate market returns is 1–3%, while daily return R² is typically sub-1%. Cross-sectional Information Coefficients (Rank IC) for realistic stock selection models rarely exceed 0.02–0.05 and are often accompanied by high volatility that makes them statistically indistinguishable from zero. The Limits-to-Learning Gap (LLG) framework reveals that true population predictability may be substantially higher (e.g., ~20% for US market returns), but finite samples and non-stationarity combine to create a fundamental ceiling that even sophisticated deep learning cannot breach. After accounting for transaction costs, post-publication decay, and researcher degrees of freedom, net deployable alpha is far smaller than headline backtest numbers suggest.

---

## Findings

### 1. Theoretical Upper Bound of Return Predictability (R²)

**The best-documented out-of-sample R² for monthly stock returns is 1–3%, even with state-of-the-art ML.**

- Gu, Kelly, and Xiu (2020) found that monthly stock-level R² rarely exceeds ~0.4%, and portfolio-level predictive R² reaches only ~1–2% even with neural networks and trees. [Source](https://www.aeaweb.org/conference/2020/preliminary/paper/KD2GZYtk)
- An information-theoretic study (arxiv 2506.03780) calibrated a polynomial lower bound: with 12,000 features and 12 monthly observations at R² = 2–3%, the required sample size to escape the bound exceeds 25–30 years of data. [Source](https://arxiv.org/pdf/2506.03780)
- The Limits-to-Learning Gap (LLG) by Kelly and Malamud (2025) provides a universal lower bound on the discrepancy between empirical and population fit. For US market excess returns, the LLG implies true population predictability of **at least 20%** — approximately 10–20× higher than typical realized OOS R². This means predictability exists in principle but is extremely hard to learn from finite samples. [Source](https://arxiv.org/html/2512.12735v1)
- The high-dimensional learning regime (P > T, where parameters exceed observations) guarantees that standard OOS metrics systematically understate true predictability, creating a "dark matter" problem in asset pricing.

### 2. Information Coefficient (IC) Benchmarks

**Realistic IC values in quantitative finance are materially close to zero and highly volatile.**

- The Information Coefficient (IC) is defined as the cross-sectional correlation between factor scores at time t and realized returns at t+1: \(IC_t = \text{corr}(f_t, r_{t+1})\). [Source](https://bagelquant.com/predictability-measure/)
- Rank IC (Spearman) is preferred over Pearson IC in practice because it is more robust to outliers. [Source](https://dev.to/linou518/quant-factor-research-in-practice-ic-ir-and-the-barra-multi-factor-model-1h8k)
- The fundamental law of active management: \(IR = IC \times \sqrt{\text{breadth}}\). A typical IC of 0.02–0.05 across 100–500 stocks yields an Information Ratio of 0.2–1.0. [Source](https://www.pfolio.io/academy/information-coefficient)
- Research shows IC from realistic stock selection models "can hardly be materially different from zero and is often accompanied with high volatility." [Source](https://arxiv.org/pdf/2010.08601)
- Measuring forecasting skill is fundamentally difficult: the theoretical understanding of IC has been "limited to the partial equilibrium setting" and standard IC estimates are unreliable in finite samples. [Source](https://onlinelibrary.wiley.com/doi/10.1002/for.2320)

### 3. Efficient Market Hypothesis and ML Evidence

**Markets are not perfectly efficient, but the predictable component is small, noisy, and non-stationary.**

- A comprehensive meta-study across multiple stock indices found ANN, LR, and SVM achieve >70% directional accuracy, but this converges to 50% out-of-sample for all hyperparameter combinations — indistinguishable from random. In-sample accuracy reached 100% for many configurations. [Source](https://www.sciencedirect.com/science/article/abs/pii/S105752192400406X)
- The nonstationarity-complexity tradeoff: complex models reduce misspecification error but require longer training windows, which introduce stronger non-stationarity from structural breaks and regime shifts. During NBER recessions, simpler models on shorter windows consistently outperform complex models on longer windows. [Source](https://arxiv.org/html/2512.23596v1)
- GRU-D neural networks applied globally find statistically significant departures from purely random behavior, but the economic magnitude is small. [Source](https://www.mdpi.com/2227-7072/14/2/46)
- A study on S&P 500 over 22 years using LSTM found "predictive patterns" consistent with adaptive market hypothesis (AMH) rather than strict EMH — markets exhibit time-varying efficiency. [Source](https://www.mdpi.com/2227-7390/12/19/3066)

### 4. Why Financial Factors Have Limited Predictive Power

**Three structural reasons create a hard ceiling on ML performance in finance.**

**a) Low Signal-to-Noise Ratio (SNR):** Daily stock returns have an SNR below 0.01. The irreducible noise component (\(\sigma_\varepsilon^2\)) dominates the explained variance (\(E[f_t^2]\)). In Kelly & Malamud's decomposition: \(R^2_* = 1 - \sigma_\varepsilon^2 / (E[f_t^2] + \sigma_\varepsilon^2)\). With daily return volatility of ~2% and predictable signal of <0.05%, the theoretical maximum R² is inherently bounded.

**b) Non-Stationarity:** Financial data-generating processes shift over time due to structural breaks, regime changes, and economic cycles. The prediction error bound decomposes into: Misspecification(\(\mathcal{F}\)) + Uncertainty(\(\mathcal{F}, n_k\)) + Non-stationarity(k). Increasing training window k reduces statistical uncertainty but increases non-stationarity bias, creating a fundamental tradeoff that cannot be eliminated. [Source](https://arxiv.org/html/2512.23596v1)

**c) Curse of Dimensionality:** The LLG \(\widehat{\mathcal{L}} = O(P/T)\) grows with the ratio of features to observations. In the over-parameterized regime (\(P > T\)), the LLG can be very large. Kelly & Malamud show that with typical parameter choices, standard ML approaches can substantially understate true predictability — but they cannot escape the LLG bound. [Source](https://arxiv.org/html/2512.12735v1)

### 5. Maximum Practical Performance — Realistic Ceilings

**A comprehensive synthesis of empirical evidence yields these realistic performance ceilings:**

| Metric | Paper Benchmark | Post-Friction Realistic |
|--------|----------------|------------------------|
| Monthly OOS R² (market) | 1–3% | <1% net |
| Daily OOS R² (cross-section) | <0.5% | negligible |
| Cross-sectional Rank IC | 0.03–0.08 (academic factors) | 0.01–0.04 (post-publication) |
| Gross Sharpe (DL portfolio) | 1.5–2.4 (Oxford benchmark) | 0.5–1.0 after costs |
| Net Sharpe (deployable) | — | 0.3–0.7 (realistic) |

**Key evidence:**

- Accounting for transaction costs, post-publication decay, and post-decimalization liquidity, cumulative ML strategy performance is reduced by **~57%** on average. LSTM-based strategies remain the most profitable. [Source](https://www.aeaweb.org/conference/2025/program/paper/TihddefB)
- The Oxford large-scale benchmark (2010–2025, futures across 5 asset classes) found VLSTM achieves gross Sharpe 2.39 (net would be substantially lower). Most generic deep learning models show Sharpe <1.0. Linear benchmarks produce Sharpe <1.0. [Source](https://www.arxiv.org/pdf/2603.01820)
- Design choice variation in ML portfolio construction exceeds standard error by **59%** — meaning two researchers with identical data and model class can produce meaningfully different results. Monthly top-minus-bottom returns range from 0.13% to 1.98%. [Source](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5031755)
- The "Alpha Illusion": reported LLM trading agent Sharpe ratios (e.g., FinCon 3.27) are not deployment evidence — they fail to control for temporal contamination, full friction modeling, short-sample uncertainty, miscalibration, and parametric prior lock-in. [Source](https://arxiv.org/html/2605.16895)
- Post-cutoff evaluation: FinMem's total return drops by ~71.85% and QuantAgent's Sharpe by ~51.48% when the evaluation window crosses the model's pretraining cutoff — directly quantifying the contamination component of reported alpha. [Source](https://arxiv.org/pdf/2510.07920)

### 6. Intraday vs. Daily Predictability

**Intraday return predictability is more persistent than daily.**

- The largest intraday study (~900M observations) finds consistent out-of-sample predictability across market, sector, and individual stock returns at various intraday horizons. Linear models have the strongest statistical power, but nonlinear models economically dominate (higher Sharpe from long-short portfolios). [Source](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4496917)
- Higher-frequency data contains more exploitable patterns, but these decay rapidly and are more sensitive to transaction costs and market microstructure noise.

---

## Sources

### Kept (Primary Evidence)

- **"Limits To (Machine) Learning"** (Kelly & Malamud, 2025) — Foundational paper deriving the LLG bound; empirically finds true market return predictability ≥20% vs realized OOS R² of 1–2%. Essential for understanding the theoretical ceiling. [arxiv.org/html/2512.12735v1](https://arxiv.org/html/2512.12735v1)
- **"The nonstationarity-complexity tradeoff in return prediction"** (Capponi, Huang, Wang, 2025) — Characterizes the fundamental tradeoff between model complexity and training window under non-stationarity; provides finite-sample error bounds. [arxiv.org/html/2512.23596v1](https://arxiv.org/html/2512.23596v1)
- **"Deep Learning for Financial Time Series: A Large-Scale Benchmark"** (Saly-Kaufmann et al., 2026) — Comprehensive benchmark of modern DL architectures on futures 2010–2025; VLSTM achieves Sharpe 2.39 (gross), most models <1.0. [arxiv.org/pdf/2603.01820](https://www.arxiv.org/pdf/2603.01820)
- **"The Alpha Illusion"** (Ye et al., 2025) — Systematic critique of LLM trading agent claims; establishes P1–P6 protocols for deployment-grade evidence. [arxiv.org/html/2605.16895](https://arxiv.org/html/2605.16895)
- **"Chaos, overfitting and equilibrium: To what extent can ML beat the market?"** (2024) — OOS accuracy converges to 50% for all hyperparameter combinations; majority of models do not outperform buy-and-hold. [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S105752192400406X)
- **High-dimensional learning in finance** (arxiv 2506.03780) — Information-theoretic lower bounds: 12K features, 12 monthly obs, R² 2–3% requires >25–30 years data. [arxiv.org/pdf/2506.03780](https://arxiv.org/pdf/2506.03780)
- **"Information Coefficient as a Performance Measure"** (2020) — IC from realistic models can hardly be materially different from zero. [arxiv.org/pdf/2010.08601](https://arxiv.org/pdf/2010.08601)
- **"Design choices, ML, and the cross-section of stock returns"** (SSRN 5031755) — Non-standard error from design choices exceeds standard error by 59%. [papers.ssrn.com](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5031755)
- **"Intraday Stock Predictability Everywhere"** (SSRN 4496917) — 900M observations; linear models strongest statistically, nonlinear models dominate economically. [papers.ssrn.com](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4496917)
- **"Ensemble ML and Stock Return Predictability"** (AEA 2020) — Ensemble methods achieve monthly OOS R² of 2–3%. [aeaweb.org](https://www.aeaweb.org/conference/2020/preliminary/paper/KD2GZYtk)

### Dropped

- Multiple MDPI survey papers (e.g., Stock Market Prediction survey) — Redundant with systematic reviews above; lower journal quality.
- "Applying ML to predict stock price trend – Vietnam" — Single-market study with claimed 93% accuracy; likely methodological issues (look-ahead bias).
- Blog posts and non-academic sources (BagelQuant, DEV Community, pfolio academy) — Used only for IC definition references; primary evidence from academic papers.
- LLM-specific trading papers (FinCon, FinMem, TradingAgents) — Evaluated via "The Alpha Illusion" critique paper, which comprehensively covers these.

---

## Gaps

1. **Daily-frequency R² benchmarks are sparse.** Most rigorous empirical work operates at monthly frequency. The best daily-frequency estimates come from the nonstationarity-complexity paper (OOS R² ≈ 0.049 for industry portfolios) and the Oxford benchmark (Sharpe-based, not R²). The direct daily R² ceiling for individual stock returns is not precisely quantified in the literature.
2. **Post-cost net Sharpe estimates are model-dependent.** While the 57% performance reduction from costs/publication decay is documented, specific net Sharpe ceilings for different model classes (e.g., transformer vs. LSTM vs. linear) are not systematically compared.
3. **The LLG framework is new (Dec 2025) and has not been independently replicated.** Its implication that true population R² for market returns is ≥20% is striking and challenges conventional wisdom. Independent validation would strengthen or qualify this claim.
4. **Intraday vs. daily predictability boundary.** The intraday paper finds "predictability everywhere," but the realistic ceiling after microstructure costs (bid-ask bounce, latency, adverse selection) at the intraday frequency is not quantified in a unified framework.

---

## Practical Implications for This Project

For the Glaubenskrieg stock prediction project (daily frequency, multi-asset), the following ceilings should guide expectations:

- **Daily OOS R² ceiling:** ≤0.5–1.0% for individual stocks; 3–5% for portfolios (before costs)
- **Rank IC ceiling:** 0.02–0.05 (Spearman) for daily cross-sectional prediction
- **Gross Sharpe ceiling:** ~1.5–2.0 for sophisticated DL models on liquid instruments
- **Net Sharpe (realistic, after costs):** 0.3–0.8 — this is what a deployable strategy can achieve
- **Critical design choices:** Non-stationarity handling (adaptive window selection), transaction cost modeling, and out-of-sample protocol rigor matter more than marginal improvements in model architecture
- **Warning:** Any daily R² above ~2% or Sharpe above ~3 (gross) should be treated as likely contaminated by look-ahead bias, short-sample noise, or unmodeled costs
