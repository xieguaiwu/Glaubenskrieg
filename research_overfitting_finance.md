# Research: Overfitting in Deep Learning for Financial Time Series (2023–2026)

## Summary

Financial deep learning models consistently suffer from overfitting due to low signal-to-noise ratios (SNR ~0.8–1%), non-stationarity, and regime shifts. The literature reveals three convergent findings: (1) complex models (Transformers, deep LSTMs) often underperform simpler alternatives (linear, SVM, shallow RNNs) in volatile markets; (2) the primary defense is not architecture complexity but feature engineering, adaptive validation, and alignment of the loss function with trading objectives; (3) walk-forward validation with purging and embargoing remains the gold standard, but adaptive window selection (ATOMS) that jointly optimizes model class and training window size yields 14–23% out-of-sample R² improvements over fixed-window baselines.

---

## Findings

### 1. Overfitting Diagnosis Methods

1. **LSTM collapse to unconditional mean** — A key diagnostic signature: when the SNR is too low, neural networks learn the optimal MSE-minimizing strategy of predicting zero/mean. Prediction variance goes to zero and attention weights become uniform across lags. Hit rate drops below 50%. [Source: Kang (2026), "The Limits of Complexity"](https://arxiv.org/html/2601.07131)

2. **Uniform attention weights** — Multi-head attention weights uniformly distributed (0.10 across all lags) indicate failure to learn meaningful temporal patterns. This is a reliable diagnostic that the model has insufficient signal. [Source](https://arxiv.org/html/2601.07131)

3. **Train-test gap widening under aggressive regularization** — In the TIPS paper, low-temperature distillation improved test SR but widened the train-test gap, reflecting higher overfitting risk. Aggressive label smoothing and SWA were required to close this gap. [Source: TIPS (KDD 2026)](https://arxiv.org/html/2603.16985v2)

4. **Backtest Overfitting Probability (PBO)** — Arian et al. (2024) provide a comprehensive comparison of out-of-sample testing methods in a synthetic controlled environment, showing that standard train/test splits are insufficient and that PBO-style metrics should be used to quantify the probability that a strategy's performance is due to overfitting. [Source: SSRN 4686376](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376)

5. **Seed sensitivity analysis** — The Oxford benchmark (Saly-Kaufmann et al., 2026) explicitly tests robustness to seed selection by comparing full-budget (50 seeds, top 10) vs. reduced-budget (25 seeds, top 5) evaluation. Performance rankings remained stable, validating that differences were not artifacts of favorable initialization. [Source: arxiv 2603.01820](https://arxiv.org/html/2603.01820)

### 2. Regularization Techniques That Work for Financial Time Series

1. **GMADL Loss Function (2024)** — Michałków et al. propose the Generalized Mean Absolute Directional Loss as a replacement for MSE/MAE. It's fully differentiable (unlike MADL), directly optimizes for profit direction, and includes parameters `a` (slope sensitivity around zero) and `b` (reward for high-magnitude returns) that let practitioners tune the loss to penalize excessive trading. GMADL outperforms MSE across Transformer, LSTM, and Perceptron architectures. [Source: arxiv 2412.18405](https://arxiv.org/html/2412.18405v1)

2. **Direct Sharpe Ratio optimization** — The Oxford benchmark trains all models end-to-end by minimizing negative annualized Sharpe Ratio on pooled portfolio returns. This aligns training directly with the trading objective, reducing the disconnect between forecast accuracy and investment performance. [Source: arxiv 2603.01820](https://arxiv.org/html/2603.01820)

3. **Variable Selection Networks (VSN) as adaptive feature gating** — VSN performs dynamic soft selection of relevant covariates at each time step, suppressing noisy features. VLSTM (VSN+LSTM) achieved the highest overall Sharpe ratio (2.39) in the Oxford benchmark by adaptively filtering features. [Source: arxiv 2603.01820](https://arxiv.org/html/2603.01820)

4. **Ensemble seed selection** — The Oxford benchmark averages positions from the top-S seeds (by validation loss) rather than a single best model, reducing turnover and improving robustness to transaction costs. [Source: arxiv 2603.01820](https://arxiv.org/html/2603.01820)

5. **TIPS: Distillation with aggressive regularization** — The TIPS framework (KDD 2026) trains bias-specialized teacher Transformers (causal, local, periodic attention masks), then distills into a student with aggressive label smoothing (ε=0.9), low-temperature distillation (τ=0.01), and Stochastic Weight Averaging. This achieves ensemble-level robustness at single-model inference cost. [Source: arxiv 2603.16985v2](https://arxiv.org/html/2603.16985v2)

6. **Structural Risk Minimization (SVM)** — Kaygın et al. (2026) show SVM consistently outperforms LSTM/GRU in high-volatility crypto assets (BTC MAPE: 0.55% vs 2.93%), demonstrating that SRM-based models resist noise better than ERM-based deep networks. [Source: Mathematics 14(6), 989](https://www.mdpi.com/2227-7390/14/6/989)

7. **Ridge/LASSO with adaptive regularization** — The nonstationarity-complexity tradeoff paper finds that ridge regression with L2 regularization on shorter training windows often outperforms random forests on long windows during recessionary periods. [Source: arxiv 2512.23596](https://arxiv.org/html/2512.23596v1)

### 3. The Validation-Test Gap

1. **"Chaos, overfitting and equilibrium" (2024)** — In-sample accuracy reached 100% for many hyperparameter combinations, but out-of-sample accuracy converged to 50% for ALL combinations. Most models did not outperform buy-and-hold. This is the starkest demonstration of the validation-test gap in the literature. [Source: ScienceDirect S105752192400406X](https://www.sciencedirect.com/science/article/abs/pii/S105752192400406X)

2. **Design choice variance exceeds statistical error** — A study fitting >1,000 ML models for stock returns found that the "non-standard error" in portfolio returns arising from design choices (algorithm, target variable, feature selection, training methodology) exceeds the standard error by 59%. This means that two researchers with the same data can get wildly different results based on arbitrary choices. [Source: SSRN 5031755](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5031755)

3. **Train-test gap control via SWA** — The TIPS ablation shows that vanilla distillation has a large train-test SR gap. Adding low-temperature distillation widens the gap further. Aggressive label smoothing reduces it, and SWA yields the strongest and most consistent improvement while closing the gap. [Source: arxiv 2603.16985v2](https://arxiv.org/html/2603.16985v2)

4. **Posterior drift in overparametrized models** — Overparametrized models suffer from posterior drift where the relationship learned in training degrades out-of-sample. Applied to equity premium forecasting, results show sensitivity of market timing strategies to sub-periods and bandwidth parameters. [Source: arxiv 2506.23619](https://arxiv.org/pdf/2506.23619)

5. **In-sample standardization in RFF creates spurious kernel approximations** — A theoretical paper proves that within-sample standardization in Random Fourier Features implementations alters the underlying kernel approximation, replacing shift-invariant kernels with training-set dependent ones, contributing to the train-test gap. [Source: arxiv 2506.03780](https://arxiv.org/pdf/2506.03780)

### 4. Simpler Models Beat Complex Models in Finance

1. **Feature engineering beats LSTM (2026)** — Kang shows a parsimonious linear model with market cap-normalized flows achieves Sharpe 1.30 and 272.6% cumulative return, while the full ICA-Wavelet-LSTM pipeline delivers Sharpe 0.07 and -5.1% return. The LSTM collapsed to predicting the unconditional mean (hit rate 47.5%). [Source: arxiv 2601.07131](https://arxiv.org/html/2601.07131)

2. **SVM outperforms LSTM/GRU in crypto (2026)** — Kaygın et al. evaluate SVM, LSTM, GRU, and hybrids across 5 major cryptocurrencies (2020–2025). SVM achieves the lowest errors across most assets; the complex GRU+LSTM hybrid is the worst performer (BTC MAPE: SVM 0.55% vs GRU+LSTM 4.91%). [Source: Mathematics 2026, 14(6), 989](https://www.mdpi.com/2227-7390/14/6/989)

3. **Linear models on short windows beat complex models on long windows during recessions** — Capponi et al. (2025) demonstrate that during NBER recessions, ridge regression on 64 months of data consistently outperforms random forests on all historical data. During the 1990 Gulf War recession, the simple model achieves positive R² while complex models are negative. [Source: arxiv 2512.23596](https://arxiv.org/html/2512.23596v1)

4. **"Simplified: A Closer Look at the Virtue of Complexity" (2025)** — Re-examines Kelly, Malamud, and Zhou's "Virtue of Complexity" claim and shows that the finding of strictly increasing portfolio performance with complexity is driven by look-ahead bias and incorrect confidence bands. After correction, simpler models are not dominated. [Source: SSRN 5239006](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5239006)

5. **Classical architectures beat Transformers on financial data** — The TIPS paper benchmark shows that GRU, LSTM, and TCN substantially outperform generic time-series SOTA Transformers (iTransformer, PatchTST, AutoFormer) on financial forecasting tasks across 4 markets, despite having far fewer parameters. [Source: arxiv 2603.16985v2](https://arxiv.org/html/2603.16985v2)

6. **DLinear/NLinear competitive with complex models** — The Oxford benchmark confirms that simple linear models (DLinear, NLinear) occasionally perform competitively in specific high-volatility subperiods, though they lack consistency across regimes. VLSTM (LSTM + VSN) provides the best overall balance. [Source: arxiv 2603.01820](https://arxiv.org/html/2603.01820)

7. **iTransformer achieves extremely low turnover but weak performance** — The lowest-turnover model (iTransformer, turnover=36) trades least but has Sharpe 0.35 — demonstrating that under-reactive models destroy economic value just as overfitting does. [Source: arxiv 2603.01820](https://arxiv.org/html/2603.01820)

### 5. Walk-Forward Validation Best Practices

1. **ATOMS: Adaptive Tournament Model Selection (2025)** — A novel framework that jointly optimizes model class and training window size using a tournament procedure with adaptive validation window selection. Outperforms fixed-window baselines by 14–23% improvement in OOS R² on 17 industry portfolios. The key insight: during recessions, shorter validation windows are optimal; during stable periods, longer windows work better. [Source: arxiv 2512.23596](https://arxiv.org/html/2512.23596v1)

2. **Purged K-Fold Cross-Validation** — Standard k-fold CV leaks information across folds in time series. The correct approach is purged k-fold: purge overlapping observations and embargo periods to prevent train-test contamination. This is widely recommended across trading blogs and institutional frameworks. [Source: Technical Analysis Pro](https://www.technical-analysis-pro.com/strategies-ai-backtesting-walk-forward-model-validation/)

3. **Walk-forward with purging** — López de Prado's framework: never use future information for normalization or feature construction; compute all transformations within each walk-forward fold independently. Normalization calculated on the full dataset is the most common form of lookahead bias. [Source: barmenteros FX](https://barmenteros.com/machine-learning-trading-backtesting/)

4. **34 independent test periods** — One framework enforces strict information set discipline with rolling window validation across 34 independent test periods, maintaining complete interpretability. [Source: arxiv 2512.12924](https://arxiv.org/pdf/2512.12924)

5. **Volatility targeting and breakeven transaction cost analysis** — The Oxford benchmark demonstrates proper backtesting protocol: volatility target at 10%, train on gross returns (cost=0), then compute per-asset breakeven transaction costs post-hoc. Models are retrained every 5 years with rolling windows. [Source: arxiv 2603.01820](https://arxiv.org/html/2603.01820)

6. **HAC-adjusted t-statistics for statistical significance** — Use Newey-West heteroskedasticity and autocorrelation consistent (HAC) standard errors when testing strategy returns for significance. Critical in financial time series where returns are autocorrelated. [Source: arxiv 2603.01820](https://arxiv.org/html/2603.01820)

7. **Multiple random seed evaluation** — Run each configuration with 5+ random seeds and aggregate top-N by validation performance. The Oxford benchmark uses 50 seeds (top 10) for primary results, validates with 25 seeds (top 5) for robustness. [Source: arxiv 2603.01820](https://arxiv.org/html/2603.01820)

---

## Specific Technique Recommendations

### For Your Project (RecurrentCTM + Ensemble on Financial Data):

| Technique | Priority | Implementation Guidance |
|-----------|----------|------------------------|
| **Sharpe/directional loss** | HIGH | Replace MSE with GMADL or direct Sharpe optimization (differentiable via pooled returns). This aligns training with trading objectives. |
| **Variable Selection Networks** | HIGH | Add VSN before temporal encoder to adaptively filter noisy features. VLSTM showed best overall Sharpe (2.39). |
| **Adaptive training window** | MEDIUM | Implement ATOMS-like adaptive window selection: jointly tune model and window size on validation data, shrink windows during volatile regimes. |
| **Walk-forward with purging** | HIGH | Ensure normalization, feature construction, and class weights are computed within each fold only. Use purged k-fold or rolling walk-forward. |
| **Seed ensemble** | HIGH | Train 5–25 seeds, select top-N by validation Spearman IC (not Sharpe, which is gameable), average their position signals. |
| **Label smoothing + SWA** | MEDIUM | If using distillation or ensemble, apply aggressive label smoothing (ε=0.5–0.9) + SWA to reduce train-test gap. |
| **HAC significance testing** | MEDIUM | Report Newey-West t-statistics for all strategy returns. |
| **Breakeven cost analysis** | LOW | Compute per-asset breakeven transaction costs to identify which assets survive realistic frictions. |
| **Early stopping on Spearman IC** | HIGH | Already implemented — this is correct. Spearman IC is more robust than Sharpe for early stopping. |
| **Dropout + weight decay** | HIGH | Already implemented (p=0.1, wd=0.05) — appropriate for financial data. Consider temporally adaptive dropout. |

---

## Sources

### Kept:
- **Oxford Benchmark** (Saly-Kaufmann et al., 2026, arxiv 2603.01820) — Most comprehensive DL benchmark for financial time series; 15 years, 15+ architectures, risk-adjusted evaluation with seed robustness.
- **The Limits of Complexity** (Kang, 2026, arxiv 2601.07131) — Strongest empirical evidence that feature engineering beats deep learning; LSTM collapse diagnostic.
- **Nonstationarity-Complexity Tradeoff** (Capponi et al., 2025, arxiv 2512.23596) — Theoretical + empirical framework for joint model/window selection; ATOMS algorithm.
- **TIPS** (KDD 2026, arxiv 2603.16985v2) — Knowledge distillation framework; merging penalty phenomenon; aggressive regularization closing train-test gap.
- **GMADL Loss** (Michańków et al., 2024, arxiv 2412.18405) — Differentiable directional loss function; outperforms MSE across architectures.
- **SVM vs DL in Crypto** (Kaygın et al., 2026, Mathematics 14(6):989) — Empirical evidence that SVM beats LSTM/GRU in volatile crypto markets.
- **Chaos, Overfitting and Equilibrium** (2024, ScienceDirect) — Demonstrates in-sample 100% accuracy → OOS 50% accuracy collapse.
- **Design Choices and Non-Standard Error** (2024, SSRN 5031755) — Design choice variance 59% larger than standard error.
- **Simplified: Virtue of Complexity Re-examined** (2025, SSRN 5239006) — Shows complexity advantage is partly methodological artifact.
- **Backtest Overfitting in ML Era** (Arian et al., 2024, SSRN 4686376) — Comprehensive comparison of OOS testing methods.
- **Walk-Forward Analysis Guide** (StratBase, ARIA Analyst, Falco Insights) — Practical implementation guides for walk-forward validation.
- **Temporally Adaptive Dropout LSTM** (2026, Springer) — Dropout probabilities adapt per timestep based on actual vs predicted differences.

### Dropped:
- **DELPHYNE** (arxiv 2506.06288) — Pre-trained model for financial time series; not directly about overfitting.
- **Financial Fine-tuning Large Time Series Model** (arxiv 2412.09880) — TimesFM application; limited overfitting relevance.
- **RLSTM (2021)** — Older paper; random noise for overfitting prevention; superseded by newer work.
- **Meegle.com overview** — Non-academic source; useful background but not primary.
- **Overparametrized models with posterior drift** (arxiv 2506.23619) — Relevant but results less actionable than other sources.

---

## Gaps

1. **No consensus on optimal regularization strength** — Papers use different dropout rates (0.1–0.2), weight decay values (0.05–0.15), and label smoothing coefficients (0.5–0.9). Optimal values appear data- and architecture-dependent.
2. **Limited cross-asset class validation** — Most papers focus on equities; crypto and futures are covered but fixed income and FX are underrepresented.
3. **No standardized overfitting benchmark** — Unlike ImageNet for vision, there is no agreed-upon financial overfitting benchmark. The Oxford benchmark is the closest attempt.
4. **Transaction cost modeling absent from most regularization studies** — Only GMADL and the Oxford benchmark explicitly address transaction costs in the training objective.
5. **Computational constraints** — The ATOMS algorithm requires training many candidate models (model_class × window_size), which may be computationally prohibitive for large-scale experiments. Warm-starting from previous windows could mitigate this.

## Suggested Next Steps for Glaubenskrieg Project

1. Replace MSE/MAE loss with GMADL or direct Sharpe optimization
2. Add VSN before Mamba/CTM encoder for adaptive feature selection
3. Implement purged walk-forward validation (normalize features within each fold)
4. Run 5–25 seeds per configuration and ensemble top-N
5. Report HAC-adjusted t-statistics and seed sensitivity analysis
6. Consider adaptive training window selection based on recent market volatility
