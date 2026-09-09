# Research: Why Tree-Based Models Outperform Deep Learning in Financial Prediction

## Summary

Gradient-boosted decision trees (GBDTs: XGBoost, LightGBM, CatBoost) consistently match or outperform deep learning models (LSTM, Transformer, Mamba) on financial prediction tasks involving tabular features—particularly daily return forecasting. Three structural reasons explain this: (1) financial data has severe low signal-to-noise ratios (~80% noise) that cause high-capacity deep models to memorize noise; (2) financial prediction with engineered features is fundamentally a tabular problem where tree-based inductive biases (threshold splits, robustness to uninformative features, irregular function learning) provide a natural fit; and (3) typical datasets are too small (hundreds to low thousands of samples) for deep learning to generalize, while GBDTs regularize well. Hybrid architectures (CNN → LightGBM, LSTM → GBDT ensembles) consistently outperform either paradigm alone, but the tree component remains the dominant performance driver.

---

## Findings

### 1. The Tabular Data Gap: Trees Outperform NNs on Structured Data

Grinsztajn, Oyallon & Varoquaux (2022) conducted the most rigorous controlled study, benchmarking 7 algorithms on 45 datasets with 20,000 compute-hours of hyperparameter tuning. They found tree-based models (XGBoost, Random Forest, Gradient Boosting) remain state-of-the-art on medium-sized tabular data (~10K samples) even after exhaustive hyperparameter optimization of NNs (MLP, ResNet, FT-Transformer, SAINT). Three inductive biases explain the gap **[Grinsztajn et al., 2022]**:

- **Smoothness bias**: NNs are biased toward low-frequency (smooth) functions per the spectral bias of gradient descent (Rahaman et al., 2019). When the target function has irregular, non-smooth patterns—as financial data invariably does—trees learn piecewise-constant boundaries directly while NNs struggle. They empirically demonstrated this by Gaussian-smoothing targets: tree performance plummeted (irregular patterns are their advantage), NN performance was barely affected.
- **Uninformative feature robustness**: Tabular datasets contain many uninformative features. MLP-like architectures degrade significantly as noise features increase, while tree-based models and FT-Transformer maintain performance through feature selection at split time.
- **Rotation invariance harms NNs**: MLPs and ResNets are rotationally invariant—they treat all linear combinations of features equally. But tabular features have meaning individually (e.g., "volume," "RSI"), and mixing them destroys this structure. Randomly rotating features reversed the performance order (NNs > trees), confirming that rotation invariance is actively harmful. Decision trees' axis-aligned splits are a better inductive bias for this data structure.

### 2. Financial Data Has Critical Properties Hostile to Deep Learning

Financial time series has three structural properties that specifically disadvantage deep learning **[D&T Systems, 2024]**:

**(a) Low Signal-to-Noise Ratio**: A typical daily price series is roughly 80% noise. The predictable component is small and unstable. Deep models with millions of parameters are extremely effective at memorizing noise—the opposite of what is needed for out-of-sample generalization. GBDTs, with their built-in regularization (shallow depth, subsampling, leaf constraints), naturally resist noise-fitting.

**(b) Non-Stationarity**: The distribution shifts constantly. What predicted direction during a trending regime in 2021 may predict the opposite in a mean-reverting regime in 2023. Trees handle regime changes through feature-threshold splits that are relatively stable as distributions shift; NNs learn distributed representations that are more brittle to distributional changes.

**(c) Data Scarcity**: 3 years of daily bars = ~750 samples. With sequence length 20, that's ~730 training sequences. A conservative one-layer LSTM with 64 hidden units has ~21,500 parameters in the recurrent layer alone—yielding <0.03 samples per parameter. Deep learning was designed for millions of examples. XGBoost, by contrast, regularizes well at ~500–2,000 rows via learning rate, max depth, and subsampling **[D&T Systems, 2024]**.

Research on weak signals (SSRN, 2024) confirms that "Random Forest generally outperforms Gradient Boosted Regression Trees when signals are weak," while both tree families outperform neural networks without heavy regularization **[Can Machines Learn Weak Signals?, 2024]**.

### 3. Financial Prediction Is Fundamentally Tabular, Not Sequential

By the time features enter a model, the temporal work has already been done. RSI(14) is a 14-bar aggregation. ATR(20) is a 20-bar aggregation. Each feature compresses a time window into a single number. The model receives a row of numbers—one value per feature, one row per bar. This is a tabular dataset, not a sequence modeling problem **[D&T Systems, 2024]**. LSTM's ability to model sequential dependencies provides no benefit when sequence information is already baked into the features. The model pays the parameter cost without the benefit.

The FinMamba paper explicitly acknowledges this: "stock prices often lack regular repetitive patterns due to low signal-to-noise ratios, with similar patterns appearing at different times but under different market regimes" **[FinMamba, 2025]**. This differs from typical deep learning time-series domains (solar energy, traffic flow) where repetitive patterns are extractable.

### 4. Empirical Evidence in Financial Prediction Specifically

**(a) 15-Year S&P 500 Comparison (2025)**: A comprehensive comparison of Linear Regression, Lasso, Random Forest, XGBoost, and LSTM on 12 S&P 500 equities over 15 years found that "despite the complexity of advanced models, simple linear and decision-tree based approaches often outperform deep learning" and "all models struggle to consistently surpass basic baselines due to the efficient and noisy nature of financial markets" **[15-Year Empirical Comparison, 2025]**.

**(b) CNN-LightGBM Hybrid > All Standalone Models (2026)**: A carefully controlled study on NIFTY50 next-day log-return forecasting compared CNN, LightGBM, LSTM-LightGBM, TimesNet, and CNN-LightGBM under a strictly chronological no-look-ahead protocol. CNN-LightGBM achieved the lowest RMSE (0.008334) and highest R² (0.0277), significantly outperforming all competitors including standalone LightGBM (p<0.01, Diebold-Mariano). Crucially, standalone LightGBM significantly outperformed the zero-return baseline at p<0.10, while standalone CNN could not **[CNN-LightGBM Hybrid, MDPI Symmetry, 2026]**. The ablation study revealed that wavelet denoising and embedding standardization were critical to the hybrid's success—removing denoising dropped OOS-R² from 0.0285 to 0.0021 (non-significant).

**(c) GBDT+LSTM Ensemble (2025)**: A stacking ensemble of CatBoost + LightGBM + LSTM improved accuracy 10–15% over individual models on S&P 500 data. Notably, standalone CatBoost (R²=0.7882) dramatically outperformed standalone LSTM (R² was not even reported as competitive) **[GBDT+LSTM Ensemble, arXiv 2025]**.

**(d) Gu, Kelly & Xiu (2020)**: In their landmark "Empirical Asset Pricing via Machine Learning," trees and neural nets performed comparably for measuring the equity risk premium, with neural nets benefiting most from very large numbers of predictors and trees performing better with moderate feature sets—a finding that aligns with GBDTs' robustness to uninformative features **[Gu, Kelly & Xiu, 2020]**.

### 5. Mamba/SSM Models: Promising but Not Yet Competitive with GBDTs

MambaStock (2024) claims to outperform XGBoost, LSTM, BiLSTM, Transformer, and hybrid models on 4 Chinese bank stocks **[MambaStock, 2024]**. However, several methodological concerns exist:

- **Price prediction vs. return prediction**: MambaStock predicts stock prices directly (not returns), a task with extremely high autocorrelation (price at t+1 ≈ price at t). This inflates R² scores dramatically (the paper reports R² near 0.99 on some stocks) and makes comparisons misleading for the return prediction task.
- **Small evaluation scope**: Only 4 stocks, all Chinese banks, tested on a fixed 300-day test window.
- **No XGBoost comparison in the primary benchmark tables**: The comparison tables show results vs. ARIMA, KF, ARIMA-NN, LSTM, BiLSTM, Transformer, TL-KF, and AttCLX—but XGBoost is mentioned as a baseline only in the method description, with no explicit tabular comparison.

The FinMamba (2025) paper acknowledges the fundamental challenge: "stock prices often lack regular repetitive patterns due to low signal-to-noise ratios" and proposes a graph-enhanced architecture combining inter-stock relationships with Mamba's temporal modeling. This is a more realistic framing, but it recognizes that Mamba alone is insufficient for financial prediction without carefully constructed auxiliary structure **[FinMamba, 2025]**.

Other Mamba variants for finance (Mamba Meets Financial Markets, CMDMamba, AG-STFT+Mamba) all remain proof-of-concept studies without rigorous GBDT baselines or proper walk-forward validation.

### 6. The Meta-Learning Exception: When NNs Can Win

McElfresh et al. (2023) conducted the largest tabular data analysis to date (19 algorithms, 176 datasets, 538K models trained) and found that GBDTs outperform NNs on datasets that are more "irregular" (skewed/heavy-tailed feature distributions), larger, and have high ratio of size to features. However, they also noted that **TabPFN**—a prior-data fitted network—achieves near-top performance across all datasets, "despite only seeing part of the training dataset for many datasets" **[McElfresh et al., 2023, NeurIPS 2023]**. This suggests that the key advantage of GBDTs is not fundamental superiority of trees, but rather that standard NNs lack the right inductive biases for tabular data. Meta-learned priors (TabPFN) or carefully designed architectures may close the gap.

### 7. Practical Takeaways: The Recommended Workflow

Based on the consensus across studies:

1. **Start with linear models** (Ridge/Lasso). If the signal cannot be captured by a regularized linear model, no deep model will help—find better features.
2. **Use XGBoost/LightGBM with Optuna tuning** and walk-forward cross-validation (5+ folds, ≥20% OOS per fold). This is the pragmatic baseline that succeeds on most financial prediction tasks.
3. **Add deep learning only when**: (a) >50K samples, (b) raw or minimally processed features with genuine sequential structure, and (c) XGBoost accuracy has plateaued across multiple tuning trials **[D&T Systems, 2024]**.
4. **Consider hybrid architectures** (CNN/Transformer encoder → GBDT regressor) when representation learning can complement tree-based final prediction. The CNN-LightGBM hybrid showed consistent improvements over both standalone models and simple averaging baselines, with statistical significance **[CNN-LightGBM, 2026]**.
5. **Never skip walk-forward validation with strict chronology**. A single train/test split on financial data is almost always misleading. Deep models are particularly susceptible to overfitting look-ahead leakage artifacts **[CNN-LightGBM, 2026]**.

---

## Sources

### Kept

- **Grinsztajn, Oyallon & Varoquaux (2022)** — "Why do tree-based models still outperform deep learning on tabular data?" NeurIPS 2022 Datasets & Benchmarks. [arXiv:2207.08815](https://arxiv.org/abs/2207.08815) — The canonical empirical study identifying three inductive bias reasons for GBDT superiority. 45 datasets, 20,000 compute-hours.
- **McElfresh et al. (2023)** — "When Do Neural Nets Outperform Boosted Trees on Tabular Data?" NeurIPS 2023. [arXiv:2305.02997](https://arxiv.org/abs/2305.02997) — Largest tabular benchmark (176 datasets, 19 algorithms, 538K models). Identifies dataset irregularity as key predictor of GBDT superiority.
- **CNN-LightGBM Hybrid (2026)** — "Strictly Chronological CNN Embeddings with Gradient-Boosted Trees for Next-Day Log-Return Forecasting." MDPI Symmetry. [DOI](https://www.mdpi.com/2073-8994/18/3/416) — The most rigorously controlled financial prediction comparison. Shows CNN-LightGBM > LightGBM > LSTM-LightGBM > CNN for daily return forecasting with statistical significance.
- **D&T Systems Blog (2024)** — "XGBoost Beats LSTM and Transformers on Most Financial Time Series." [Link](https://dtsystems.dev/blog/xgboost-vs-lstm-financial-time-series) — Practitioner perspective explaining data math: 730 training sequences vs. 33,000 LSTM parameters.
- **15-Year Empirical Comparison (2025)** — "A 15-Year Empirical Comparison of Deep Learning, Tree-Based, and Regression Models for Stock Market Forecasting." [DOI:10.58445/rars.3334](https://research-archive.org/index.php/rars/preprint/view/3334) — 12 S&P 500 equities, 15 years; simple models often outperform deep learning.
- **GBDT+LSTM Ensemble (2025)** — "Gradient Boosting Decision Tree with LSTM for Investment Prediction." [arXiv:2505.23084](https://arxiv.org/abs/2505.23084) — Stacking ensemble shows standalone CatBoost (R²=0.788) dramatically outperforms standalone LSTM.
- **FinMamba (2025)** — "FinMamba: Market-Aware Graph Enhanced Multi-Level Mamba for Stock Movement Prediction." [arXiv:2502.06707](https://arxiv.org/abs/2502.06707) — Acknowledges low signal-to-noise ratio and lack of regular repetitive patterns as key challenges for deep sequence models in finance.
- **MambaStock (2024)** — "MambaStock: Selective State Space Model for Stock Prediction." [arXiv:2402.18959](https://arxiv.org/abs/2402.18959) — First Mamba financial application. Claims superiority but on price prediction (high autocorrelation task), not return prediction.
- **Gu, Kelly & Xiu (2020)** — "Empirical Asset Pricing via Machine Learning." Review of Financial Studies. [DOI](https://doi.org/10.1093/rfs/hhaa009) — Landmark study showing trees and NNs perform comparably for risk premium measurement; trees perform better with moderate feature sets.
- **Can Machines Learn Weak Signals? (2024)** — SSRN. [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4722678) — Random Forest outperforms GBRT when signals are very weak; NNs need L2 regularization to be competitive.
- **Shwartz-Ziv & Armon (2021)** — "Tabular Data: Deep Learning is Not All You Need." Information Fusion. [DOI](https://doi.org/10.1016/j.inffus.2021.11.011) — GBDTs perform better on average; ensembling GBDTs + NNs achieves best performance.

### Dropped

- **ITM Web of Conferences (2026)** — "Stock prediction by means of XGBoost, LSTM, and Transformer" — Single stock (NVDA), no walk-forward validation, price prediction (not return prediction). Limited methodological rigor.
- **MDPI Risks (2025)** — "A Comparative Study of Transformer-Based and Classical Models" — Uses future CPI/GDP/policy rate in features, creating look-ahead bias for daily prediction.
- **SCITEPRESS (2025)** — "LSTM Significantly Outperforms XGBoost" — Only 4 tech stocks, short time period, likely look-ahead bias in feature engineering.
- **E3S Conferences (2021)** — "Stock Price Prediction Based on XGBoost and LightGBM" — High-frequency trading price prediction with severe data leakage (uses same-day volume and turnover to predict same-day price).
- **arXiv 2410.03707** — Graph-Mamba for stock price prediction — Proof-of-concept without rigorous GBDT baselines.

---

## Gaps

1. **No apples-to-apples Mamba vs. GBDT comparison on return prediction**: Existing Mamba financial papers predict prices (easy, high autocorrelation) rather than returns. A rigorous comparison of Mamba variants against XGBoost/LightGBM on daily return forecasting with walk-forward validation and Diebold-Mariano testing does not exist in the literature.

2. **Limited multi-asset studies**: Most comparative studies use single stocks or indices. The scaling behavior when moving from 1 asset to 200+ assets (as in the Glaubenskrieg project) is not well-studied. Cross-asset correlation and the O(N²) attention cost may fundamentally change the comparison landscape.

3. **Mamba for finance is still in proof-of-concept stage**: All Mamba financial papers (MambaStock, FinMamba, Mamba Meets Financial Markets, CMDMamba) were published 2024-2025 and focus on architecture proposals rather than rigorous benchmarking against well-tuned GBDTs.

4. **The role of data volume**: McElfresh et al. (2023) show the NN-GBDT gap narrows with larger datasets (50K vs 10K samples), but financial return prediction rarely has 50K+ samples without using intraday data. The point at which NNs become competitive for daily return prediction is unknown.

5. **Ensemble weighting strategies**: Both GBDT+LSTM (2025) and CNN-LightGBM (2026) show hybrid models outperform either paradigm alone, but optimal ensemble architectures and the relative contribution of representation learning vs. tree regression remain underexplored.

---

## Implications for Glaubenskrieg Project

The literature strongly supports the current project architecture direction:

1. **The tree component (GBDT) should remain the primary prediction engine**. Across all rigorous studies, GBDTs match or outperform deep learning on daily return forecasting. The D&T Systems blog's warning is apt: "Complexity is added only when the simpler step reaches its ceiling"—and for daily return prediction with ~750 samples, trees are nowhere near their ceiling.

2. **Hybrid architectures (encoder → GBDT) are the most promising research direction**. The CNN-LightGBM paper demonstrates that a lightweight encoder with explicit embedding standardization can provide statistically significant improvement over standalone GBDT. This aligns with the Glaubenskrieg project's use of MambaBlock as an encoder feeding into a GBDT or ensemble head.

3. **Mamba/SSM as a feature encoder is a reasonable bet**, but expectations should be calibrated. No paper has demonstrated Mamba outperforming XGBoost on return prediction with proper evaluation protocol. The value of Mamba in this context likely comes from its ability to learn a compact state representation of market dynamics—not from raw predictive accuracy.

4. **Walk-forward validation and strict chronology are non-negotiable**. The CNN-LightGBM paper's methodology (expanding walk-forward, training-only statistics, Diebold-Mariano testing) is the gold standard. The Glaubenskrieg project's use of `shuffle=False` for test evaluation and strict temporal splits aligns with best practices.

5. **The law of small numbers dominates**: With ~750 training samples (3 years daily), the inductive biases of the model matter far more than its capacity. This is the core reason GBDTs win. Any deep learning architecture for this task must have explicit mechanisms to resist noise-fitting (dropout, weight decay, early stopping, and most importantly, tight capacity constraints).
