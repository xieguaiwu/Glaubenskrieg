# Research: Theoretical Reasons GBDTs Outperform Deep Learning on Financial Tabular Data

## Summary

Gradient boosted decision trees (GBDTs) outperform neural networks on financial tabular data due to a confluence of theoretical factors: (1) trees' piecewise-constant inductive bias naturally fits the irregular, non-smooth target functions prevalent in financial data, while neural networks exhibit a spectral bias toward low-frequency/smooth solutions; (2) trees handle uninformative features via axis-aligned splits that implicitly perform feature selection, while MLPs pay a linear sample-complexity penalty for rotational invariance; (3) GBDTs are vastly more sample-efficient, regularizing well on the ~750–2000 samples typical of daily financial data where deep networks overfit; (4) under squared loss with near-martingale price processes, the Bayes-optimal predictor collapses to a trivial constant, and highly expressive models (Transformers) amplify noise through interpolation rather than extracting signal; (5) financial time series' non-stationarity and regime shifts are better handled by trees' threshold-based splits and rapid retrainability, while deep networks' distributed representations smear across regimes.

---

## Findings

### 1. Inductive Bias: Trees Learn Irregular Functions, Neural Nets Are Biased Toward Smoothness

**The spectral bias of neural networks.** Rahaman et al. (2019) demonstrated that neural networks learn low-frequency functions first—a phenomenon known as spectral bias. Grinsztajn et al. (2022) confirmed this empirically for tabular data: smoothing the target function on the training set (via Gaussian kernel smoothing) barely impacts neural network performance, but *markedly degrades* tree-based model accuracy. This reveals that real-world tabular target functions are *not smooth*—they contain irregular, high-frequency patterns that trees capture natively but NNs struggle to learn. [Source](https://ar5iv.labs.arxiv.org/html/2207.08815)

**Trees as piecewise-constant approximators.** Decision trees learn piecewise-constant functions through axis-aligned splits. This inductive bias is a *feature, not a bug* for financial data: it directly models discontinuous threshold effects (e.g., "funding rate crosses zero → regime flip"). Neural networks, by contrast, construct smooth interpolations that smear across such boundaries. [Source](https://dtsystems.dev/blog/xgboost-vs-lstm-financial-time-series)

**Financial relevance.** Financial markets exhibit sharp regime boundaries—trending vs. mean-reverting, high-vol vs. low-vol, risk-on vs. risk-off. These are naturally represented as threshold conditions on features (RSI > 70, VIX > 30, etc.), which axis-aligned splits capture in a single node. A neural network must learn to approximate a step function through compositions of smooth activations, requiring far more parameters and data. [Source](https://staff.fnwi.uva.nl/a.khedher/winterschool/19ReynersPaper.pdf)

### 2. Rotational Invariance Is Harmful for Tabular Data

**The Ng (2004) result.** A rotationally invariant learning procedure—one whose behavior is unchanged when a unitary matrix is applied to features—has worst-case sample complexity that grows *at least linearly* in the number of irrelevant features. Standard MLPs are rotationally invariant; tree-based models are not. [Source](https://dl.acm.org/doi/10.5555/3600270.3600307)

**Why this matters.** Tabular features carry individual meaning (e.g., "volume," "spread," "PE ratio"). Their original orientation encodes this semantic structure. When an MLP mixes features through dense linear transformations, it must first "rediscover" the original basis before it can identify which features are informative—a statistically expensive operation. Tree-based splits operate directly on individual features, preserving the natural basis. [Source](https://ar5iv.labs.arxiv.org/html/2207.08815)

**Empirical confirmation.** Grinsztajn et al. showed that randomly rotating features *reverses* the performance order: ResNets (rotationally invariant) now outperform tree-based models, because rotation has destroyed the natural feature basis that trees rely on. This confirms that rotation invariance is *undesirable* for tabular data. [Source](https://ar5iv.labs.arxiv.org/html/2207.08815)

### 3. Robustness to Uninformative Features

**Tabular datasets contain many uninformative features.** In financial contexts, practitioners often include dozens or hundreds of candidate features (technical indicators, macro variables, sentiment scores). Many are noise. Grinsztajn et al. showed that removing up to 50% of features ranked by Random Forest importance barely affects GBDT accuracy, while the removed features themselves carry almost no predictive signal—they are truly uninformative, not merely redundant. [Source](https://ar5iv.labs.arxiv.org/html/2207.08815)

**MLP-like architectures suffer disproportionately.** Adding random Gaussian noise features (uncorrelated with target) widens the performance gap between MLPs/ResNets and tree-based models + FT-Transformers. Removing uninformative features narrows the gap. This shows MLPs are uniquely vulnerable to irrelevant features—they allocate modeling capacity to noise because their dense connections cannot easily "ignore" inputs. Trees perform implicit feature selection at every split: if a feature doesn't help reduce impurity, it is simply not used. [Source](https://ar5iv.labs.arxiv.org/html/2207.08815)

### 4. Sample Efficiency: The 730-Samples-vs-33,000-Parameters Problem

**The data math.** Three years of daily bars yields ~750 samples. A conservative LSTM with 64 hidden units and 20 input features has ~33,000 parameters. The ratio is 0.02 samples per parameter. GBDTs with tuned depth (max_depth=3–6, ~100–200 leaves) have 500–2,000 effective degrees of freedom, yielding 0.4–1.5 samples per parameter—a 20–75× improvement in sample efficiency. [Source](https://dtsystems.dev/blog/xgboost-vs-lstm-financial-time-series)

**GBDT regularization mechanisms.** Trees regularize through multiple mechanisms operating simultaneously: (a) learning rate (shrinkage) prevents any single tree from overfitting; (b) subsampling of rows (stochastic gradient boosting) and columns introduces randomness that acts as a variance-reduction technique; (c) max_depth caps model complexity; (d) minimum samples per leaf prevents fitting to outliers. These compound to produce strong generalization even at very small sample sizes. [Source](https://staff.fnwi.uva.nl/a.khedher/winterschool/19ReynersPaper.pdf)

**McElfresh et al. (2023) large-scale study.** Across 176 datasets and 19 algorithms, GBDTs (CatBoost, XGBoost) consistently outperformed neural networks on larger datasets and datasets with high feature-to-sample ratios. GBDTs particularly excelled when the dataset was "irregular"—featuring skewed, heavy-tailed, or high-variance feature distributions. [Source](https://arxiv.org/html/2305.02997v4)

### 5. Why Attention Fails on Financial Time Series

**Theoretical explanation 1: Asymmetric Learning (Ke et al., 2025).** When the sign of the previous step's residual is inconsistent with the current step's sign (the norm in financial returns), attention networks fail to learn residual features. This is because the attention mechanism's kernel structure induces *asymmetric learning*—it can model one direction of dependence but fails when sign patterns flip unpredictably. A simple linear residual network easily handles this, but attention cannot. [Source](https://proceedings.mlr.press/v280/ke25a.html)

**Theoretical explanation 2: Forecast Collapse under Squared Loss (2025).** Under squared trajectory loss, the Bayes-optimal predictor is the conditional mean of the future given the past. For financial price processes that are approximately martingales (𝔼[X_{t+1}|ℱ_t] = X_t), the Bayes-optimal forecast collapses to a constant: flat for prices, zero for returns. When a highly expressive model (Transformer) is trained via empirical risk minimization in this regime, it does not uncover hidden structure—it *amplifies noise through interpolation*. The model's additional expressivity creates spurious fluctuations around the trivial optimal predictor, increasing variance without reducing bias. The paper proves that the interpolating predictor's MSE is ≥ 2Hσ² (double the irreducible error), while a simple linear predictor achieves Hσ² + O(1/n). [Source](https://ar5iv.labs.arxiv.org/html/2604.00064)

**Empirical confirmation on FX data.** Testing PatchTST on EUR/USD 30-second data: the Transformer-based predictor produced larger trajectory errors than a linear model on ~92% of forecasting windows, with an average error ratio of 1.71×. The degradation was *pervasive across the distribution*, not driven by outliers. [Source](https://ar5iv.labs.arxiv.org/html/2604.00064)

### 6. Non-Stationarity and Distribution Shift

**Financial data is fundamentally non-stationary.** Market regimes change: what predicted direction in a 2021 trending regime may predict the opposite in a 2023 mean-reverting regime. Feature distributions, correlations, and target-conditioning all drift over time. [Source](https://dtsystems.dev/blog/xgboost-vs-lstm-financial-time-series)

**Why trees are more robust to distribution shift:**

- **Axis-aligned splits are monotonic transforms of individual features.** A split like "RSI > 70" remains meaningful even if the joint distribution of all features shifts, as long as the univariate conditional relationship holds approximately. Neural networks' distributed representations entangle features, so a shift in one feature's marginal distribution can corrupt the entire learned representation.

- **Rapid retraining.** XGBoost trains in <30 seconds on CPU for typical daily datasets. This enables weekly or even daily retraining as new data arrives—a practical hedge against non-stationarity. Deep networks require GPU training runs of minutes to hours, making frequent retraining costlier. [Source](https://dtsystems.dev/blog/xgboost-vs-lstm-financial-time-series)

- **Boosting's sequential nature as implicit adaptation.** Each new tree in gradient boosting fits the residual of the current ensemble. This means later trees automatically adapt to recent patterns that earlier trees missed—providing a form of online adaptation without explicit continual learning machinery.

**McElfresh et al. metafeature analysis.** GBDTs outperformed NNs specifically on datasets with irregular feature distributions: high skewness, high kurtosis, high variance. Financial data is archetypally irregular in this sense—featuring heavy tails, volatility clustering, and extreme events that produce highly non-Gaussian feature distributions. [Source](https://arxiv.org/html/2305.02997v4)

### 7. The "Deep Learning Not Suitable for Tabular Data" Debate: Current Consensus

**Shwartz-Ziv & Armon (2021).** Tested 4 deep tabular models (TabNet, NODE, DNF-Net, 1D-CNN) against XGBoost on 11 datasets. Found XGBoost outperformed deep models on 8/11 datasets, with the deep models generalizing poorly to datasets outside their original papers. An ensemble of XGBoost + deep models performed best overall, but XGBoost alone was the strongest single model. [Source](https://ar5iv.labs.arxiv.org/html/2106.03253)

**McElfresh et al. (2023) — the largest study to date.** 19 algorithms, 176 datasets, 538,650 trained models. Key conclusions: (a) the "NN vs. GBDT" debate is overemphasized—for ~1/3 of datasets, light hyperparameter tuning on CatBoost yields more improvement than switching between NNs and GBDTs; (b) GBDTs outperform NNs on irregular, large, and high-dimensional datasets; (c) no single algorithm dominates—nearly every method ranks first on at least one dataset. [Source](https://arxiv.org/html/2305.02997v4)

**Grinsztajn et al. (2022) — the definitive theoretical explanation.** Demonstrated three key inductive biases: (1) NNs are biased toward smooth solutions; (2) MLPs are not robust to uninformative features; (3) rotational invariance is harmful for tabular data. These biases are intrinsic to the architecture class, not artifacts of training or hyperparameters. [Source](https://ar5iv.labs.arxiv.org/html/2207.08815)

### 8. Feature Interactions: How Trees vs. NNs Handle Them

**Trees learn multiplicative feature interactions through hierarchical splits.** A path through a decision tree like "split on RSI > 70, then split on Volume > 1M" represents the interaction RSI × Volume. Deep trees can model high-order interactions through long paths. Gradient boosting chains hundreds of such trees, each potentially capturing different interaction patterns.

**Neural networks learn interactions through universal approximation.** In theory, even a single-hidden-layer MLP can approximate any continuous function (including all interactions). In practice, learning high-order interactions from limited data requires strong inductive biases. The spectral bias means NNs will learn main effects and low-order interactions before capturing higher-order structure—exactly the opposite of what financial data, with its threshold-based regime logic, demands.

**Financial relevance.** Financial alpha often lives in specific interaction regimes: "momentum works when volatility is low and rates are falling" is a 3-way interaction. Trees can discover this through 3 consecutive splits; an MLP must learn the equivalent AND-gate through its hidden layers, requiring substantially more data.

---

## Sources

### Kept

- **Grinsztajn et al. (2022)** — "Why do tree-based models still outperform deep learning on tabular data?" NeurIPS 2022. The foundational empirical+theoretical paper establishing three key inductive bias differences (smoothness bias, uninformative features, rotational invariance). 45 datasets, extensive hyperparameter search. [Link](https://ar5iv.labs.arxiv.org/html/2207.08815)
- **Forecast Collapse Paper (2025)** — "Forecast collapse of transformer-based models under squared loss in financial time series." Provides rigorous theoretical proof that under martingale assumptions, highly expressive models *necessarily* produce worse forecasts than simple linear models due to noise amplification through interpolation. Empirical validation on EUR/USD FX data. [Link](https://ar5iv.labs.arxiv.org/html/2604.00064)
- **McElfresh et al. (2023)** — "When Do Neural Nets Outperform Boosted Trees on Tabular Data?" NeurIPS 2023. Largest tabular benchmark to date (176 datasets, 19 algorithms, 538K models). Identifies dataset irregularity as key predictor of GBDT superiority. Released TabZilla benchmark. [Link](https://arxiv.org/html/2305.02997v4)
- **D&T Systems (2024)** — "XGBoost Beats LSTM and Transformers on Most Financial Time Series." Industry perspective with concrete data math (730 samples vs 33K parameters), practical workflow, and SHAP-based debugging. [Link](https://dtsystems.dev/blog/xgboost-vs-lstm-financial-time-series)
- **Shwartz-Ziv & Armon (2021)** — "Tabular Data: Deep Learning is Not All You Need." Information Fusion 2022. Showed deep tabular models fail to generalize to unseen datasets; XGBoost dominant as single model; ensemble of XGBoost + DL performs best. [Link](https://ar5iv.labs.arxiv.org/html/2106.03253)
- **Ke et al. (2025)** — "Curse of Attention: A Kernel-Based Perspective." PMLR 2025. First theoretical explanation of why Transformers fail on time series forecasting: asymmetric learning in attention when residual signs are inconsistent. [Link](https://proceedings.mlr.press/v280/ke25a.html)
- **Reyners (2019)** — "Gradient Boosting for Quantitative Finance." VU Amsterdam Winter School. Domain-specific evidence for GBDT superiority in financial applications, covering regularization mechanisms and practical considerations. [Link](https://staff.fnwi.uva.nl/a.khedher/winterschool/19ReynersPaper.pdf)

### Dropped

- **Various tabular-specific NN architecture papers** (TabNet, NODE, SAINT, FT-Transformer, ExcelFormer, etc.) — These claim to match or beat GBDTs but were evaluated in their own papers; independent benchmarks (Grinsztajn, Shwartz-Ziv, McElfresh) consistently show limited generalization.
- **APAR (2024)** — Proposes arithmetic-aware pre-training for tabular regression; too specialized and unvalidated independently.
- **TabPFN-related papers** — A meta-learned Bayesian inference network that performs remarkably well on small datasets (<3000 samples), but is fundamentally limited by quadratic scaling in training size and doesn't address the core theoretical question of *why* trees work better on typical financial data.

---

## Gaps

1. **Very large financial datasets (>1M samples).** The Grinsztajn and McElfresh benchmarks show the NN-GBDT gap narrowing at larger sample sizes (~50K). For intraday/tick data with millions of samples, deep learning may close or reverse the gap. No rigorous theoretical treatment exists for this threshold.

2. **Causal structure.** Trees' axis-aligned splits may implicitly encode causal relationships (splitting on causes before effects). No formal analysis exists of whether GBDTs' inductive bias provides causal advantages over NNs in financial prediction.

3. **Online/continual learning comparison.** While GBDTs' rapid retraining is a practical advantage, systematic theoretical comparisons of online learning regret bounds for trees vs. recurrent/attention architectures on non-stationary financial data are lacking.

4. **Probabilistic forecasting.** The forecast collapse result applies to point forecasts under squared loss. Whether distributional forecasting (predicting full conditional distributions) could rescue deep learning for finance remains an open theoretical question.

5. **Hybrid CNN-GBDT architectures.** Recent work on CNN embeddings fed into LightGBM shows promise but lacks theoretical analysis of *why* the hybrid outperforms either pure approach.

---

## Progress Update

