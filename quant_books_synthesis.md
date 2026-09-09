# Quant Finance Books Synthesis

> **Sources synthesized:**
> 1. *Advances in Financial Machine Learning* — Marcos López de Prado (2018)
> 2. *Deep Learning for Finance* — Sofien Kaabar (2024)
> 3. *Greedy Function Approximation: A Gradient Boosting Machine* — Jerome Friedman (2001)
> 4. *Stochastic Gradient Boosting* — Jerome Friedman (2002)

---

## 1. The Standard Quant Workflow (López de Prado)

López de Prado defines a **five-station production chain** that distinguishes professional quant finance from amateurish "plug-and-play" ML:

### The Five Stations

| Station | Role | Key Chapters |
|---------|------|--------------|
| **Data Curator** | Collect, clean, index, adjust, deliver data | Ch. 2 |
| **Feature Analyst** | Transform raw data → informative signals; discover features | Ch. 2–9, 17–19 |
| **Strategist** | Formulate theory explaining features; design strategy as experiment | Ch. 10, 16 |
| **Backtester** | Assess profitability under multiple scenarios; estimate PBO | Ch. 11–16 |
| **Deployment** | Integrate code into production; optimize latency | Ch. 20–22 |

**Critical principle:** "Backtesting is not a research tool. Feature importance is." — Marcos' First Law of Backtesting. Adjusting your model based on backtest results is dangerous. Invest effort in data structuring, labeling, weighting, ensembles, cross-validation, feature importance — *then* backtest once.

### The Complete Workflow (in order)

1. **Sample bars optimally** — Use dollar bars or information-driven bars (tick imbalance, volume imbalance), not time bars. Markets do not process information at constant time intervals.

2. **Label with triple-barrier method** — Not fixed-time horizon. Three barriers: profit-taking (upper horizontal), stop-loss (lower horizontal), expiration (vertical). Labels are path-dependent.

3. **Apply meta-labeling** — Separate the side prediction (primary model) from the size decision (secondary ML model). The secondary model learns {0,1} — take or pass.

4. **Assign sample weights** — Correct for overlapping outcomes using average uniqueness. Apply time decay (older observations less relevant).

5. **Fractionally differentiate features** — Find minimum *d* needed for stationarity while preserving maximum memory. Standard integer differentiation (*d*=1) over-differentiates. For E-mini S&P 500, stationarity is achieved at *d*≈0.35 with correlation 0.995 to original series.

6. **Ensemble with bagging (not boosting)** — Bagging reduces variance and addresses overfitting (the bigger concern in finance). Boosting reduces bias but risks overfitting the low signal-to-noise ratio.

7. **Purged k-fold cross-validation** — Purge training observations whose labels overlap with testing set. Apply embargo (≈1% of T) after each test set.

8. **Feature importance (MDI, MDA, SFI)** — Always before backtesting. Validate with PCA orthogonalization: if PCA's unsupervised ranking matches feature importance ranking, it's confirmatory evidence.

9. **Hyper-parameter tuning** — Use PurgedKFold with `neg_log_loss` scoring (not accuracy). Log loss accounts for prediction confidence, which matters for PnL.

10. **Bet sizing from probabilities** — Translate predicted probabilities → bet sizes via sigmoid or power functions. Average active bets, discretize to prevent overtrading.

11. **Backtest with PBO estimation** — Use Combinatorially-Symmetric Cross-Validation (CSCV) to estimate Probability of Backtest Overfitting.

---

## 2. The Industry-Standard Approach to Stock Prediction

### Preprocessing Steps (Essential)

| Step | Method | Why |
|------|--------|-----|
| **Bar sampling** | Dollar bars or tick imbalance bars | IID-Normal returns; homoscedasticity |
| **Stationarity** | Fractional differentiation (FFD) | Preserves memory while achieving stationarity; integer d=1 is over-differentiation |
| **Feature engineering** | Technical indicators, microstructural features, PCA orthogonalization | Reduce substitution effects; validate with unsupervised PCA |
| **Labeling** | Triple-barrier method with dynamic volatility thresholds | Path-dependent; accounts for stop-losses and profit-taking |
| **Weighting** | Sample weights (avg uniqueness × absolute return attribution) + time decay | Corrects for non-IID overlapping outcomes |
| **Cross-validation** | Purged k-fold with embargo | Prevents leakage from testing → training |

### Labeling: Fixed vs. Triple-Barrier vs. Meta-Labeling

**Fixed-time horizon** (what most papers use — López de Prado calls this a mistake):
- Labels based on return over h bars: y ∈ {−1, 0, 1}
- Same threshold τ regardless of volatility → most labels are 0
- Ignores path; no stop-loss limits → unrealistic

**Triple-barrier method** (López de Prado's innovation):
- Three barriers: upper (profit-taking), lower (stop-loss), vertical (expiration)
- Label = sign of return at first barrier touch
- Barriers are dynamic functions of estimated volatility
- Path-dependent (must track entire path [t₀, t₀+h])
- Configurations: [pt, sl, t1] — each 0/1
  - [1,1,1] = standard (profit, stop-loss, expiration)
  - [0,0,1] = fixed-time horizon (but applicable to dollar bars)

**Meta-labeling** (López de Prado's key innovation):
- Primary model decides **side** (long/short)
- Secondary ML model decides **size** ({0,1} — take or pass)
- Labels are {0, 1} not {−1, 0, 1}
- Benefits:
  1. ML layer on top of white-box fundamental model ("quantamental")
  2. Limits overfitting (ML doesn't decide side, only size)
  3. Decouples side-prediction features from size-prediction features
  4. Corrects low-precision primary models by filtering false positives
  5. Increases F1-score

---

## 3. Meta-Labeling in Detail

### How It Changes the Prediction Problem

**Standard labeling:**
- ML learns {−1, 0, 1} — side and size together
- Must get both right to profit
- Confusion matrix: FP (false long when should be short) and FN (missed opportunity)

**Meta-labeling:**
- Primary model (any source: ML, econometric, fundamental, even human intuition) suggests side
- Secondary ML model learns {0, 1} — should we act on this signal?
- "It is not its purpose to come up with a betting opportunity. Its purpose is to determine whether we should act or pass on the opportunity that has been presented."
- Horizontal barriers can be asymmetric (known side → differentiate PT from SL)

**Practical implementation (from Snippets 3.6–3.7):**
1. Get side from primary model
2. Pass `side` argument to `getEvents()`
3. Horizontal barriers: `ptSl = [pt, sl]` (asymmetric)
4. Labels from `getBins()` are {0, 1}
5. Fit secondary classifier (RF, SVC) on meta-labels
6. Use predicted probability → bet size via sigmoid function

**For quantamental firms:** "Meta-labeling will help us figure out when we should pursue or dismiss a discretionary PM's call."

---

## 4. Fractional Differentiation (FFD)

### The Stationarity vs. Memory Dilemma

- **Returns** (d=1): stationary but memory-less
- **Prices** (d=0): have memory but non-stationary
- **FFD(d)**: find minimum d such that ADF p-value < 5%, preserving maximum memory

### Key Results

| Series | ADF Statistic | Correlation to Original |
|--------|---------------|------------------------|
| Original prices (d=0) | −0.34 | 1.000 |
| FFD(d=0.35) | −2.86 (stationary at 95%) | 0.995 |
| Returns (d=1) | −46.91 | 0.030 |

**Finding:** Most studies over-differentiate. Standard d=1 removes much more memory than necessary.

### Implementation
- Fixed-width window FFD: drop weights after |ωₖ| falls below threshold τ (e.g., τ=1e-5)
- Iterative weight generation: ωₖ = ωₖ₋₁ × (k−1−d)/k
- Expanding window causes negative drift; FFD is driftless

---

## 5. Backtesting Protocols That Prevent Overfitting

### The Problem

"The maddening thing about backtesting is that, the better you become at it, the more likely false discoveries will pop up."

A flawless backtest (no survivorship bias, no look-ahead, realistic costs) is **still probably wrong** — because the expert has run tens of thousands of backtests, and selection bias guarantees false positives.

### Key Protocols

**1. Purged k-fold CV (Ch. 7)**
- Purge training observations whose labels overlap with testing set
- Apply embargo (≈1% of T) after each test set to prevent serial correlation leakage
- Never shuffle before partition (shuffling leaks information in financial time series)

**2. Combinatorially-Symmetric Cross-Validation (CSCV) for PBO (Ch. 11)**
- Form matrix M (T × N) of PnL series from N trials
- Partition into S subsets, form all combinations of S/2
- For each combination: train on subset, test on complement
- Compute logit λ = rank of OOS performance of IS-optimal strategy
- PBO = ∫_{-∞}^{0} f(λ)dλ (probability IS-optimal strategy underperforms OOS)

**3. Deflated Sharpe Ratio (Ch. 14)**
- Corrects Sharpe ratio for: non-normality, serial correlation, and **number of trials**
- A Sharpe of 2.0 based on 200 trials is not impressive

**4. Never backtest until model is fully specified**
- "Backtesting while researching is like drinking and driving" (Marcos' Second Law)
- Backtest is to **discard** bad models, not to improve them

**5. Simulate scenarios, not history**
- History is one random path. Test under thousands of "what if" scenarios.
- Harder to overfit scenarios than a single historical path.

**6. Feature importance before backtesting**
- MDI (in-sample), MDA (permutation-based), SFI (single feature)
- Validate with PCA: if PCA ranking matches feature importance ranking → confirmatory evidence

**7. Bagging for variance reduction**
- Set `max_samples = avg uniqueness` to account for redundant observations
- Sequential bootstrap to reduce correlation between estimators

### Common Backtest Sins (Luo et al., 2014)
1. Survivorship bias
2. Look-ahead bias
3. Storytelling (ex-post justification)
4. Data mining / data snooping
5. Ignoring transaction costs
6. Outlier-driven strategies
7. Ignoring shorting constraints

---

## 6. Deep Learning for Retail Traders (Kaabar)

### What Actually Works?

**Kaabar's book is introductory/practical, not state-of-the-art research.** Key takeaways:

**Models tested (on lagged returns):**
| Model | Training Accuracy | Test Accuracy | OOS Correlation |
|-------|-----------------|---------------|-----------------|
| Dummy Regressor | 49.3% | 49.3% | — |
| Linear Regression | 58.5% | 49.5% | 0.014 |
| SVR (RBF kernel) | 57.9% | 50.1% | 0.024 |
| SGD Regression | Similar | ~50% | ~0.02 |
| MLP (2 hidden layers) | 92.4% | 54.9% | 0.044 |

**Critical observation:** All models show **massive in-sample / out-of-sample gap**. The MLP achieves 92.4% training accuracy but only 54.9% test accuracy — clear overfitting. The gap between training and test correlation (0.989 vs 0.044) is even more stark.

### Practical Architectures for Retail

**What Kaabar recommends for retail traders:**

1. **Feature engineering matters more than model choice**
   - Use technical indicators (RSI, moving averages, Bollinger Bands) as features, not just lagged returns
   - Combine multiple indicators → better than raw prices

2. **Simple ML first**
   - Linear regression, SVR, SGD as baselines
   - MLP (2-3 hidden layers, ReLU activation, Adam optimizer)
   - LSTM/RNN for sequential patterns (but prone to overfitting on small data)

3. **Regularization is essential**
   - Dropout (randomly omit neurons during training)
   - Early stopping (stop when validation loss plateaus)
   - Batch normalization

4. **Model evaluation metrics (in order of importance)**
   - Profit factor = gross profits / gross losses (>1.0 is profitable)
   - Sharpe ratio = (μ − r_f) / σ (>1.0 desirable)
   - Maximum drawdown (avoid >20-30%)
   - Accuracy / hit ratio (less important than risk-adjusted returns)

5. **Data preprocessing for retail**
   - Always difference price data to achieve stationarity (Kaabar uses integer d=1)
   - Split 80/20 or 70/30 train/test
   - Scale features (StandardScaler for SVR, SGD)

**Kaabar's limitations:** Does not cover fractional differentiation, meta-labeling, triple-barrier method, purged CV, or PBO estimation. The book is at an intermediate level — useful for getting started but insufficient for professional-grade quant work.

---

## 7. Theoretical Foundation of GBDT (Friedman)

### Greedy Function Approximation (2001)

**Core insight:** Function estimation is reframed as **numerical optimization in function space**, not parameter space.

**Mathematical framework:**
- Goal: Find F*(x) = argmin_F E_{y,x}[L(y, F(x))]
- Represent F(x) as additive expansion: F(x) = Σ β_m h(x; a_m)
- Each h(x; a_m) is a simple parameterized function (e.g., regression tree)

**Steepest-descent in function space:**
- At each iteration m, compute "pseudo-residuals":
  - ỹ_im = −[∂L(y_i, F(x_i)) / ∂F(x_i)] at F = F_{m−1}
- Fit base learner h(x; a_m) to pseudo-residuals by least squares
- Line search: β_m = argmin_β Σ L(y_i, F_{m−1}(x_i) + β h(x_i; a_m))

**For regression trees (TreeBoost):**
- Tree partitions x-space into L disjoint regions {R_lm}
- Each region predicts constant: ȳ_lm = mean of pseudo-residuals in R_lm
- Line search reduces to per-region location estimate
- Key insight: the same least-squares fitting criterion works for **any differentiable loss function** by transforming the problem through pseudo-residuals

**Loss functions supported:**
- Least squares: (y − F)²
- Least absolute deviation: |y − F|
- Huber-M (robust regression)
- Multiclass logistic likelihood (classification: log(1 + e^{−2yF}))

### Stochastic Gradient Boosting (2002)

**Key innovation:** Add **randomization** by subsampling training data at each iteration.

- At each iteration m, draw a random subsample (without replacement) from full training data
- Fit base learner and compute model update on this subsample only
- Benefits:
  1. **Substantially improves approximation accuracy** (empirically shown)
  2. **Increases execution speed** (smaller N per iteration)
  3. **Increases robustness** against overcapacity of base learner

**Gradient TreeBoost Algorithm (simplified):**

```
1. F₀(x) = argmin_γ Σ L(y_i, γ)
2. For m = 1 to M:
   a. ỹ_im = −[∂L(y_i, F(x_i))/∂F(x_i)] at F = F_{m−1}
   b. Fit L-terminal regression tree to {ỹ_im, x_i}
   c. γ_lm = argmin_γ Σ_{x_i ∈ R_lm} L(y_i, F_{m−1}(x_i) + γ)
   d. F_m(x) = F_{m−1}(x) + ν · γ_lm · 1(x ∈ R_lm)
```

**Shrinkage parameter ν ∈ (0, 1]**: Controls learning rate. Small values (ν ≤ 0.1) lead to much better generalization error.

### Why GBDT Matters for Quant Finance

- GBDT (as XGBoost, LightGBM, CatBoost) is arguably the **most used algorithm** in quantitative finance competitions and industry
- Handles mixed data types, missing values, non-linear relationships
- Feature importance (MDI) is built-in
- More robust to outliers than neural networks
- Bagging variant (Random Forest) is preferred over boosting for finance because:
  - **Boosting** reduces bias → risks overfitting the low SNR
  - **Bagging** reduces variance → addresses the bigger concern

---

## 8. Synthesis: What Does This Mean for Stock Prediction?

### The Industry-Standard Pipeline

```
Raw tick data
  ↓ Dollar bars / tick imbalance bars
Stationary price series
  ↓ Fractional differentiation (FFD, find min d)
Predictive features
  ↓ (Technical indicators, microstructural features, PCA)
Feature matrix X
  ↓ Triple-barrier labeling (dynamic volatility thresholds)
Labels y with t1 (first barrier touch time)
  ↓ Meta-labeling (if primary model exists for side)
Meta-labels {0, 1} + sample weights (avg uniqueness × return attribution)
  ↓ Purged k-fold CV with embargo
Train ML model (RF bagging with max_samples = avg uniqueness)
  ↓ Feature importance (MDI + MDA + SFI + PCA validation)
Select features
  ↓ Hyper-parameter tuning (PurifiedGridSearchCV, neg_log_loss scoring)
Final model
  ↓ Bet sizing from sigmoid(predicted_probability)
  ↓ Average active bets + discretization
Trading signals
  ↓ Backtest (one shot, no iteration)
  ↓ PBO estimation (CSCV), Deflated Sharpe Ratio
Go/No-Go decision
```

### Key Takeaways

1. **Most financial ML papers are wrong** — they use time bars, fixed-time labeling, standard k-fold CV, and integer differentiation. All of these introduce methodological errors.

2. **Meta-labeling is the single most practical innovation** — it allows quantamental integration, limits overfitting, and separates the side and size problems.

3. **Fractional differentiation preserves predictive signal** — standard returns (d=1) throw away ~97% of the memory in prices. FFD(d≈0.35) keeps 99.5% while achieving stationarity.

4. **Backtest overfitting is the fundamental problem** — PBO estimation and Deflated Sharpe Ratio are essential. No backtest should be trusted without knowing the number of trials.

5. **For retail/small-scale traders (Kaabar):** Start with feature engineering, use simple ML (RF, SVR, shallow MLP), regularize aggressively, and evaluate on Sharpe ratio and drawdown — not just accuracy.

6. **GBDT is the theoretical workhorse** — Friedman's insight (optimization in function space via pseudo-residuals) made gradient boosting practical. Stochastic subsampling (2002) added robustness. The shrinkage parameter (ν ≤ 0.1) is critical for generalization.

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Kaabar, S. (2024). *Deep Learning for Finance*. O'Reilly.
- Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189–1232.
- Friedman, J.H. (2002). "Stochastic Gradient Boosting." *Computational Statistics & Data Analysis*, 38(4), 367–378.
- Bailey, D., Borwein, J., López de Prado, M., & Zhu, J. (2017). "The Probability of Backtest Overfitting." *Journal of Computational Finance*, 20(4), 39–70.
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 40(5), 94–107.
- Luo, Y., et al. (2014). "Seven Sins of Quantitative Investing." Deutsche Bank Markets Research.
