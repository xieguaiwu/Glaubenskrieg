# Why GBDT Beats Deep Learning on Tabular Financial Data

> **Source files**: `research_gbdt_vs_dl.md`, `research_theoretical_why.md`, `research_finance_arch.md`

---

## 1. The Tabular Data Gap: Three Structural Reasons

Source: Grinsztajn, Oyallon & Varoquaux (2022) — 45 datasets, 20,000 compute-hours, NeurIPS 2022.

### 1.1 Smoothness Bias (Spectral Bias)

Neural networks are biased toward learning low-frequency (smooth) functions first (Rahaman et al., 2019). Financial target functions are **inherently non-smooth** — they contain:
- Sharp regime boundaries (trending vs mean-reverting)
- Discontinuous threshold effects (RSI > 70 → overbought)
- Irregular, heavy-tailed patterns

**Empirical proof**: Gaussian-smoothing training targets barely affects NN performance but **markedly degrades GBDT accuracy**. Trees capture irregular patterns natively; NNs cannot.

### 1.2 Rotational Invariance Is Harmful (Ng 2004)

MLPs are rotationally invariant — they treat all linear combinations of features equally. This is **harmful** for tabular data where individual features have semantic meaning (e.g., "volume," "RSI").

**Empirical proof**: Randomly rotating features **reverses the performance order** (NNs > trees). Rotation destroys the natural feature basis that axis-aligned tree splits rely on.

### 1.3 Robustness to Uninformative Features

Tabular datasets contain many uninformative features. MLPs allocate modeling capacity to irrelevant inputs because dense connections cannot easily "ignore" them. Trees perform implicit feature selection at every split.

**Empirical proof**: Adding random Gaussian noise features widens the NN-GBDT gap. Removing irrelevant features narrows it.

---

## 2. Financial Data Properties Hostile to Deep Learning

### 2.1 Low Signal-to-Noise Ratio

**Daily stock returns are ~80% noise.** The predictable component is small and unstable. Deep models with millions of parameters are extremely effective at memorizing noise — the opposite of what is needed for OOS generalization.

### 2.2 Non-Stationarity

Market regimes shift constantly. Trees handle regime changes through feature-threshold splits that are relatively stable. NNs learn distributed representations that are more brittle to distributional changes.

### 2.3 Data Scarcity

3 years of daily bars = ~750 samples. A conservative LSTM (64 hidden units, 20 features) has ~33,000 parameters — yielding **0.02 samples per parameter**. XGBoost with max_depth=6 has ~500 effective degrees of freedom — **1.5 samples per parameter** (75× better sample efficiency).

---

## 3. The 730 vs 33,000 Math

Source: D&T Systems (2024).

| Model | Effective DOF | Samples | Samples/Param |
|---|---|---|---|
| Linear Ridge | 451 | 750 | 1.66 |
| XGBoost (depth=6, 100 trees) | ~500 | 750 | 1.50 |
| LSTM (1 layer, 64 hidden) | ~33,000 | 750 | **0.02** |
| CTM (Mamba, v5) | 97,000 | 750 | **0.008** |

The Delta: **GBDT has 75–1,875× better sample efficiency** than deep learning for typical financial datasets.

---

## 4. Why Attention Fails on Financial Data

### 4.1 Asymmetric Learning (Ke et al., 2025, PMLR)

When the sign of the previous residual is inconsistent with the current step's sign (the norm in financial returns), attention networks fail to learn residual features. The attention kernel induces asymmetric learning — it can model one direction of dependence but fails when sign patterns flip unpredictably.

### 4.2 Forecast Collapse under Squared Loss (2025)

Source: `research_theoretical_why.md` §5.

For financial price processes that are approximately martingales (E[X_{t+1}|F_t] = X_t), the Bayes-optimal forecast collapses to a constant. When a highly expressive model (Transformer) is trained via ERM, it does not uncover hidden structure — it **amplifies noise through interpolation**.

**Theoretical proof**: The interpolating predictor's MSE ≥ 2Hσ² (double irreducible error), while simple linear predictor achieves Hσ² + O(1/n).

**Empirical confirmation**: Testing PatchTST on EUR/USD 30-second data — Transformer produced larger trajectory errors than linear model on ~92% of forecasting windows, with average error ratio of 1.71×.

---

## 5. McElfresh et al. (2023) — Largest Tabular Benchmark

Source: NeurIPS 2023, 176 datasets, 19 algorithms, 538,650 trained models.

### Key Conclusions

1. **No single algorithm dominates** — nearly every method ranks first on at least one dataset
2. **GBDTs outperform NNs on**: large datasets, irregular (skewed/heavy-tailed) distributions, high feature-to-sample ratios
3. **CatBoost is the single best algorithm overall**
4. For ~1/3 of datasets, **light tuning on CatBoost gives more improvement than switching between NNs and GBDTs**
5. Datasets where NNs win: very small (TabPFN), or very large with smooth target functions

**Financial data archetype**: large N_samples (relative to N_features), heavy-tailed, skewed, irregular → **GBDT favored**.

---

## 6. CNN-LightGBM Hybrid: The Only Architecture That Beats Pure GBDT

Source: Bai et al. (Symmetry 2026), NIFTY50 daily log-returns.

### Ablation Results

| Configuration | OOS-R² | Drop from Best |
|---|---|---|
| Full CNN-LightGBM | 0.0285 | — |
| No Denoising | 0.0021 | **-93%** |
| No Embedding Standardization | 0.0210 | -26% |
| No Recency Weighting | 0.0237 | -17% |

**The hybrid's gain is from denoising, not architecture**. CNN encoder alone contributes only 7% of total gain.

### Economic Evaluation

- Sharpe 1.69 at 5bp costs
- Falls to 0.39 at 30bp costs
- Edge is extremely thin

---

## 7. The Recommended Workflow

Source: D&T Systems (2024), validated by Glaubenskrieg experience.

1. **Start with linear models** (Ridge/Lasso). If Ridge cannot capture signal, no DL will help.
2. **Use XGBoost/LightGBM with walk-forward CV** (5+ folds, ≥20% OOS per fold).
3. **Add deep learning only when**: >50K samples, raw features with genuine sequential structure, and GBDT has plateaued.
4. **Consider hybrid architectures** (encoder → GBDT) when representation learning can complement tree-based prediction.
5. **Never skip walk-forward + held-out test** on financial data.

---

## 8. Glaubenskrieg-Specific Validation

| Literature Claim | Glaubenskrieg Evidence |
|---|---|
| GBDT > DL on tabular finance | ✅ CTM (97K params) IC ≈ 0.006; Ridge (451 params) IC = 0.006; GBDT IC = 0.008 |
| Linear model = best baseline | ✅ Ridge and LightGBM produce identical IC. Ridge has higher Sharpe |
| Denoising is critical | ❌ EMA smoothing reduced IC (contradicts Bai et al. — likely data/feature specific) |
| Signal-to-noise ceiling | ✅ IC ≈ 0 confirms OHLCV-based daily return SNR is below detectability threshold |
| Walk-forward validation essential | ✅ v3 → v5 IC drop from 0.14 → 0.00 after bug fixes confirms necessity |
