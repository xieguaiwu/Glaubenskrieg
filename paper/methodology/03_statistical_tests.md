# Statistical Tests: DM Test, Permutation Test, Paired t-Tests

> **Source files**: `progress.md` (§7), `docs/architecture_assessment.md`, `research_overfitting_finance.md`

---

## 1. Diebold-Mariano Test for Predictive Accuracy

Source: Phase 1 FE-LightGBM experiment.

### Design

Compare LightGBM predictions against zero-prediction null:
```
H₀: E[L(e_model)] = E[L(e_null)]   (equal predictive accuracy)
H₁: E[L(e_model)] < E[L(e_null)]   (model is better)

L = squared error loss
Modified DM statistic with Newey-West HAC variance (truncated lag = 4)
```

### Results

| Config | DM p-value range | Windows significant at 5% |
|---|---|---|
| EMA=1 (raw features) | 0.014–0.124 | **4/5 (80%)** |
| EMA=10 (smooth) | 0.136–0.220 | **0/3 (0%)** |

### Interpretation

DM test serves as an **effective stopping criterion**:
- Raw features: 4/5 windows pass → "marginal" predictability (but IC still ≈ 0)
- After EMA smoothing: 0/3 windows pass → feature engineering destroyed what little structure existed
- **Without DM test, one might incorrectly conclude EMA smoothing is beneficial**

**Recommendation**: DM test should be the **primary gate** for any feature engineering step. If adding a feature reduces DM significance, remove it.

---

## 2. Paired t-Tests for Model Comparison

### 2.1 GARCH vs LightGBM (47 stocks)

Source: `results/garch_baseline.json`.

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| Paired t-test LGB vs GARCH QLIKE | t = -1.50 | 0.14 | No significant difference (LGB numerically worse) |
| DM LGB vs GARCH | DM = -6.73 | **1.6e-11** | GARCH significantly better (time-series aware) |

**Why DM and paired t disagree**: The paired t-test ignores time series structure. The DM test captures autocorrelation in loss differentials, giving higher power.

### 2.2 Sharpe-Tuned vs MSE Validation (8 windows)

Source: `results/portfolio_optimizer.json`.

| Comparison | Mean Δ Sharpe | p-value | Sig@5% |
|---|---|---|---|
| SharpeTuned vs MSE | **+0.79** | **0.011** | ✅ |
| LambdaRank vs MSE | -1.87 | — | ❌ (worse) |

**Interpretation**: The +0.79 increase in validation Sharpe is statistically significant (p=0.011). However, this comes from optimizing for Sharpe directly — the improvement is in the **validation metric**, not in test prediction accuracy.

---

## 3. Permutation Test (Recommended but Not Implemented)

**Design**:
1. Shuffle time-series labels (preserving autocorrelation structure? — difficult)
2. Retrain model on shuffled data
3. Compute test IC
4. Repeat 1,000× to build null distribution
5. Compare actual test IC against null

**Expected null distribution**: IC ≈ 0 with std ≈ 0.01–0.03 (depends on N_stocks and T)

**Glaubenskrieg prediction**: Our test_IC = 0.006 would fall well within the null distribution (p > 0.5), confirming no signal.

**Why not implemented**: Computing 1,000 retrains × 10 windows × 5 seeds = 50,000 training runs — computationally prohibitive for the V100 setup. However, for a published paper, this is essential.

---

## 4. HAC-Adjusted t-Statistics

Source: Oxford benchmark (arxiv 2603.01820) recommendation.

### Standard t-statistic problem

```
t = mean_return / (std_return / √N)

But: returns are autocorrelated and heteroskedastic
→ Standard t-statistic overstates significance
```

**Fix**: Newey-West HAC standard errors:
```
Var(mean) = γ₀ + 2Σ_{j=1}^{L} (1-j/(L+1))γⱼ
where γⱼ = autocovariance at lag j
L = floor(N^{1/4}) truncation lag
```

**Implementation** in `scipy.stats` or `statsmodels`:
```python
from statsmodels.stats.sandwich_covariance import cov_hac
```

### When to use
- Any strategy return Sharpe ratio
- DM test (already uses HAC)  
- Cross-window IC comparison
- Per-stock IC significance

---

## 5. Deflated Sharpe Ratio

Source: López de Prado.

### Problem
With multiple trials (N_windows × N_seeds × N_models = 150+), the maximum Sharpe is upward biased.

### DSR Adjustment
```
DSR = SR_observed / sqrt(1 + (N_trials - 1) × Var(SR_null))
```

### Glaubenskrieg Context
- 3 models × 5 seeds × 10 windows = 150 SR observations
- Maximum observed Sharpe = 2.92 (SharpeTuned, window 1)
- DSR after adjusting for 150 trials: likely not significant
- Minimum SR required for significance at 150 trials: DSR threshold ≈ 2.0–2.5

---

## 6. Seed Sensitivity Analysis

Source: Oxford benchmark methodology.

### Design
- Run each configuration with 5+ random seeds
- Aggregate top-N by validation performance (not mean)
- Test stability: do rankings change if we use 3 seeds vs 5 seeds vs 10 seeds?

### Glaubenskrieg Results

| Model | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 1024 | Mean ± Std |
|---|---|---|---|---|---|---|
| CTM val_IC | 0.044 | 0.049 | 0.050 | 0.034 | 0.065 | 0.048 ± 0.011 |
| CTM test_Sharpe | -0.092 | -0.016 | -0.268 | -0.130 | +0.041 | -0.093 ± 0.114 |
| GBDT val_IC | 0.005 | 0.009 | 0.004 | 0.008 | 0.015 | 0.008 ± 0.004 |
| GBDT test_Sharpe | -0.531 | -0.108 | +0.209 | -0.444 | +0.432 | -0.088 ± 0.409 |

**Observation**: CTM val_IC is relatively stable (CV = 23%), but test_Sharpe varies enormously (CV > 100%). This is expected: when IC ≈ 0, Sharpe is dominated by noise.

---

## 7. Cross-Validation for Hyperparameter Selection

### Walk-Forward CV (used)
- Each hyperparameter configuration evaluated on N validation windows
- Mean validation performance used for selection
- Final evaluation on held-out test set

### Purged K-Fold CV (recommended but not used)
- K = 5 folds with purging
- Each fold: train on K-4 blocks, validate on 1 block (with purge buffer)
- More computationally expensive but more robust

---

## 8. Summary: Statistical Testing Checklist

| Test | Used? | Purpose |
|---|---|---|
| Diebold-Mariano | ✅ | Feature engineering gates |
| Paired t-test | ✅ | Model comparison (GARCH vs LGB) |
| Sharpe t-test (uncorrected) | ✅ | Strategy evaluation |
| Per-window IC stability | ✅ | Signal consistency |
| Per-stock IC decomposition | ✅ | Signal source analysis |
| HAC-adjusted errors | ❌ | Autocorrelation correction |
| Permutation test | ❌ | Formal H₀ significance |
| Deflated Sharpe Ratio | ❌ | Multiple testing correction |
| Seed sensitivity | ✅ | Training stability |
| DM test for model comparison | ✅ | GARCH vs LGB |
