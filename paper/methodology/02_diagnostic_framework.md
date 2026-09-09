# The IC Paradox: A 5-Step Diagnostic Framework

> **Source file**: `progress.md` (§4.1), `docs/architecture_assessment.md` (§3, Appendix D)
> **Script**: `scripts/diagnose_ic.py`

---

## 1. The Problem

**The IC Paradox**: Walk-forward validation suggests IC ≈ 0.04–0.14, but held-out test IC ≈ 0.00. How to distinguish genuine signal from overfitting?

### Glaubenskrieg's Experience

| Stage | val_IC | test_IC | Interpretation |
|---|---|---|---|
| v3 (buggy) | 0.14 | Unknown | Apparent signal |
| v5 (fixed) | 0.049 | 0.006 | Suspect overfit |
| IC Paradox Diagnosis | 0.049 | 0.006 | **Confirmed overfit** |

---

## 2. The 5-Step Diagnostic Framework

### Step 1: Checkpoint → Held-Out Test IC

**Method**: Take the checkpoint selected by walk-forward validation and compute IC directly on the never-before-seen test set.

| Model | val_IC | test_IC | Gap |
|---|---|---|---|
| CTM v5 | 0.049 | **0.006** | 0.043 |
| Linear Ridge | — | **0.006** | — |

**Interpretation**: If test_IC ≈ 0, walk-forward val_IC is overfit. Compare with simple linear baseline — if DL doesn't beat Ridge, complexity is not helping.

### Step 2: Train/Val Loss Gap Analysis

**Method**: Monitor train and val loss curves during training. In genuine signal, val loss tracks train loss. In overfitting, val loss diverges.

```
Healthy:    train_loss ↘  val_loss ↘  (gap constant or narrowing)
Overfitting: train_loss ↘  val_loss →  (gap widening)
```

**Glaubenskrieg finding**: Val loss stopped improving before train loss → classic overfitting signature.

### Step 3: Linear Ridge Baseline

**Method**: Train a simple L2-regularized linear model (451 params for 9 features × 50 stocks + bias) on the same data. If DL with 97K params doesn't significantly outperform Ridge, complexity adds no value.

| Model | Params | test_IC | test_Sharpe |
|---|---|---|---|
| CTM (Mamba) | 97,000 | ≈ 0.00 | -0.028 |
| Linear Ridge | 451 | 0.006 | +0.55 |
| GBDT | ~5,000 | 0.008 | -0.088 |

**Decision rule**: If Ridge IC ≥ DL IC, stop. No further DL investments justified.

### Step 4: Cross-Window IC Stability

**Method**: Compute IC separately for each walk-forward window. Genuine signal should be consistent across windows.

| Window | CTM val_IC |
|---|---|
| w0 | 0.019 |
| w1 | **0.074** (outlier) |
| w2 | 0.041 |
| **Mean** | **0.049** |

**Interpretation**: Window 1 drives the mean. Without it, μ = 0.030. With only 3 windows, the outlier is suspicious. A minimum of 10 windows is recommended for stability assessment.

### Step 5: Per-Stock IC Decomposition

**Method**: Compute IC separately for each stock. Genuine signal should be distributed across many stocks, not driven by a few.

| Metric | Value |
|---|---|
| Mean per-stock IC | 0.016 |
| Std per-stock IC | 0.069 |
| % positive | 60% |
| Statistical significance | **Not significant** (std > mean) |

**Interpretation**: 60% positive with σ=0.069 is exactly what noise looks like (expected 50% with high variance). If a few stocks drove the mean, that would suggest stock-specific overfitting rather than systematic signal.

---

## 3. Implementation: diagnose_ic.py

The diagnostic script (`scripts/diagnose_ic.py`, 503 lines) automates these steps:

```python
# Step 1: Load walk-forward checkpoints
model = MultiAssetCTM(config)
model.load_state_dict(torch.load(best_ckpt_path))

# Step 2: Evaluate on held-out test
test_ic = compute_ic(model, test_loader)
print(f"test_IC = {test_ic:.4f}")

# Step 3: Train Ridge baseline
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"ridge_IC = {compute_ic(ridge, X_test, y_test):.4f}")

# Step 4: Per-window IC
per_window_ic = [compute_ic_for_window(model, w) for w in windows]
print(f"Window ICs: {per_window_ic}")

# Step 5: Per-stock IC
per_stock_ic = [compute_ic_for_stock(model, s) for s in stocks]
print(f"Mean per-stock IC = {np.mean(per_stock_ic):.4f} ± {np.std(per_stock_ic):.4f}")
```

---

## 4. The Go/No-Go Decision Tree

```
Start: walk-forward val_IC > 0?
        ↓
    Step 1: Checkpoint → held-out test IC
        ↓
    test_IC > 0.02? ── Yes ──→ Step 3: Linear baseline
        ↓ No                        ↓
    ❌ STOP: No signal         Ridge IC < DL IC?
                                  ↓ Yes      ↓ No
                              Proceed → ❌ STOP
                                 ↓
                          Step 4: Window stability
                                  ↓
                          Consistent across windows?
                                  ↓ Yes    ↓ No
                              Proceed → ⚠️ Flag for further validation
                                 ↓
                          Step 5: Per-stock IC
                                  ↓
                          Broad distribution?
                                  ↓ Yes
                              ✅ Signal confirmed
```

---

## 5. Diagnostic Thresholds

| Metric | Healthy | Suspicious | Pathological |
|---|---|---|---|
| test_IC - val_IC gap | < 0.01 | 0.01–0.05 | > 0.05 |
| Ridge IC vs DL IC | DL > Ridge × 1.5 | DL ≈ Ridge | DL < Ridge |
| Window IC std/mean | < 1.0 | 1.0–2.0 | > 2.0 |
| Per-stock % positive | > 65% | 55–65% | < 55% |
| Per-stock t-stat (mean/SE) | > 2.0 | 1.0–2.0 | < 1.0 |

**Glaubenskrieg scores**: Gap = 0.043 (pathological), DL ≤ Ridge (pathological), Window std/mean = 0.025/0.044 ≈ 0.57 (healthy — but only 3 windows), Per-stock positive = 60% (suspicious).

**Composite**: Framework correctly diagnoses overfitted IC.

---

## 6. Generalization to Other Projects

The framework is designed for any walk-forward IC-based validation in finance:

1. **Prerequisite**: Held-out test set (never used in validation)
2. **Models**: At least one linear baseline (Ridge/Lasso)
3. **Windows**: Minimum 3 (recommended ≥ 10)
4. **Stocks**: Minimum 20 for per-stock decomposition
5. **Seeds**: Minimum 5 for seed sensitivity

---

## 7. Limitations

1. **Step 4 requires sufficient windows** — With only 3 windows, stability assessment is unreliable.
2. **Step 5 assumes cross-sectional independence** — Industry/sector correlations reduce effective N.
3. **The framework doesn't detect all forms of overfitting** — e.g., test set snooping, data-dependent hyperparameter tuning.
4. **Linear Ridge is itself fallible** — If true relationship is nonlinear, Ridge might miss signal that DL could find. But in practice, financial return signals are at most weakly nonlinear.
5. **No permutation test included** — A formal permutation test (shuffle targets, retrain, build H0 distribution) would strengthen the framework.
