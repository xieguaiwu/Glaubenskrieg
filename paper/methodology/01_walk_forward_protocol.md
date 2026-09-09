# Purged Walk-Forward Protocol: Design and Validation

> **Source files**: `progress.md` (§2), `.omo/session_context.md`, `docs/architecture_assessment.md` (§2–3)
> **Code references**: `src/train/advanced_trainer.py`, `src/train/_walk_forward_utils.py`, `src/data/dataset.py`

---

## 1. Protocol Overview

```
Data: 200 stocks × ~1,110 days (HK) / ~2,500 days (US)
      OHLCV → 9 derived features → T=63 sequences
      
Walk-forward:
  [train | purge | val | step → train | purge | val | step → ...]
         ↓                            ↓
    Model W0                      Model W1
         ↓                            ↓
    val_IC_0                      val_IC_1
```

---

## 2. Walk-Forward Parameters (Final v5)

| Parameter | v3 (buggy) | v5 (fixed) | Reason |
|---|---|---|---|
| **train_period** | 504 days | 504–1,000 days | More data for DL |
| **purge_period** | 126 | 126 | Prevent leakage from overlapping sequences |
| **val_period** | 200 | 200 | Sufficient for Sharpe estimation (~0.07 SE) |
| **step_size** | 21 | 63 | 96% → 67% window overlap |
| **purge_months** | 6 | 6 | Standard (López de Prado) |
| **embargo_months** | 0 | 0 | Not implemented (T=63 sequences handle most leakage) |
| **N windows (HK)** | 28–37 | 3 (v5) / 10 (US) | Larger step drastically reduces N |

### Window Overlap Calculation

With T=63 sequences:
```
step_size = 21: val_overlap = (200 - 21) / 200 = 89.5%
step_size = 63: val_overlap = (200 - 63) / 200 = 67.0%
```

**Effective independent windows** = N_windows × (1 - overlap) ≈ 10 × 0.33 ≈ 3.3 for US.

---

## 3. Purge Mechanism

Source: `src/train/_walk_forward_utils.py` L75-84.

```
train range: [0, 1008)
purge range: [1008, 1134)
val range:   [1134, 1386)

train sequences ending in [1008, 1134) are purged
→ prevents val sequences from having training data overlap

Sequence t uses data [t, t+63) to predict t+63
Purge removes sequences where t+63 falls in or near val range
```

### Audit Result

| Check | Status | Evidence |
|---|---|---|
| Feature causality | ✅ | 9 features use only t and before (`features.py` L16-104) |
| Target alignment | ✅ | `forward_return[t] = close[t+1]/close[t]-1` |
| Train/val/test split | ✅ | Strict chronological 70/15/15 |
| Purge period | ✅ | 126 samples between train end and val start |
| Cross-window contamination | ✅ | Each window trains independent model |
| Test set isolation | ✅ | Never used for training or early stopping |

Source: `docs/architecture_assessment.md` §2.2.

---

## 4. Bug Fixes Impacting Protocol Integrity

| Bug | v3 State | v5 Fix | Effect on val IC |
|---|---|---|---|
| **test eval shuffle=True** | Used for final test | shuffle=False | Inflated test Sharpe (eliminated) |
| **test-prep warm-start** | Independent init | Walk-forward model weights | Disconnected val/test (fixed) |
| **Class weights from full data** | Included test period | Train-only | Minor leakage (fixed) |
| **Early stopping on Sharpe** | Noisy, gameable | Spearman IC | More stable selection |
| **LR warmup > n_epochs** | 200 > 100 (never decays) | 20 | Model never in decay phase (fixed) |

### v3 → v5 IC Evolution

```
v3: val_IC = 0.14, test_IC = unknown (buggy eval)
       ↓  Fix 5 bugs
v5: val_IC = 0.049, test_IC = 0.006
       ↓  Add held-out test
    True IC ≈ 0.00
```

---

## 5. US Full Baseline Walk-Forward Config

Source: `results/us_full_baseline.json`.

```
train_days:    1,000
purge_days:      126
val_days:        200
step_days:       126
seed:             42

Windows created: 10 (from 2,504 days)
Date range: 2015-11-20 to 2026-05-28
```

### Window Structure (US)

| Window | Train Range | Val Range | N_train | N_val |
|---|---|---|---|---|
| 0 | [0, 1000) | [1126, 1386) | 95400 | 95400 |
| 1 | [126, 1126) | [1252, 1512) | 95400 | 95400 |
| ... | ... | ... | ... | ... |
| 9 | [1134, 2134) | [2260, 2520) | 95400 | 95400 |

N_train = N_val = 95400 = 477 stocks × 200 days

---

## 6. Hold-Out Test Set Protocol

The final evaluation uses a **separate held-out test set** that is never used in any walk-forward window:

1. Walk-forward produces N window models
2. Each model is evaluated on its validation window → val_IC
3. The checkpoint with best val_IC is applied to the held-out test set
4. Reported test_IC and test_Sharpe are from this single evaluation

**Critical**: The v3 bug allowed the test model to be separately trained (not using walk-forward weights). v5 properly uses walk-forward model for test evaluation.

---

## 7. Comparison with Best Practices

### Oxford Benchmark Protocol (arxiv 2603.01820)

| Aspect | Oxford | Glaubenskrieg | Alignment |
|---|---|---|---|
| Retrain frequency | Every 5 years | Every 63 days | ✅ Both rolling |
| Transaction costs | Breakeven analysis | Not modeled | ❌ |
| Volatility target | 10% | Not applied | ❌ |
| Seeds | 50 (top 10) | 5 | ⚠️ Lower |
| HAC-adjusted t-stat | Yes | No | ❌ |
| Purged CV | Yes | Yes | ✅ |
| Time-series split | Yes | Yes | ✅ |

### Bai et al. Gold Standard (Symmetry 2026)

| Aspect | Bai et al. | Glaubenskrieg | Alignment |
|---|---|---|---|
| Per-window normalization | ✅ | ❌ (global) | ❌ |
| One-sided denoising | ✅ | ❌ (global) | ❌ |
| Training-only statistics | ✅ | ❌ | ❌ |
| DM-test significance | ✅ | ✅ (Phase 1) | ✅ |
| Walk-forward | ✅ | ✅ | ✅ |

---

## 8. Limitations

1. **Small N windows** — With step=63 and 2,500 days, only 10 windows for US, 3 for HK. This limits the statistical power of cross-window comparisons.
2. **No embargo period** — Purge is implemented but embargo (additional buffer after purge) is not. López de Prado recommends both.
3. **Global normalization** — Features are normalized on the full dataset, not per-window. This introduces minor future information leakage.
4. **Single seed for US baseline** — The 477-stock US run used only seed 42. Seed sensitivity not tested for full US.
5. **No alignment test** — We never tested whether prediction timing aligns with market microstructure (e.g., opening auction mechanics).
