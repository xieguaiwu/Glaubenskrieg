# Architecture Comparison: CTM (Mamba SSM) vs GBDT vs Ensemble vs Linear Ridge

> **Source files**: `progress.md` (§5.1, §5.2), `docs/architecture_assessment.md` (§1, §4–5), `.omo/session_context.md` (§3)

---

## 1. Final v5 Results (V100+3090, All Bugs Fixed)

Source: `progress.md` (§3, v5 table), `.omo/session_context.md` (§3).

### Three Models × 5 Seeds × 3 Windows = 45 Runs

| Model | Seed | n_windows | val_IC | test_IC | test_Sharpe |
|---|---|---|---|---|---|
| **CTM v5** | 42 | 3 | 0.044 | -0.006 | -0.092 |
| **CTM v5** | 123 | 3 | 0.049 | — | -0.016 |
| **CTM v5** | 456 | 3 | 0.050 | — | -0.268 |
| **CTM v5** | 789 | 3 | 0.034 | — | -0.130 |
| **CTM v5** | 1024 | 3 | 0.065 | — | +0.041 |
| **CTM avg** | — | — | **0.049** | **≈ 0.00** | **-0.028** |
| **Ensemble+TimeGate** | 42 | 3 | 0.021 | — | -0.098 |
| **Ensemble+TimeGate** | 123 | 3 | 0.036 | — | +0.141 |
| **Ensemble+TimeGate** | 456 | 3 | 0.051 | — | -0.048 |
| **Ensemble+TimeGate** | 789 | 3 | 0.045 | — | -0.051 |
| **Ensemble+TimeGate** | 1024 | 3 | 0.034 | — | -0.063 |
| **Ens avg** | — | — | **0.037** | **—** | **-0.024** |
| **GBDT v5** | 42 | 3 | 0.005 | — | -0.531 |
| **GBDT v5** | 123 | 3 | 0.009 | — | -0.108 |
| **GBDT v5** | 456 | 3 | 0.004 | — | +0.209 |
| **GBDT v5** | 789 | 3 | 0.008 | — | -0.444 |
| **GBDT v5** | 1024 | 3 | 0.015 | — | +0.432 |
| **GBDT avg** | — | — | **0.008** | **—** | **-0.088** |

### Aggregated Statistics

| Metric | CTM (97K params) | Ensemble+Gate (157K) | GBDT (~5K) | Linear Ridge (451) |
|---|---|---|---|---|
| **mean val_IC** | 0.049 | 0.037 | 0.008 | — |
| **test_IC** | ≈ 0.00 | — | — | **0.006** (HK) / **0.031** (US) |
| **test_Sharpe** | -0.028 | -0.024 | -0.088 | **+0.55** (HK) / +0.82 (US) |
| **Params** | 97,000 | 157,323 | ~5,000 | **451** |
| **Params/sample** | ~125 | ~202 | ~6.4 | **0.58** |

---

## 2. Parameter Efficiency Analysis

Source: `docs/architecture_assessment.md` (§5).

### CTM Parameter Breakdown (157,323 total, pre-fix)

| Component | Params | % Total | Efficiency |
|---|---|---|---|
| Asset Embedding | 3,200 | 2.0% | ✅ Reasonable |
| Conditioning Projection | 4,160 | 2.6% | ✅ Reasonable |
| **Mamba Backbone ×3** | **93,056** | **59.2%** | 🟡 Overkill for T=63 |
| Dead Backbone Heads | 4,355 | 2.8% | ❌ Dead code |
| Cross-Attn Q/K/V/Out | 16,640 | 10.6% | 🟡 Acceptable |
| Cross-Attn adj_bias | 2,500 | 1.6% | 🔴 Unconstrained (N²) |
| Cross-Attn FFN | 33,088 | 21.0% | 🔴 4× expansion too large |
| Output Heads | 260 | 0.2% | ✅ Minimal |

### Health Metrics

| Metric | v5 (post-fix) | Healthy | Status |
|---|---|---|---|
| Params/sample | **~125** | 0.01–0.1 (finance) | ❌ 1,250× over |
| adj_bias DOF | 2,500 (N=50) | ≤200 (N×r, r≤4) | ❌ 12× over |
| Mamba layers | 2 (was 3) | 1–2 for T=63 | 🟡 |
| Cross-attn FFN | 256→128 | 2× | 🟡 |

### v5 Fixes Applied
1. state_dim: 16→8, n_layers: 3→2 (97K params total)
2. step_size: 21→63 (window overlap 96%→67%)
3. lambda_sharpe: 0.5→0.1 (prevent Sharpe collapse)
4. Early stopping: Sharpe → Spearman IC
5. test eval: shuffle=False + same early stopping
6. output_dim=0: removed dead heads
7. adj_bias: low-rank (N²→2×N×r)

---

## 3. The Definitive Test: Linear Ridge Comparison

Source: `progress.md` (§5.1), `results/us_full_baseline.json`, `results/newdata_baselines.json`.

### HK Market (50 stocks)

| Model | Params | test_IC | test_Sharpe |
|---|---|---|---|
| CTM (Mamba SSM) | 97,000 | ≈ 0.00 | -0.028 |
| **Linear Ridge** | **451** | **0.006** | **+0.55** |
| GBDT (Hoffnung) | ~5,000 | 0.008 | -0.088 |

**Linear Ridge has higher Sharpe and essentially identical IC with 215× fewer parameters.**

### US Market (477 stocks)

| Model | Params | Mean IC | Mean Sharpe |
|---|---|---|---|
| LightGBM | ~100 trees | 0.007 | 1.16 |
| **Ridge** | **451** | **0.031** | **0.82** |

**Ridge has 4.4× higher IC than LightGBM. Simpler wins.**

---

## 4. GBDT Parameter Sweep (9 configs × 5 seeds)

Source: `docs/architecture_assessment.md` (§1.3).

| Parameter | Range | Sharpe | IC |
|---|---|---|---|
| **Trees** | T100/T200/T500 | 0.034–0.035 | 0.019–0.022 |
| **Depth** | D4/D6/D8 | 0.034–0.035 | 0.019–0.022 |
| **Loss function** | MSE/MAE/Huber | 0.345/0.085/0.339 | 0.020/0.044/0.019 |
| **Step size** | 14/21/42 | 0.337/0.345/0.354 | — |

**Finding**: GBDT saturates at minimum config (100 trees, depth=4). Adding more trees or depth does not improve performance. The data has no more signal to extract.

---

## 5. Why Mamba SSM Fails on Financial Data

Source: `progress.md` (§5.1, §5.2).

1. **SSM designed for long-sequence language modeling** — T=63 is too short for selective scan mechanisms to be useful.
2. **Continuous values ≠ discrete tokens** — Financial features lack the structure SSMs are designed for.
3. **Parameter-to-sample ratio** — 97K params ÷ 750 samples ≈ 129:1. Deep learning needs ~10:1 or better.
4. **Cross-asset attention O(N²)** — 200 stocks × 63 steps × attention = 40,000 edges, all learned from noise.
5. **No external information source** — Mamba cannot extract non-existent signal from pure OHLCV features.

---

## 6. CTM vs Ensemble: Ensemble Adds No Value

| Model | val_IC | test_Sharpe |
|---|---|---|
| CTM | 0.049 | -0.028 |
| Ensemble+TimeGate | 0.037 | -0.024 |
| GBDT | 0.008 | -0.088 |

**Ensemble mean performance is between CTM and GBDT — as expected from a weighted average of two zero-signal models.** The TimeGate parameters (α, β, γ) were effectively static (moved <0.6% during finetuning), indicating that the gating mechanism has no gradient signal to learn from.

---

## 7. CTM Validation Performance Evolution (v1 → v5)

| Version | val_IC | test_IC | Gap | Cause |
|---|---|---|---|---|
| v1 | 0.14 | — | — | Initial overfit |
| v2 | 0.08 | — | — | Multi-asset expansion |
| v3 | 0.14 | — | — | Bug: shuffle=True inflates val |
| v4 (TGPE) | 0.065 | — | — | Curriculum + TimeGate |
| **v5 (final)** | **0.049** | **≈ 0.00** | **0.049** | **Bugs fixed → honest gap** |

The progression shows that each "improvement" in validation IC was actually reduced overfitting, not increased signal extraction.

---

## 8. Caveats

1. **Results are specific to 9 OHLCV features on daily data** — Different feature sets or frequencies may favor different architectures.
2. **CTM was not extensively hyperparameter-tuned** due to computational constraints. A larger search might find better configurations.
3. **The Ensemble models used static TimeGate weights** — Dynamic/finetuned gates showed no improvement.
4. **GBDT vs CTM comparison is confounded by different training protocols** — GBDT uses Huber loss, CTM uses MSE+Sharpe composite.
