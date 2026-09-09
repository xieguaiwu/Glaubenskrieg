# Negative Results: Meta-Labeling, Cross-Sectional Ranking, Curriculum Learning, TimeGate

> **Source files**: `progress.md` (§2 Stages 8–10, §3), `.omo/session_context.md` (§3), `docs/architecture_assessment.md`

---

## 1. Meta-Labeling (López de Prado §3.7)

Source: `progress.md` (Stage 10), `.omo/session_context.md` (§3).

### Design
- Primary model: LightGBM lambdarank → directional predictions
- Meta-model: XGBoost binary classifier, 8-dimensional meta-features
- Walk-forward + hold-out validation, 5 seeds

### Results

| Seed | Δ Sharpe (WF) | Δ Sharpe (Hold-out) | Meta Precision |
|---|---|---|---|
| 42 | -0.15 | -0.22 | ~0.50 |
| 123 | -0.08 | -0.31 | ~0.50 |
| 456 | -0.12 | -0.18 | ~0.50 |
| 789 | -0.19 | -0.27 | ~0.50 |
| 1024 | -0.11 | -0.14 | ~0.50 |
| **Mean** | **-0.13** | **-0.22** | **~0.50** |

### Conclusion
Meta-Labeling **cannot create signal where none exists**. López de Prado's framework requires:
1. Primary model with positive expected IC → **not met (IC ≈ 0)**
2. Meta-features correlated with primary model error patterns → **not met (same OHLCV features)**

Result: Meta-Precision ≈ 0.50 (random), Δ Sharpe uniformly negative.

---

## 2. Cross-Sectional Ranking (lambdarank)

Source: `progress.md` (Stage 8), `.omo/session_context.md` (§3).
Reference: López de Prado's "Proposal 1" — cross-sectional ranking should outperform time-series.

### Design
- CS z-scores + 5/21/63 day momentum ranking
- LightGBM lambdarank (objective=lambdarank, metric=NDCG)
- Walk-forward protocol, 3 seeds

### Results

| Seed | Walk-Forward Rank IC | Test Rank IC |
|---|---|---|
| 42 | Negative | Negative |
| 123 | Negative | Negative |
| 456 | Negative | Negative |
| **All** | **Negative** | **Negative** |

### Conclusion
Cross-sectional ranking paradigm **produces negative Rank IC** — worse than random. The CS approach is even less effective than time-series for OHLCV-only features. This refutes López de Prado's suggestion that CS might be "easier" than TS.

---

## 3. Curriculum Learning (TGPE Variant C)

Source: `progress.md` (Stage 4).

### Design
- Training progressively reduces CTM weight, increases GBDT weight
- Scheduling: linear/step-based dropout rates
- CTM+Curriculum vs Ensemble comparison

### Results

| Experiment | Val IC | Test Sharpe |
|---|---|---|
| CTM+Curriculum (seed 123, S2) | 0.065 | -0.09 |
| CTM+Curriculum (seed 456, S1) | 0.040 | -0.16 |
| Ensemble+TimeGate (seed 123, S2) | 0.028 | -0.05 |

### Root Causes

| # | Issue | Severity |
|---|---|---|
| 1 | Curriculum Dropout contaminates test-prep training | 🔴 P0 |
| 2 | Test-prep model independent of walk-forward (warm-start missing) | 🔴 P0 |
| 3 | GBDT quality gating inconsistent (models accepted despite zero signal variance) | 🟡 P1 |

Curriculum learning requires non-zero signals to learn the curriculum schedule. With zero-signal GBDT and CTM, curriculum weighting is meaningless.

---

## 4. TimeGate (TGPE Variants A & B)

Source: `progress.md` (Stage 4), `docs/architecture_assessment.md` (Appendix D).

### Design
- **TimeDecayGate** (A): α, β, γ learnable decay parameters weighting CTM vs GBDT over time
- **Time-Amp GBDT Modulator** (B): Attention-based amplification of recent GBDT signals
- V4 experiment: 3 variants × 3 seeds

### Results: Gate Parameter Movement

| Parameter | Initial | After 50 epochs | Movement |
|---|---|---|---|
| α | 1.0 | 1.006 | +0.6% |
| β | 1.0 | 1.004 | +0.4% |
| γ | 0.5 | 0.497 | -0.6% |

**All three parameters moved <1%** over 50 finetuning epochs with learning rate 1e-4. The gating mechanism receives near-zero gradient signal because both CTM and GBDT produce identical (zero) signals.

### TGPE Final Verdict

| Variant | Test Sharpe | vs v3 baseline |
|---|---|---|
| TGPE-A (TimeDecayGate) | Negative | ↓ Worse |
| TGPE-B (Time-Amp Modulator) | Negative | ↓ Worse |
| TGPE-C (Curriculum Dropout) | Negative | ↓ Worse |
| **Any TGPE** | **Uniformly worse** | **❌ All failed** |

---

## 5. Feature Engineering: EMA Smoothing

Source: `progress.md` (Stage 7), `docs/architecture_assessment.md` (Appendix D).

### Design
- Bai et al. (Symmetry 2026) methodology: EMA smooth + walk-forward LightGBM + DM-test
- EMA α ∈ {1 (raw), 10 (smooth)}
- 5 seeds

### Results

| Config | test_IC | test_Sharpe | DM p | Sig@5% |
|---|---|---|---|---|
| EMA=1 (raw) | -0.004±0.007 | 0.62±0.03 | 0.014-0.124 | 4/5 |
| EMA=10 (smooth) | -0.013±0.007 | 0.47±0.02 | 0.136-0.220 | 0/3 |

### Key Findings
1. **EMA reduces prediction quality** (Sharpe 0.62 → 0.47, DM significance disappears)
2. test_IC ≈ 0 regardless of smoothing — feature engineering cannot create new information
3. DM test serves as effective stopping criterion: when smoothing causes 4/5 → 0/3 significance, stop
4. The positive Sharpe (0.62) with IC = -0.004 is the Sharpe paradox again

---

## 6. Across-Model Correlation

Source: `docs/architecture_assessment.md` (§1.4).

### GBDT Parameter Insensitivity

| Config Range | Sharpe Range | IC Range |
|---|---|---|
| Trees (100–500) | 0.34–0.35 | 0.019–0.022 |
| Depth (4–8) | 0.34–0.35 | 0.019–0.022 |
| Loss (MSE/MAE/Huber) | 0.085–0.345 | 0.019–0.044 |

**GBDT is completely insensitive to parameter variation** — the data has no structure for different parameter configurations to exploit differently.

---

## 7. Summary: All 6 Paradigms Failed

| # | Paradigm | Method | Key Metric | Verdict |
|---|---|---|---|---|
| 1 | Deep Learning | CTM (Mamba SSM) | Test IC ≈ 0.00 | ❌ No signal |
| 2 | Gradient Boosting | GBDT (Hoffnung C++) | Val IC = 0.008 | ❌ No signal |
| 3 | Hybrid Ensemble | Ensemble+TimeGate (TGPE) | Test Sharpe uniformly negative | ❌ Degrades |
| 4 | Feature Engineering | LightGBM + EMA + DM test | Test IC = -0.004 | ❌ Sharpe = artifact |
| 5 | Cross-Sectional Ranking | lambdarank + CS z-scores | Test Rank IC negative | ❌ Worse than TS |
| 6 | Meta-Labeling | XGBoost binary filter | Δ Sharpe = -0.22 | ❌ Reduces Sharpe |

**Six independent paradigms, each using different model families (Mamba, GBDT, XGBoost, LightGBM, Ridge) and evaluation protocols, all converge on the same result: zero detectable daily return predictability from OHLCV-derived features.**

---

## 8. Caveats

1. **Negative results are dataset-specific** — These results apply only to the specific 9 OHLCV features tested. Adding non-price data sources may change outcomes.
2. **Curriculum learning might work with stronger signal** — The failure is due to zero base signal, not curriculum design per se.
3. **Meta-Labeling requires a positive-IC primary model** — Our results confirm this prerequisite, not refute the method.
4. **Cross-sectional ranking needs stronger features** — CS z-scores from OHLCV features provide zero ranking signal.
