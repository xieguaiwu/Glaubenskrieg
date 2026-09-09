# Glaubenskrieg v4 Experiment Plan

> **Date**: 2026-06-05 | **Designer**: Prometheus (architecture/planning)
> **Principle**: Statistical rigor over exploration — fewer configs, more seeds, more windows
> **v3 baseline**: Test Sharpe -0.39 ± 0.72 (n=9), 3/9 positive, ensemble ctm_weight=0.968, step_size=63 (4 windows)

---

## 0. Design Philosophy: What v4 Must Answer

v3 left one question unanswered: **Does CTM architecture produce a reliable positive Sharpe on daily return prediction?**

The answer requires three things v3 lacked:
1. **Enough walk-forward windows** for stable metric estimation (4→28)
2. **Enough seeds** for statistical significance testing (3→5-10)
3. **A clean GBDT baseline** with identical methodology for head-to-head comparison

Everything else (model capacity, loss function, n_assets scaling) is secondary. v4 prioritizes the primary question.

### Design Principles

| Principle | Implication |
|-----------|------------|
| **Fewer configs, more seeds** | 4-6 CTM configs × 5 seeds vs. v3's 9 configs × 1-3 seeds |
| **Proof of signal first** | If the primary experiment fails, secondary experiments are moot |
| **Honest GBDT baseline** | Same data, same windows, same seeds → apples-to-apples comparison |
| **Ascend-aware scheduling** | GPU experiments ordered by scientific priority; CPU GBDT runs in parallel |

---

## 1. Methodology: What's Preserved from v3, What Changes

### 1.1 Preserved (v3 fixes that work)

| Component | Setting | Reason |
|-----------|---------|--------|
| LR warmup clamping | `lr_warmup_epochs = min(config, max(5, n_epochs//5))` | Fixes v2's warmup>n_epochs bug |
| Early stopping metric | Validation loss (not Sharpe) | Avoids noisy-Sharpe early-stop from v2 |
| Grad clip + weight decay | 1.0 / 0.05 | Standard regularization |
| Cross-asset attention | Enabled (low-rank, rank=4) | Core CTM differentiator |
| Learnable loss weights | `LearnableWeights()` | Auto-balances MSE/directional/pinball |
| Sharpe loss warmup | 2000 steps pure MSE, 3000 step ramp | Prevents early variance collapse |
| Test evaluation protocol | Train on train+val, eval on held-out test | No look-ahead leakage |
| seq_len=63 | 63-day lookback window | Standard, matches quarterly patterns |

### 1.2 Changed

| Component | v3 | v4 | Rationale |
|-----------|----|----|-----------|
| **step_size** | 63 (4 windows) | **21 (28 windows)** primary; 42 (14 windows) secondary | Statistical power: 28 windows enables t-tests |
| **Ensemble** | Enabled (worthless: ctm_weight=0.968) | **Disabled** — pure CTM | Remove dead weight; focus on CTM signal |
| **Seeds per config** | 1-3 | **5 minimum** for all configs | Detect Sharpe > 0 with p<0.10 |
| **GBDT baseline** | Separate (mixed methodology) | **Same data, same windows, same seeds** | Apples-to-apples head-to-head |
| **n_epochs** | 100 (config default) | **50** for CTM, N/A for GBDT | 50 epochs + early stopping sufficient; speeds experiments |
| **Loss components** | MSE+Sharpe+Directional+Pinball | **Ablation: composite vs MSE-only** | Test whether multi-loss helps or hurts for this SNR |

### 1.3 Dropped

| Component | Reason |
|-----------|--------|
| Ensemble (CTM+GBDT fusion) | v3 proved worthless: ctm_weight=0.968, ensemble Sharpe ≈ CTM Sharpe |
| P3 three-stage fusion | Inherits ensemble worthlessness |
| Time-gated fusion variants (A/B/C) | Premature — need to prove CTM signal first |
| n_assets=1 (single-asset) | Cross-attention requires ≥2 assets; v3 showed multi-asset DA>60% |
| step_size=63 | Insufficient statistical power (4 windows) |

---

## 2. Experiment Matrix

### Group A: GBDT Baseline (CPU, Server 2)

**Purpose**: Establish the credible floor for daily return prediction with these exact features. The baseline must use identical methodology to CTM experiments (same data, same walk-forward windows, same seeds, same metrics).

**Runtime**: GBDT is CPU-only and fast (~2 min/seed with trees=200, depth=6, n=50, 28 windows). Run all Group A experiments in parallel with GPU Group B.

| ID | n_assets | trees | depth | loss | step_size | seeds | ~time/seed | ~total wall | Priority | Question |
|----|----------|-------|-------|------|-----------|-------|-----------|-------------|----------|----------|
| **A1** | 50 | 200 | 6 | mse | 21 | **10** | 2 min | 4 min (4×par) | **P0** | Primary baseline: 10-seed GBDT Sharpe distribution |
| A2 | 50 | 200 | 6 | mse | 42 | 5 | 1 min | 2 min | P1 | Speed/stat trade-off: 14 windows enough for GBDT? |
| A3 | 20 | 200 | 6 | mse | 21 | 5 | 1 min | 2 min | P2 | Scaling: does n=20 differ from n=50 for GBDT? |
| A4 | 50 | 200 | 6 | huber | 21 | 5 | 2 min | 2 min | P2 | Loss robustness: Huber vs MSE for GBDT |

**Seeds A1**: 42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768
**Seeds A2-A4**: 42, 123, 456, 789, 1024

**Wall time**: All Group A < 10 min total with 4 parallel workers (CPU multi-core).

---

### Group B: CTM Core (GPU, Server 1, Ascend 910B × 2)

**Purpose**: Answer the primary question: does CTM produce positive average test Sharpe with adequate statistical power?

**Runtime model** (per seed, Ascend 910B):
- Baseline: n=50, dim=64, 28 windows × ~6.25 min/window (n_epochs=50) = **~2.9h/seed**
- dim=128: ~2× slower Mamba kernel + larger attention → **~5h/seed**
- n=20: ~1.5× faster (fewer assets → smaller attention, same Mamba) → **~2.0h/seed**
- step_size=42: 14/28 = 0.5× windows → **~1.5h/seed** (dim=64)

| ID | n_assets | model_dim | state_dim | n_layers | step_size | n_epochs | seeds | ~h/seed | ~total GPU-h | Priority | Question |
|----|----------|-----------|-----------|----------|-----------|----------|-------|---------|-------------|----------|----------|
| **B1** | 50 | 64 | 8 | 2 | 21 | 50 | **5** | 2.9 | 14.5 | **P0** | **Primary: CTM with 28 windows — does it beat 0?** |
| B2 | 50 | 64 | 8 | 2 | 42 | 50 | **5** | 1.5 | 7.5 | P1 | Speed/stat trade-off: 14 windows sufficient for CTM? |
| B3 | 50 | 128 | 16 | 2 | 21 | 50 | **5** | 5.0 | 25.0 | P2 | Capacity: does model_dim=128 help or overfit? |
| B4 | 20 | 64 | 8 | 2 | 21 | 50 | **5** | 2.0 | 10.0 | P2 | Scaling: does n=20 perform differently from n=50? |
| B5 | 50 | 64 | 8 | 2 | 21 | 50 | **5** | 2.9 | 14.5 | P2 | Loss ablation: MSE-only (no Sharpe/directional/pinball) |

**Seeds**: 42, 123, 456, 789, 1024 (extends v3's 42, 123, 456)

**Config for B1-B4**: `configs/default.yaml` with overrides via CLI:
```bash
--config configs/default.yaml \
--data-dir /root/stock_data \
--multi-asset --n-assets {n} \
--step-size {step} --n-epochs 50 \
--seed {seed} --device cuda
```

**Config for B5** (MSE-only):
```bash
# Override loss weights: MSE only
--lambda-mse 1.0 --lambda-sharpe 0.0 \
--lambda-directional 0.0 --lambda-pinball 0.0 --lambda-reg 0.01
```

### Total GPU time (Group B):
- B1: 14.5 GPU-h → ~7.3h wall (2 GPUs, sequential seeds per GPU)
- B2: 7.5 GPU-h → ~3.8h wall
- B3: 25.0 GPU-h → ~12.5h wall
- B4: 10.0 GPU-h → ~5.0h wall
- B5: 14.5 GPU-h → ~7.3h wall
- **Grand total**: 71.5 GPU-h → **~36h wall** (2 GPUs, all experiments sequential)

### Optimized scheduling (2 GPUs, pipelined):

| GPU 0 | GPU 1 | Wall time |
|-------|-------|-----------|
| B1-s42 (2.9h) | B1-s123 (2.9h) | 0-2.9h |
| B1-s456 (2.9h) | B1-s789 (2.9h) | 2.9-5.8h |
| B1-s1024 (2.9h) → B2-s42 (1.5h) | B4-s42 (2.0h) | 5.8-8.7h |
| B2-s123 (1.5h) → B2-s456 (1.5h) | B4-s123 (2.0h) | 8.7-10.7h |
| B2-s789 (1.5h) → B2-s1024 (1.5h) | B4-s456 (2.0h) | 10.7-13.7h |
| B5-s42 (2.9h) | B4-s789 (2.0h) + B4-s1024 (2.0h) | 13.7-17.7h |
| B5-s123 (2.9h) | B3-s42 (5.0h) | 17.7-22.7h |
| B5-s456 (2.9h) | B3-s123 (5.0h) | 22.7-27.7h |
| B5-s789 (2.9h) | B5-s1024 (2.9h) + B3-s456 (5.0h) | 27.7-32.7h → done |

**Optimized wall time**: ~28h (sequential overlap reduces from ~36h theoretical)

**If time-constrained**: Drop B3 first (dim=128 is speculative, highest cost). Then: ~23h wall.

---

### Group C: CTM Sensitivity (Lower Priority, Conditional on B1 Success)

Run ONLY if B1 shows mean test Sharpe > 0 with p < 0.10 (one-sided t-test across 5 seeds).

| ID | n_assets | Param varied | Values | Question |
|----|----------|-------------|--------|----------|
| C1 | 50 | n_layers | 1, 3 | Does depth matter? (v3 used 2, v2 used 3) |
| C2 | 50 | dropout | 0.1, 0.3, 0.5 | Regularization sensitivity |
| C3 | 50 | train_window | 500, 1500 | Window length trade-off |
| C4 | 50 | n_epochs | 25, 100 | Training budget sensitivity |
| C5 | 50 | seq_len | 21, 126 | Lookback window length |

Each: 5 seeds × 2 values × ~2.9h ≈ 29 GPU-h per sweep. Run only if B1 succeeds.

---

## 3. Scientific Questions & Expected Outcomes

### Q1: Does CTM with 28 windows produce reliably positive Sharpe?

**Null hypothesis H₀**: μ_sharpe ≤ 0
**Test**: One-sided t-test across 5 seeds (B1), α=0.10
**Required effect**: With σ≈0.72 (from v3), need μ≥0.55 to detect with n=5 at p<0.10.
  - t_crit(4, 0.10) = 1.533
  - μ ≥ 1.533 × 0.72/√5 = 0.49
  - This is a large required effect. 10 seeds (as in GBDT A1) would need μ ≥ 1.383 × 0.72/√10 = 0.31

**Implication**: If the true Sharpe is between 0 and 0.3, we won't detect it with 5 seeds. This is an honest limitation. Report non-significance honestly — null results in ML-for-finance are publishable (see research_limits_ml_finance.md).

**Alternative**: Pool all 140 window-level results (5 seeds × 28 windows) for a more powerful test, but window-level results are not independent (same seed → correlated windows).

### Q2: Is model_dim=128 better than model_dim=64?

**Hypothesis**: Larger models overfit more on low-SNR financial data.
**Expected**: B3 Sharpe ≤ B1 Sharpe (consistent with research_overfitting_finance.md: simpler models often win).
**Test**: Paired t-test across 5 seeds (same seeds for B1 and B3).

### Q3: Is 14 windows (step_size=42) sufficient?

**Hypothesis**: 14 windows provide adequate statistical power with 40% less GPU time.
**Test**: Compare B2 mean and std vs B1. If B2 Sharpe is within 0.1 of B1, 14 windows may suffice for future experiments.
**Risk**: With only 14 windows, the walk-forward metric is noisier → harder to detect small effects.

### Q4: Does n_assets affect CTM performance?

**Hypothesis**: More assets → more cross-attention signal → better Sharpe (diminishing returns after ~20-30).
**Test**: B4 (n=20) vs B1 (n=50). Paired comparison.
**Note**: The O(N²) cross-attention cost is dominated by Mamba SSM on Ascend 910B, so n=20 doesn't save as much time as expected.

### Q5: Does composite loss help or hurt?

**Hypothesis**: With low SNR, the Sharpe loss component may still cause subtle variance collapse even with warmup. MSE-only may produce more stable training.
**Test**: B5 vs B1. This is a critical ablation — if MSE-only outperforms composite loss, it simplifies all future experiments.

---

## 4. Parameter Details

### 4.1 Fixed Parameters (all Group B experiments)

```yaml
model:
  input_dim: 9            # from compute_all_features()
  seq_len: 63
  conv_kernel: 3
  output_dim: 1
  dropout: 0.2
  use_decomp: false
  bidirectional: false
  parallel_scan: true
  return_hidden: false
  use_cross_attention: true

scaling:
  embedding_dim: null     # = model_dim

loss:
  lambda_sharpe: 0.1
  lambda_directional: 1.0
  lambda_pinball: 0.1
  lambda_reg: 0.01
  pinball_tau: 0.05

trainer:
  train_window: 1000
  val_window: 200
  purge_period: 126
  batch_size: 8
  lr: 0.0003
  weight_decay: 0.05
  grad_clip: 1.0
  patience: 10
  warmup_steps: 2000
  ramp_steps: 3000
  lr_warmup_epochs: 10   # clamped to min(10, n_epochs//5) = min(10, 10) = 10

device: cuda
```

### 4.2 Parameter Count Estimates

| Config | model_dim | state_dim | n_layers | n_assets | ~Params | Notes |
|--------|-----------|-----------|----------|----------|---------|-------|
| B1 (v4 baseline) | 64 | 8 | 2 | 50 | ~97K | Same as v3 |
| B2 (step_size=42) | 64 | 8 | 2 | 50 | ~97K | Same model, fewer windows |
| B3 (dim=128) | 128 | 16 | 2 | 50 | ~388K | 4× params vs B1 |
| B4 (n=20) | 64 | 8 | 2 | 20 | ~97K | Same model, fewer assets |
| B5 (MSE-only) | 64 | 8 | 2 | 50 | ~97K | Same model, simpler loss |

### 4.3 Walk-Forward Window Counts

With B ≈ 1938 sequences (2000 trading days - seq_len + 1), train_window=1000, val_window=200, purge_period=126:

| step_size | Max pos | n_windows | Per-seed train time (50 epochs) |
|-----------|---------|-----------|-------------------------------|
| 21 | pos ≤ 612 | **28** | ~2.9h (primary) |
| 42 | pos ≤ 612 | **14** | ~1.5h (fast variant) |
| 63 | pos ≤ 612 | **9** | ~1.0h (v3 used this, got 4) |

Note: v3 reported 4 windows with step_size=63, but the calculation suggests ~9. The discrepancy may be due to shorter common date intersection for n=50 assets or larger train_window. We'll use the actual observed n_windows from each run.

---

## 5. Execution Plan

### Phase 0: Pre-flight (immediate, Server 2 CPU)
1. Verify GBDT C++ module imports correctly
2. Verify stock data CSV integrity (all 50 assets have sufficient data)
3. Run one quick GBDT seed to confirm walk-forward window count with step_size=21

### Phase 1: GBDT Baseline A1 (0-30 min, Server 2 CPU)
- 10 seeds, 4 parallel workers
- Output: `results/v4/gbdt_n50_step21_s{seed}.json`
- Confirm: mean Sharpe, std Sharpe, IC, directional accuracy across 28 windows × 10 seeds

### Phase 2: CTM Primary B1 (0-12h, Server 1 GPU × 2)
- 5 seeds, pipelined on 2 GPUs
- Monitor: per-epoch train/val loss, gate mean (if time-gating enabled), pred_std (variance collapse detector)
- Output: `results/v4/ctm_n50_dim64_step21_s{seed}.json`

### Phase 3: CTM B2 + B4 (parallel, 8-18h)
- B2: 5 seeds, step_size=42
- B4: 5 seeds, n_assets=20
- These answer: is 14 windows enough? does n_assets matter?
- Output: `results/v4/ctm_n50_dim64_step42_s{seed}.json`, `results/v4/ctm_n20_dim64_step21_s{seed}.json`

### Phase 4: CTM B5 + B3 (serial, 18-28h)
- B5: 5 seeds, MSE-only loss (critical ablation)
- B3: 5 seeds, model_dim=128 (speculative, highest cost)
- If B1 fails to show positive Sharpe, skip B3 entirely

### Phase 5: Analysis (post-experiment)
1. Aggregate all results into a single comparison table
2. Paired t-tests for all head-to-head comparisons
3. Window-level analysis: Sharpe distribution across 28 windows
4. Learning curve: Sharpe vs epoch within each window
5. Decision: proceed to Group C (sensitivity) or pivot to GBDT-only?

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| CTM Sharpe still negative (B1 fails) | **High** (67% based on v3) | Core hypothesis disproven | Accept null result; report honestly; pivot to GBDT-only paper |
| Ascend 910B OOM with dim=128 (B3) | Medium | Experiment lost | Monitor VRAM on first batch; fall back to dim=96 if OOM |
| Mamba kernel crashes on certain seeds | Low-Medium | Lost seed, re-run | Known issue with Ascend compatibility; use seed+1 if crash (document) |
| 28 windows still insufficient for significance | Medium | Weak conclusions | Accept as methodological limitation; note in paper |
| GBDT baseline Sharpe also ~0 | Low | Means features have no signal | This is a valid finding (see research_limits_ml_finance.md) |
| Server power loss / SSH disconnect | Low | Lost partial results | Use nohup; save per-window checkpoints; resume from last completed seed |
| GPU queue contention with other experiments | Medium | Delayed schedule | v4 is the only planned GPU workload; coordinate timing |

---

## 7. Output Specification

### Per-Seed Output (`results/v4/{type}_{config}_s{seed}.json`)

```json
{
  "config": {
    "n_assets": 50, "model_dim": 64, "state_dim": 8, "n_layers": 2,
    "step_size": 21, "n_epochs": 50, "seed": 42,
    "loss": "composite"  // or "mse_only" for B5
  },
  "n_windows": 28,
  "mean_sharpe": 0.12,
  "std_sharpe": 0.45,
  "max_sharpe": 0.89,
  "min_sharpe": -1.23,
  "mean_ic": 0.015,
  "std_ic": 0.04,
  "mean_directional_accuracy": 0.52,
  "test_metrics": {
    "test_loss": 0.87,
    "test_sharpe": -0.15,
    "test_directional_accuracy": 0.51
  },
  "windows": [
    {"window": 0, "start": 0, "end": 1326, "best_sharpe": 0.34, "epochs_run": 42},
    ...
  ],
  "runtime_seconds": 10440
}
```

### Aggregate Analysis Output (`results/v4/analysis.json`)

```json
{
  "experiments": {
    "B1": {
      "n_seeds": 5,
      "mean_test_sharpe": -0.05,
      "std_test_sharpe": 0.18,
      "p_value_h0_sharpe_le_0": 0.31,
      "p_value_vs_gbdt": 0.67,
      "mean_window_sharpe": 0.08,
      "positive_window_fraction": 0.54,
      "mean_params": 97234,
      "total_gpu_hours": 14.5
    }
  },
  "comparisons": {
    "B1_vs_A1": {
      "test": "paired_t",
      "statistic": -1.23,
      "p_value": 0.29,
      "conclusion": "CTM not significantly different from GBDT"
    }
  }
}
```

---

## 8. Go/No-Go Criteria After v4

### Proceed to Group C (sensitivity sweeps) if:
- B1 (CTM, 28 windows) mean test Sharpe > 0 with p < 0.10, OR
- B1 mean test Sharpe > A1 (GBDT) mean test Sharpe with p < 0.10

### Pivot to GBDT-only paper if:
- B1 mean test Sharpe ≤ 0 AND A1 (GBDT) mean test Sharpe ≤ 0
- → Report: "Neither CTM nor GBDT produce reliable daily return predictions with these features"

### Pivot to GBDT-primary paper if:
- A1 (GBDT) mean test Sharpe > 0 with p < 0.10 AND B1 ≤ A1
- → Report: "GBDT outperforms Mamba-based CTM for daily return prediction"

### Proceed to CTM-primary paper if:
- B1 mean test Sharpe > A1 with p < 0.10
- → This is the desired outcome; validates 6 months of architecture development

---

## 9. Quick-Start Commands

### On Server 2 (GBDT Baseline):
```bash
cd /root/Glaubenskrieg
mkdir -p results/v4

# A1: 10-seed GBDT baseline, 28 windows
for s in 42 123 456 789 1024 2048 4096 8192 16384 32768; do
  nohup python scripts/train_gbdt_only.py \
    --data-dir /root/stock_data --n-assets 50 \
    --gbdt-trees 200 --gbdt-depth 6 \
    --step-size 21 --seed $s \
    --output results/v4/gbdt_n50_step21_s${s}.json \
    > logs/v4_gbdt_s${s}.log 2>&1 &
done
```

### On Server 1 (CTM Primary):
```bash
cd /root/Glaubenskrieg
mkdir -p results/v4 logs

# B1: CTM n=50, dim=64, 28 windows, 5 seeds
# GPU 0: seeds 42, 456, 1024
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py \
  --config configs/default.yaml \
  --data-dir /root/stock_data --multi-asset --n-assets 50 \
  --step-size 21 --n-epochs 50 --seed 42 \
  --device cuda --output results/v4/ctm_n50_dim64_step21_s42.json \
  > logs/v4_ctm_b1_s42.log 2>&1 &

# GPU 1: seeds 123, 789
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train.py \
  --config configs/default.yaml \
  --data-dir /root/stock_data --multi-asset --n-assets 50 \
  --step-size 21 --n-epochs 50 --seed 123 \
  --device cuda --output results/v4/ctm_n50_dim64_step21_s123.json \
  > logs/v4_ctm_b1_s123.log 2>&1 &
```

---

## Appendix A: Rejected Alternatives

### A.1 Why not test more model_dim values (32, 96, 128, 256)?
**Rejected**: Diminishing returns on experimental budget. The primary question is signal/noise, not optimal architecture size. Two values (64 → baseline, 128 → "more capacity") bracket the interesting range. If dim=128 significantly improves Sharpe, future work can explore the continuum.

### A.2 Why not test n_assets scaling from 5-200?
**Rejected**: n=20 and n=50 bracket the regime where cross-attention might help (≥20 assets for meaningful correlations). Below 20, cross-attention is questionable. Above 50, Ascend 910B training time becomes prohibitive (>6h/seed for n=100).

### A.3 Why not test different backbone architectures (LSTM, Transformer)?
**Rejected**: Out of scope for v4. The question is whether the existing Mamba-based CTM architecture works. Architecture comparison is v5+ work.

### A.4 Why not use Bayesian optimization for hyperparameters?
**Rejected**: With 5 seeds per config, hyperparameter sweeps would multiply the experimental budget by 10-100×. The literature (research_limits_ml_finance.md) shows that hyperparameter tuning provides marginal gains for daily return prediction — the bottleneck is signal, not optimization.

### A.5 Why not keep ensemble as an ablation?
**Rejected**: v3 conclusively showed ensemble adds nothing (ctm_weight=0.968, effectively CTM-only). Re-running ensemble would consume GPU hours for a confirmed-negative result. If CTM shows positive Sharpe in v4, ensemble can be revisited in v5.

---

## Appendix B: Statistical Power Analysis

### Minimum Detectable Effect (n=5 seeds, one-sided t-test, α=0.10)

With σ_sharpe ≈ 0.72 (from v3 n=9 experiments):

| Seeds (n) | t_crit(df, 0.10) | MDE (μ units) | Detectable if true μ ≥ |
|-----------|-----------------|---------------|----------------------|
| 5 | t(4) = 1.533 | 1.533 × 0.72/√5 | 0.49 |
| 10 | t(9) = 1.383 | 1.383 × 0.72/√10 | 0.31 |
| 20 | t(19) = 1.328 | 1.328 × 0.72/√20 | 0.21 |

**Interpretation**: With 5 seeds, we can only detect a Sharpe ≥ 0.49. This is a large effect. For context, the research literature suggests realistic daily return prediction Sharpes of 0.0-0.3.

**With the v4 A1 baseline (10 GBDT seeds)**: MDE = 0.31. If GBDT's true Sharpe is 0.2, we won't detect it — but we'll have an honest confidence interval.

**Mitigation**: Use window-level pooling (5 seeds × 28 windows = 140 observations, but not independent) or Bayesian hierarchical modeling for post-hoc analysis. Also, the cross-seed std may decrease with more windows (better estimation → each seed more reliable), tightening the CI.

### Power Curves

```
Power (n=5, σ=0.72):
  μ=0.0 → power=0.05  (correct: Type I error)
  μ=0.2 → power=0.12  (78% chance of Type II error)
  μ=0.4 → power=0.35  (65% chance of Type II error)
  μ=0.6 → power=0.63  (barely adequate)
  μ=0.8 → power=0.85  (good)

Power (n=10, σ=0.72):
  μ=0.0 → power=0.05
  μ=0.2 → power=0.20
  μ=0.4 → power=0.60
  μ=0.6 → power=0.90
```

**Honest conclusion**: ~~v4 is underpowered to detect small effects (Sharpe < 0.3). This is a consequence of the Ascend 910B's Mamba kernel slowdown — on normal GPUs, we'd run 20+ seeds.~~ **Update 2026-06-06**: Hardware upgraded to V100+3090 (25-50× faster). See v5 addendum below.

---

## 6. v5 Addendum (2026-06-06)

### What changed
- **Hardware**: Ascend 910B → V100 (32GB) + RTX 3090 (24GB), native CUDA
- **Speed**: 21-35ms/forward vs 500-1000ms → experiments complete in ~10 min
- **Root cause fixes**: 5 critical bugs identified and fixed (see docs/experiments.md)

### v4 TGPE Results (concluded, negative)
TGPE v4 was uniformly worse than v3 across both variants and all seeds:
- v3 CTM IC=0.142 → v4 Curriculum IC=0.065
- v3 Ensemble IC=0.049 → v4 TimeGate IC=0.028

Five root causes identified; all fixed in v5.

### v5 Running (2026-06-06 12:00)
- 3 variants × 5 seeds × 2 servers
- GBDT baseline (CPU, done)
- CTM baseline (GPU, training)
- Ensemble+TimeGate+Finetune (GPU, training)
- Results: /root/results/v5/
