# Glaubenskrieg Phase 2 Experiment Plan

> **Date**: 2026-06-05 | **Designer**: Prometheus (architecture/planning)
> **Principle**: Repeat-multiple-times — every config gets ≥5 seeds for statistical power
> **Baseline**: GBDT v2 (trees=200, depth=6, n_assets=50, 28 windows, sharpe=0.34-0.39, IC≈0.02)

---

## 1. Current State & Resource Allocation

| Resource | Status | Available | Est. Free |
|----------|--------|-----------|-----------|
| Server 2 CPU | IDLE (load 2.5) | **NOW** | — |
| Server 1 GPU 0 | CTM seed 42/123 training | ~3h | ~16:00 |
| Server 1 GPU 1 | CTM seed 456 queued | after GPU 0 | ~19:00+ |
| Server 2 GPU 0 | Ensemble old test-prep | ~1h | ~14:00 |

**Key insight**: Server 2 CPU is completely idle. GBDT experiments can begin immediately and run in parallel with all ongoing GPU work. This is the highest-leverage action.

---

## 2. Experiment Matrix

### Tier 1 — GBDT Parameter Sweep (Server 2 CPU, START NOW)

#### 2.1 Trees × Depth Grid (Core Sensitivity Analysis)
**Scientific question**: How sensitive is GBDT performance to model complexity? Does overfitting increase with depth?

| Config ID | Trees | Depth | Est. Time/Seed | Seeds | Total Runs | Est. Total | Scientific Value |
|-----------|-------|-------|---------------|-------|------------|------------|-----------------|
| T100-D4 | 100 | 4 | ~25s | 5 | 5 | ~2 min | Low — underfit bound |
| T100-D6 | 100 | 6 | ~55s | 5 | 5 | ~5 min | Medium — shallow baseline |
| T100-D8 | 100 | 8 | ~95s | 5 | 5 | ~8 min | Medium — depth vs trees |
| T200-D4 | 200 | 4 | ~45s | 5 | 5 | ~4 min | Medium — wide shallow |
| **T200-D6** | **200** | **6** | **~95s** | **5** | **5** | **~8 min** | **BASELINE (3 done, +2)** |
| T200-D8 | 200 | 8 | ~180s | 5 | 5 | ~15 min | High — overfit test |
| T500-D4 | 500 | 4 | ~105s | 5 | 5 | ~9 min | Medium — many shallow trees |
| T500-D6 | 500 | 6 | ~240s | 5 | 5 | ~20 min | High — complex baseline |
| T500-D8 | 500 | 8 | ~420s | 5 | 5 | ~35 min | **High — worst-case overfit** |

**Total**: 9 combos × 5 seeds = 45 runs, ~106 min serial, ~30 min with 4× parallel
**Fixed params**: n_assets=50, loss=mse, step_size=21, train_window=1000, val_window=200
**Seeds**: 42, 123, 456 (already done for T200-D6), 789, 1024

#### 2.2 Loss Function Comparison
**Scientific question**: Does the loss function matter? Is MSE appropriate for financial returns?

| Config ID | Loss | Est. Time/Seed | Seeds | Total Runs | Est. Total |
|-----------|------|---------------|-------|------------|------------|
| LOSS-MSE | mse | ~95s | 5 | 5 | ~8 min |
| LOSS-MAE | mae | ~95s | 5 | 5 | ~8 min |
| LOSS-HUBER | huber | ~95s | 5 | 5 | ~8 min |

**Total**: 3 × 5 = 15 runs, ~24 min serial, ~8 min with 3× parallel
**Fixed params**: trees=200, depth=6, n_assets=50, step_size=21
**Note**: LOSS-MSE = T200-D6 baseline (reuse results)

#### 2.3 10-Seed Baseline (Statistical Power)
**Scientific question**: With 10 seeds, can we establish a credible p-value for sharpe > 0?

| Config ID | Seeds | Est. Time | Notes |
|-----------|-------|-----------|-------|
| BASELINE-10 | 42,123,456,789,1024,2048,4096,8192,16384,32768 | ~16 min | Extends existing 3 seeds |

**Total**: +7 seeds (42,123,456 already done), ~11 min
**Fixed params**: trees=200, depth=6, n_assets=50, loss=mse, step_size=21

#### 2.4 Step Size / Window Count Sweep
**Scientific question**: How stable are metrics across different numbers of walk-forward windows?

| Config ID | Step Size | ~Windows | Est. Time/Seed | Seeds | Est. Total |
|-----------|-----------|----------|---------------|-------|------------|
| STEP-14 | 14 | ~39 | ~135s | 5 | ~11 min |
| STEP-21 | 21 | ~28 | ~95s | 5 | ~8 min |
| STEP-42 | 42 | ~14 | ~48s | 5 | ~4 min |

**Total**: 3 × 5 = 15 runs, ~23 min serial
**Scientific value**: Critical — tests whether conclusions depend on window granularity
**Fixed params**: trees=200, depth=6, n_assets=50, loss=mse

---

### Tier 2 — CTM Scaling (Server 1 GPU, QUEUED)

#### 2.5 CTM n_assets Scaling
**Scientific question**: Does CTM's cross-attention mechanism benefit from more assets, or does the O(N²) cost outweigh any signal?

| Config ID | n_assets | Est. Time/Seed | Seeds | Total Runs | Est. Total |
|-----------|----------|---------------|-------|------------|------------|
| CTM-N20 | 20 | ~30 min | 3 | 3 | ~1.5h |
| CTM-N50 | 50 | ~3h | 3 | 3 | ~9h |
| CTM-N80 | 80 | ~8h | 3 | 3 | ~24h |

**Seeds**: 42, 123, 456
**Fixed params**: model_dim=64, state_dim=16, n_layers=3, batch_size=8, n_epochs=50
**Config**: configs/default.yaml with --n-assets override
**CTM-N50**: seeds 42, 123 already running; seed 456 queued → reuse results

#### 2.6 CTM More Seeds (Statistical Power)
**Scientific question**: What is the variance of CTM performance across seeds?

| Config ID | Added Seeds | Est. Time/Seed | Est. Total |
|-----------|------------|---------------|------------|
| CTM-N50-S5 | 789, 1024 | ~3h | ~6h |

**Prerequisite**: Tier 1 completion (to confirm CTM is worth the GPU time)

---

### Tier 3 — Follow-up (If Resources Permit)

#### 3.1 GBDT subsample Sweep
| Config ID | subsample_row | subsample_col | Seeds | Est. Total |
|-----------|--------------|--------------|-------|------------|
| SUB-06 | 0.6 | 0.6 | 5 | ~8 min |
| SUB-08 | 0.8 | 0.8 | 5 | ~8 min |
| SUB-10 | 1.0 | 1.0 | 5 | ~8 min |

**Fixed params**: trees=200, depth=6

#### 3.2 GBDT Learning Rate
| Config ID | lr | Seeds | Est. Total |
|-----------|-----|-------|------------|
| LR-005 | 0.05 | 5 | ~8 min |
| LR-010 | 0.10 | 5 | ~8 min |
| LR-020 | 0.20 | 5 | ~8 min |

#### 3.3 Ensemble v2 (n_assets=50)
Already queued. Compare ensemble performance vs CTM-only and GBDT-only at n_assets=50.

---

## 3. Priority Matrix (Execution Order)

| Priority | Experiment | Why Now? | Server | Est. Runtime | Blocks |
|----------|-----------|----------|--------|-------------|--------|
| **P0** | GBDT T×D Grid | CPU idle, fastest ROI | S2 CPU | ~30 min wall | Nothing |
| **P0** | GBDT 10-Seed Baseline | Statistical power, fast | S2 CPU | ~11 min | Nothing |
| **P1** | GBDT Loss Sweep | Completes GBDT picture | S2 CPU | ~8 min wall | Nothing |
| **P2** | GBDT Step Size | Window stability check | S2 CPU | ~8 min wall | Nothing |
| **P3** | CTM N20 | CTM scaling lower bound | S1 GPU0 | ~1.5h | GPU0 free |
| **P4** | CTM N80 | CTM scaling upper bound | S1 GPU0 | ~24h | After N20 |
| **P5** | GBDT subsample | Fine-tuning, optional | S2 CPU | ~10 min | Nothing urgent |
| **P6** | GBDT LR sweep | Fine-tuning, optional | S2 CPU | ~10 min | Nothing urgent |
| **P7** | CTM More Seeds | Statistical power for CTM | S1 GPU | ~6h | After baseline |

---

## 4. Batch Script Design

### 4.1 GBDT Parameter Sweep (run on Server 2)

```bash
#!/bin/bash
# save as: run_gbdt_sweep.sh
# Run on Server 2 CPU: bash run_gbdt_sweep.sh

DATA_DIR=/root/stock_data
N_ASSETS=50
OUT_DIR=results/gbdt_sweep_v2
mkdir -p $OUT_DIR

export PYTHONPATH=/root/Glaubenskrieg:/root/Hoffnung/build:/root/Hoffnung/build/python

# Run one combination
run_one() {
    local trees=$1 depth=$2 seed=$3
    local out="${OUT_DIR}/t${trees}_d${depth}_s${seed}.json"
    if [ -f "$out" ]; then
        echo "[SKIP] $out exists"
        return
    fi
    echo "[RUN] trees=$trees depth=$depth seed=$seed"
    python /root/Glaubenskrieg/scripts/train_gbdt_only.py \
        --data-dir "$DATA_DIR" \
        --n-assets "$N_ASSETS" \
        --gbdt-trees "$trees" --gbdt-depth "$depth" \
        --seed "$seed" \
        --output "$out" \
        2>&1 | tail -3
}

# Trees × Depth grid: 9 combinations × 5 seeds
SEEDS=(42 123 456 789 1024)
for trees in 100 200 500; do
    for depth in 4 6 8; do
        for seed in "${SEEDS[@]}"; do
            run_one $trees $depth $seed
        done
    done
done

echo "=== GBDT sweep complete ==="
```

### 4.2 Parallel Execution Strategy

Split into 4 parallel workers to minimize wall clock:

| Worker | Assignments | Est. Runtime |
|--------|------------|-------------|
| W1 | trees=100 × all depths × all seeds | ~15 min |
| W2 | trees=200 × all depths × all seeds | ~27 min |
| W3 | trees=500 × all depths × all seeds | ~64 min |
| W4 | Loss sweep + Step sweep + Extra seeds | ~42 min |

Launch:
```bash
# On Server 2 (from /root/Glaubenskrieg):
bash run_gbdt_sweep.sh --trees 100 &    # background
bash run_gbdt_sweep.sh --trees 200 &    # background
bash run_gbdt_sweep.sh --trees 500 &    # background
bash run_gbdt_loss_step_extra.sh &      # background
wait
```

### 4.3 CTM Scaling Script (run on Server 1 GPU)

```bash
#!/bin/bash
# save as: run_ctm_scaling.sh
# Run on Server 1 (after GPU frees)

DATA_DIR=/root/stock_data
export PYTHONPATH=/root/Glaubenskrieg:/root/Hoffnung/build:/root/Hoffnung/build/python

run_ctm() {
    local n_assets=$1 seed=$2
    echo "[CTM] n_assets=$n_assets seed=$seed"
    CUDA_VISIBLE_DEVICES=0 python /root/Glaubenskrieg/scripts/train.py \
        --config /root/Glaubenskrieg/configs/default.yaml \
        --data-dir "$DATA_DIR" \
        --multi-asset --n-assets "$n_assets" \
        --seed "$seed" \
        --device cuda \
        --output "results/ctm_v2_n${n_assets}_s${seed}.json"
}

# n_assets=20: 3 seeds
for seed in 42 123 456; do
    run_ctm 20 $seed
done

# n_assets=80: 3 seeds (long!)
for seed in 42 123 456; do
    run_ctm 80 $seed
done
```

---

## 5. Expected Scientific Contributions

### 5.1 GBDT Sensitivity Analysis
| Question | Experiment | Expected Finding |
|----------|-----------|-----------------|
| Does more trees help? | T100/T200/T500 at depth=6 | Diminishing returns after 200 trees |
| Does depth cause overfitting? | D4/D6/D8 at trees=200 | sharpe drops at depth=8 |
| Is MSE optimal? | MSE vs MAE vs Huber | MAE more robust to outliers |
| Are conclusions window-dependent? | step=14/21/42 | std_sharpe decreases with more windows ✓ |

### 5.2 CTM Scaling Law
| Question | Experiment | Expected Finding |
|----------|-----------|-----------------|
| Does cross-attention help? | N20 vs N50 vs N80 | N50 > N20, but N80 ≈ N50 (diminishing) |
| Is CTM better than GBDT? | CTM-N50 vs GBDT-T200-D6 | CTM slightly better (+0.05 sharpe?) or worse |

### 5.3 Statistical Rigor
| Metric | Current | After Phase 2 |
|--------|---------|---------------|
| GBDT seeds | 3 | 10+ (baseline), 5 (sweeps) |
| CTM seeds | 3 | 5 (N50), 3 (N20,N80) |
| GBDT configs tested | 1 | 13+ |
| CTM configs tested | 1 | 3 |
| p-value computable? | No (n=3, CI too wide) | Yes (n=10, t-test valid) |

---

## 6. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| GBDT depth=8, trees=500 OOM on CPU | Experiment crashes | Run depth=8 separately, monitor memory |
| CTM n_assets=80 takes >24h | Blocks GPU queue | Run overnight; cancel if >24h |
| Cross-attention kernel crashes on Ascend 910B | Results unusable | Test with n_assets=20 first (fastest) |
| Server 2 CPU not truly idle | GBDT runtime 2-3× estimate | Conservative: run 3 parallel max |
| Parameter interaction missed | Grid too sparse | Focus on 1D sweeps first, then 2D grid |
| Results all noise (sharpe ~0) | No publishable finding | Report null honestly; this IS a valid finding |

---

## 7. Output Naming Convention

```
results/gbdt_sweep_v2/t{trees}_d{depth}_s{seed}.json
results/gbdt_sweep_v2/loss_{loss}_s{seed}.json
results/gbdt_sweep_v2/step_{step}_s{seed}.json
results/gbdt_baseline_v2/s{seed}.json
results/ctm_scaling_v2/n{n_assets}_s{seed}.json
```

---

## 8. Quick-Start Commands

### On Server 2 (CPU) — Copy-paste to start immediately:

```bash
cd /root/Glaubenskrieg
mkdir -p results/gbdt_sweep_v2

# Tier 1.1: Trees×Depth Grid (run 3 parallel workers)
# Worker 1: trees=100
nohup bash -c '
for d in 4 6 8; do for s in 42 123 456 789 1024; do
  python scripts/train_gbdt_only.py --data-dir /root/stock_data --n-assets 50 \
    --gbdt-trees 100 --gbdt-depth $d --seed $s \
    --output results/gbdt_sweep_v2/t100_d${d}_s${s}.json 2>&1 | tail -1
done; done' > /root/logs/gbdt_w1.log 2>&1 &

# Worker 2: trees=200
nohup bash -c '
for d in 4 6 8; do for s in 42 123 456 789 1024; do
  python scripts/train_gbdt_only.py --data-dir /root/stock_data --n-assets 50 \
    --gbdt-trees 200 --gbdt-depth $d --seed $s \
    --output results/gbdt_sweep_v2/t200_d${d}_s${s}.json 2>&1 | tail -1
done; done' > /root/logs/gbdt_w2.log 2>&1 &

# Worker 3: trees=500
nohup bash -c '
for d in 4 6 8; do for s in 42 123 456 789 1024; do
  python scripts/train_gbdt_only.py --data-dir /root/stock_data --n-assets 50 \
    --gbdt-trees 500 --gbdt-depth $d --seed $s \
    --output results/gbdt_sweep_v2/t500_d${d}_s${s}.json 2>&1 | tail -1
done; done' > /root/logs/gbdt_w3.log 2>&1 &

# Tier 1.2-1.4: Loss + Step + Extra Seeds (Worker 4)
nohup bash -c '
# Loss sweep
for loss in mae huber; do for s in 42 123 456 789 1024; do
  python scripts/train_gbdt_only.py --data-dir /root/stock_data --n-assets 50 \
    --gbdt-loss $loss --seed $s \
    --output results/gbdt_sweep_v2/loss_${loss}_s${s}.json 2>&1 | tail -1
done; done
# Step sweep
for step in 14 42; do for s in 42 123 456 789 1024; do
  python scripts/train_gbdt_only.py --data-dir /root/stock_data --n-assets 50 \
    --step-size $step --seed $s \
    --output results/gbdt_sweep_v2/step_${step}_s${s}.json 2>&1 | tail -1
done; done
# Extra 7 seeds for baseline
for s in 789 1024 2048 4096 8192 16384 32768; do
  python scripts/train_gbdt_only.py --data-dir /root/stock_data --n-assets 50 \
    --seed $s \
    --output results/gbdt_baseline_v2/s${s}.json 2>&1 | tail -1
done' > /root/logs/gbdt_w4.log 2>&1 &

echo "All 4 GBDT workers launched. Monitor: tail -f /root/logs/gbdt_w*.log"
```

---

## 9. Analysis Plan (Post-Experiment)

After all experiments complete, produce:

1. **Heatmap**: rows=trees, cols=depth, cell=mean_sharpe ± std across seeds
2. **Learning curve**: sharpe vs (trees × depth) to detect overfitting elbow
3. **Loss comparison**: bar chart with error bars (MSE vs MAE vs Huber)
4. **Window stability**: sharpe mean and std vs step_size / n_windows
5. **CTM scaling plot**: sharpe vs n_assets (20, 50, 80) with trend line
6. **CTM vs GBDT head-to-head**: same seeds, same n_assets, same windows
7. **Statistical table**: mean ± std ± CI for each config across seeds
8. **Window-level analysis**: do all 28 windows agree on best config? (ANOVA)

