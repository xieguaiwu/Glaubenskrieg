#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Glaubenskrieg v5 — Launch scripts for BOTH servers
# 
# Experiment matrix: 3 variants × 5 seeds = 15 experiments per server
# Variant A: GBDT baseline (CPU)
# Variant B: CTM baseline (GPU)  
# Variant C: Ensemble + TimeGate + Finetune (GPU)
#
# GPU 0 → Ensemble+Gate, GPU 1 → CTM (parallel per seed)
# ═══════════════════════════════════════════════════════════════════
set -e

export PYTHONPATH=/root/Glaubenskrieg:/root/Hoffnung/build:/root/Hoffnung/build/python:$PYTHONPATH
cd /root/Glaubenskrieg
mkdir -p /root/results/v5 /root/logs /root/checkpoints
DATA_DIR=/root/data/tencent_clean

SEEDS=(42 123 456 789 1024)

echo "═══════════════════════════════════════════════════════════"
echo "  Glaubenskrieg v5 — $(hostname)  $(date)"
echo "  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Seeds: ${SEEDS[*]}"
echo "═══════════════════════════════════════════════════════════"

# ── Phase 1: GBDT baseline (CPU, parallel to GPU work) ──
echo ""
echo "── GBDT Baseline (CPU background) ──"
for s in "${SEEDS[@]}"; do
    python3 scripts/train_gbdt_only.py \
        --data-dir "$DATA_DIR" --n-assets 50 \
        --seed "$s" --train-window 400 --val-window 80 --purge-period 126 \
        --output "/root/results/v5/gbdt_s${s}.json" \
        > "/root/logs/v5_gbdt_s${s}.log" 2>&1 &
done
echo "  GBDT running in background (5 jobs)"

# ── Phase 2: CTM + Ensemble/TimeGate (parallel on both GPUs) ──
for s in "${SEEDS[@]}"; do
    echo ""
    echo "── Seed $s ──"
    
    # GPU 0: Ensemble + TimeGate + Finetune
    CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py \
        --config configs/default.yaml --data-dir "$DATA_DIR" \
        --multi-asset --n-assets 50 --device cuda \
        --seed "$s" --n-epochs 50 --batch-size 16 \
        --train-window 400 --val-window 80 --purge-period 126 \
        --ensemble --gbdt-trees 200 --gbdt-depth 6 --gbdt-loss mse \
        --time-gate --finetune-gate \
        --output "/root/results/v5/ens_gate_s${s}.json" \
        --save-dir "/root/checkpoints/v5_ens_gate_s${s}" \
        > "/root/logs/v5_ens_gate_s${s}.log" 2>&1 &
    PID_GATE=$!
    
    # GPU 1: CTM baseline (no curriculum, no gate — clean baseline)
    CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py \
        --config configs/default.yaml --data-dir "$DATA_DIR" \
        --multi-asset --n-assets 50 --device cuda \
        --seed "$s" --n-epochs 50 --batch-size 16 \
        --train-window 400 --val-window 80 --purge-period 126 \
        --output "/root/results/v5/ctm_s${s}.json" \
        --save-dir "/root/checkpoints/v5_ctm_s${s}" \
        > "/root/logs/v5_ctm_s${s}.log" 2>&1 &
    PID_CTM=$!
    
    echo "  GPU0: Ensemble+Gate PID=$PID_GATE  GPU1: CTM PID=$PID_CTM"
    wait $PID_GATE $PID_CTM
    echo "  ✅ Seed $s done ($(date +%H:%M:%S))"
done

# Wait for GBDT to finish
wait
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ALL v5 EXPERIMENTS DONE  $(date)"
echo "═══════════════════════════════════════════════════════════"
ls -la /root/results/v5/*.json 2>/dev/null
echo "  Total files: $(ls /root/results/v5/*.json 2>/dev/null | wc -l)"
