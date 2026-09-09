#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Glaubenskrieg v5 — Server 1 (V100 × 2)
# 全修复版本: step_size=63, warm-start test-prep, gate epochs=50, lr=5e-4
# ═══════════════════════════════════════════════════════════════════
set -e

export PYTHONPATH=/root/Glaubenskrieg:/root/Hoffnung/build:/root/Hoffnung/build/python:$PYTHONPATH
cd /root/Glaubenskrieg
mkdir -p /root/results/v5 /root/logs /root/checkpoints
DATA_DIR=/root/data/tencent_clean

SEEDS=(42 123 456 789 1024)
N_SEEDS=${#SEEDS[@]}

echo "═══════════════════════════════════════════════════════════"
echo "  Glaubenskrieg v5 — Server 1 (V100)  $(date)"
echo "  Seeds: ${SEEDS[*]}"
echo "═══════════════════════════════════════════════════════════"

# ── Phase 1: GBDT baseline (CPU, fast) ──
echo ""
echo "── Phase 1: GBDT Baseline ──"
for s in "${SEEDS[@]}"; do
    echo "  GBDT seed $s..."
    python3 scripts/train_gbdt_only.py \
        --data-dir "$DATA_DIR" --n-assets 50 \
        --seed "$s" --train-window 400 --val-window 80 --purge-period 126 \
        --output "/root/results/v5/gbdt_s${s}.json" \
        > "/root/logs/v5_gbdt_s${s}.log" 2>&1
done
echo "  ✅ GBDT done"

# ── Phase 2: CTM baseline (GPU 0+1 alternating) ──
echo ""
echo "── Phase 2: CTM Baseline (no curriculum, no gate) ──"
for i in $(seq 0 $((N_SEEDS - 1))); do
    s="${SEEDS[$i]}"
    gpu=$((i % 2))
    echo "  CTM seed $s on GPU$gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/train.py \
        --config configs/default.yaml --data-dir "$DATA_DIR" \
        --multi-asset --n-assets 50 --device cuda \
        --seed "$s" --n-epochs 50 --batch-size 16 \
        --train-window 400 --val-window 80 --purge-period 126 \
        --output "/root/results/v5/ctm_s${s}.json" \
        --save-dir "/root/checkpoints/v5_ctm_s${s}" \
        > "/root/logs/v5_ctm_s${s}.log" 2>&1
done
echo "  ✅ CTM done"

# ── Phase 3: Ensemble + TimeGate (GPU 0+1 alternating, with finetuning) ──
echo ""
echo "── Phase 3: Ensemble + TimeGate + Finetune ──"
for i in $(seq 0 $((N_SEEDS - 1))); do
    s="${SEEDS[$i]}"
    gpu=$((i % 2))
    echo "  Ensemble+Gate seed $s on GPU$gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/train.py \
        --config configs/default.yaml --data-dir "$DATA_DIR" \
        --multi-asset --n-assets 50 --device cuda \
        --seed "$s" --n-epochs 50 --batch-size 16 \
        --train-window 400 --val-window 80 --purge-period 126 \
        --ensemble --gbdt-trees 200 --gbdt-depth 6 --gbdt-loss mse \
        --time-gate --finetune-gate \
        --output "/root/results/v5/ens_gate_s${s}.json" \
        --save-dir "/root/checkpoints/v5_ens_gate_s${s}" \
        > "/root/logs/v5_ens_gate_s${s}.log" 2>&1
done
echo "  ✅ Ensemble+Gate done"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  SERVER 1 v5 ALL DONE  $(date)"
echo "═══════════════════════════════════════════════════════════"
echo "  Results: /root/results/v5/"
ls -la /root/results/v5/*.json 2>/dev/null | wc -l
echo "  files produced"
