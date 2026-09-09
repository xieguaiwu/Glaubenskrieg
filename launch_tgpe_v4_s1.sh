#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# TGPE v4 — Server 1 (223.109.239.36:24520)
# Runs TGPE-A (Ensemble+TimeGate) and TGPE-C (CTM+Curriculum)
# for seed 456 on GPU 0 and GPU 1.
#
# Parallel execution plan:
#   Terminal 1 → Server 1:  bash /root/launch_tgpe_v4_s1.sh
#   Terminal 2 → Server 2:  bash /root/launch_tgpe_v4_seeds.sh
#
# Server 1 handles seed 456 while Server 2 handles seed 123.
# After both servers finish, all 4 experiments are done.
# ═══════════════════════════════════════════════════════════════
set -e

PYTHON=/home/vipuser/miniconda3/bin/python3
export PYTHONPATH=/root/Glaubenskrieg:/root/Hoffnung/build:/root/Hoffnung/build/python:$PYTHONPATH
cd /root/Glaubenskrieg
mkdir -p /root/results/tgpe_v4 /root/logs /root/checkpoints
DATA_DIR=/root/data/tencent_clean

echo "═══════════════════════════════════════════════════════════"
echo "  TGPE v4 — Server 1: seeds 456  $(date)"
echo "═══════════════════════════════════════════════════════════"

# ── GPU 0: Ensemble + TimeGate (TGPE-A)  seed 456 ──
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train.py \
  --config configs/default.yaml --data-dir "$DATA_DIR" \
  --multi-asset --n-assets 50 --device cuda \
  --seed 456 --n-epochs 50 --batch-size 16 \
  --train-window 400 --val-window 80 --purge-period 63 \
  --ensemble --gbdt-trees 200 --gbdt-depth 6 --gbdt-loss mse \
  --time-gate \
  --output /root/results/tgpe_v4/ens_gate_s456.json \
  --save-dir /root/checkpoints/tgpe_v4_ens_gate_s456 \
  > /root/logs/tgpe_v4_gate_s456.log 2>&1 &
PID1=$!
echo "  [GPU0] Ensemble+TimeGate seed 456  PID=$PID1"

# ── GPU 1: CTM + Curriculum (TGPE-C)  seed 456 ──
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/train.py \
  --config configs/default.yaml --data-dir "$DATA_DIR" \
  --multi-asset --n-assets 50 --device cuda \
  --seed 456 --n-epochs 50 --batch-size 16 \
  --train-window 400 --val-window 80 --purge-period 63 \
  --curriculum \
  --output /root/results/tgpe_v4/ctm_curriculum_s456.json \
  --save-dir /root/checkpoints/tgpe_v4_ctm_curriculum_s456 \
  > /root/logs/tgpe_v4_curriculum_s456.log 2>&1 &
PID2=$!
echo "  [GPU1] CTM+Curriculum seed 456  PID=$PID2"

echo "  Waiting for seed 456 variants to complete..."
wait $PID1 $PID2
echo "  ✅ Server 1 seed 456 done at $(date)"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  SERVER 1 TGPE v4 DONE  $(date)"
echo "═══════════════════════════════════════════════════════════"
echo "  Outputs:"
echo "    /root/results/tgpe_v4/ens_gate_s456.json"
echo "    /root/results/tgpe_v4/ctm_curriculum_s456.json"
echo "  Logs:"
echo "    /root/logs/tgpe_v4_gate_s456.log"
echo "    /root/logs/tgpe_v4_curriculum_s456.log"
echo "═══════════════════════════════════════════════════════════"
