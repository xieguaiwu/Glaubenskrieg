#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# TGPE v4 — Server 2 (223.109.239.32:20248)
# Runs seeds 123 then 456 for both TGPE-A (Ensemble+TimeGate)
# and TGPE-C (CTM+Curriculum) using GPU 0 and GPU 1.
# ═══════════════════════════════════════════════════════════════
set -e

PYTHON=/home/vipuser/miniconda3/bin/python3
export PYTHONPATH=/root/Glaubenskrieg:/root/Hoffnung/build:/root/Hoffnung/build/python:$PYTHONPATH
cd /root/Glaubenskrieg
mkdir -p /root/results/tgpe_v4 /root/logs /root/checkpoints
DATA_DIR=/root/data/tencent_clean

echo "═══════════════════════════════════════════════════════════"
echo "  TGPE v4 — Phase 1: seeds 123  $(date)"
echo "═══════════════════════════════════════════════════════════"

# ── GPU 0: Ensemble + TimeGate (TGPE-A)  seed 123 ──
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train.py \
  --config configs/default.yaml --data-dir "$DATA_DIR" \
  --multi-asset --n-assets 50 --device cuda \
  --seed 123 --n-epochs 50 --batch-size 16 \
  --train-window 400 --val-window 80 --purge-period 63 \
  --ensemble --gbdt-trees 200 --gbdt-depth 6 --gbdt-loss mse \
  --time-gate \
  --output /root/results/tgpe_v4/ens_gate_s123.json \
  --save-dir /root/checkpoints/tgpe_v4_ens_gate_s123 \
  > /root/logs/tgpe_v4_gate_s123.log 2>&1 &
PID1=$!
echo "  [GPU0] Ensemble+TimeGate seed 123  PID=$PID1"

# ── GPU 1: CTM + Curriculum (TGPE-C)  seed 123 ──
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/train.py \
  --config configs/default.yaml --data-dir "$DATA_DIR" \
  --multi-asset --n-assets 50 --device cuda \
  --seed 123 --n-epochs 50 --batch-size 16 \
  --train-window 400 --val-window 80 --purge-period 63 \
  --curriculum \
  --output /root/results/tgpe_v4/ctm_curriculum_s123.json \
  --save-dir /root/checkpoints/tgpe_v4_ctm_curriculum_s123 \
  > /root/logs/tgpe_v4_curriculum_s123.log 2>&1 &
PID2=$!
echo "  [GPU1] CTM+Curriculum seed 123  PID=$PID2"

echo "  Waiting for Phase 1 (seeds 123) to complete..."
wait $PID1 $PID2
echo "  ✅ Phase 1 complete at $(date)"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  TGPE v4 — Phase 2: seeds 456  $(date)"
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
PID3=$!
echo "  [GPU0] Ensemble+TimeGate seed 456  PID=$PID3"

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
PID4=$!
echo "  [GPU1] CTM+Curriculum seed 456  PID=$PID4"

echo "  Waiting for Phase 2 (seeds 456) to complete..."
wait $PID3 $PID4
echo "  ✅ Phase 2 complete at $(date)"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ALL TGPE v4 EXPERIMENTS DONE  $(date)"
echo "═══════════════════════════════════════════════════════════"
echo "  Outputs:"
echo "    /root/results/tgpe_v4/ens_gate_s123.json"
echo "    /root/results/tgpe_v4/ctm_curriculum_s123.json"
echo "    /root/results/tgpe_v4/ens_gate_s456.json"
echo "    /root/results/tgpe_v4/ctm_curriculum_s456.json"
echo "  Logs:"
echo "    /root/logs/tgpe_v4_gate_s123.log"
echo "    /root/logs/tgpe_v4_curriculum_s123.log"
echo "    /root/logs/tgpe_v4_gate_s456.log"
echo "    /root/logs/tgpe_v4_curriculum_s456.log"
echo "═══════════════════════════════════════════════════════════"
