#!/bin/bash
# Phase 1: Feature-Engineered LightGBM Baseline
# Runs on BOTH servers with multiple seeds
# Usage: bash scripts/launch_fe_lgb.sh

SEEDS=(42 123 456 789 1024)
DATA_DIR=/root/data/tencent_clean
OUT_DIR=/root/results/fe_lgb_baseline
mkdir -p $OUT_DIR

cd /root/Glaubenskrieg
export PYTHONPATH=/root/Glaubenskrieg:/root/Hoffnung/build:/root/Hoffnung/build/python:$PYTHONPATH

echo "=== Feature-Engineered LightGBM Baseline ==="
echo "Seeds: ${SEEDS[@]}"
echo "Output: $OUT_DIR"
echo ""

for seed in "${SEEDS[@]}"; do
    out="$OUT_DIR/fe_lgb_s${seed}.json"
    if [ -f "$out" ]; then
        echo "[SKIP] $out exists"
        continue
    fi
    echo "[RUN] seed=$seed"
    python3 scripts/train_fe_lgb.py \
        --data-dir "$DATA_DIR" \
        --n-assets 50 \
        --seed "$seed" \
        --output "$out" \
        2>&1 | tail -15
    echo ""
done

echo "=== Done ==="
echo "Results in $OUT_DIR/"
ls -la $OUT_DIR/
