#!/bin/bash
export PATH=/home/vipuser/miniconda3/bin:$PATH
export PYTHONPATH=/root/Glaubenskrieg:/root/Hoffnung/build:/root/Hoffnung/build/python
cd /root/Glaubenskrieg

echo "[$(date)] Auto-waiting for ensemble seeds to finish..."
while ps aux | grep -q 'train\.py.*--ensemble'; do
  sleep 120
done
echo "[$(date)] Done. Waiting 30s for GPU release..."
sleep 30

FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep ', 0 MiB' | head -1 | cut -d, -f1)
FREE_GPU=${FREE_GPU:-1}
echo "[$(date)] Starting seed 456 on GPU $FREE_GPU..."

CUDA_VISIBLE_DEVICES=$FREE_GPU nohup python3 scripts/train.py \
  --config configs/default.yaml \
  --data-dir /root/data/tencent_clean/ \
  --multi-asset --n-assets 200 --device cuda \
  --train-window 400 --val-window 80 --step-size 21 --purge-period 63 \
  --n-epochs 50 --batch-size 8 \
  --ensemble --gbdt-trees 200 --gbdt-depth 6 --gbdt-loss mse \
  --seed 456 \
  --output /root/results/ensemble_seed456.json \
  --save-dir /root/checkpoints/ensemble_seed456 \
  > /root/logs/ensemble_seed456.log 2>&1 &
echo "Ensemble seed456: $!"
