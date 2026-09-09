# Glaubenskrieg v5 实验文档

> **最后更新**: 2026-06-06 12:00 CST
> **状态**: v5 实验中 (3变体 × 5种子 = 15实验/服务器)

---

## 服务器连接

```
# Server 1 (V100 × 2, 32GB each)
ssh root@223.109.239.36 -p 24212
密码: <REDACTED>

# Server 2 (RTX 3090 × 2, 24GB each)
ssh root@223.109.239.36 -p 33748
密码: <REDACTED>
```

两台服务器在同一物理机 (ubuntu22, Ubuntu 22.04, CUDA 12.2)，通过不同端口隔离。
原生 NVIDIA GPU → Mamba SSM 全速运行 (21-35ms/forward, 旧 Ascend 910B 为 500-1000ms)。

---

## 实验设计

### 变体 (3组)

| # | 变体 | CLI | 说明 |
|---|------|-----|------|
| A | **GBDT baseline** | `train_gbdt_only.py` | 纯 GBDT，C++ Hoffnung 后端 |
| B | **CTM baseline** | `train.py --multi-asset --n-assets 50` | 纯 CTM，无 curriculum/gate |
| C | **Ensemble+TimeGate** | `train.py --ensemble --time-gate --finetune-gate` | CTM+GBDT 融合 + Gate 微调 (50 epoch, lr=5e-4) |

### 种子: 42, 123, 456, 789, 1024

### 统一配置

| 参数 | 值 | 说明 |
|------|-----|------|
| step_size | 63 | v5 修复: 3 窗口, 90% 全窗重叠 (v4 的 21→96% 重叠已废弃) |
| purge_period | 126 | 2× step_size |
| n_epochs | 50 | CTM 每窗口训练轮数 |
| batch_size | 16 | |
| model_dim | 64 | |
| state_dim | 8 | |
| n_layers | 2 | |
| train_window | 400 | |
| val_window | 80 | |

---

## v4 → v5 修复清单

| # | 严重度 | v4 问题 | v5 修复 |
|---|:------:|---------|---------|
| P0-1 | 关键 | Curriculum dropout 在 test-prep 中 80% 归零 → 模型坍缩 | 完全移除 test-prep 中的 curriculum dropout |
| P0-2 | 关键 | test-prep 模型随机初始化 → 与 walk-forward 无关 | warm-start 从最佳 walk-forward 窗口 |
| P1-3 | 高 | step_size=21 → 96% 窗口重叠，3× 计算量无益 | 恢复 step_size=63 |
| P1-4 | 高 | Gate 微调 10 epoch, lr=1e-4 → 参数移动 <0.6% | epochs→50, lr→5e-4 |
| P2-5 | 中 | GBDT 坍缩时仍执行 gate 微调 → 污染集成 | GBDT 质检通过后才微调 |
| Bug 2 | 关键 | validate() 对 demeaned targets 取 cross-sectional mean → 指标坍缩 | 逐 asset 展开计算指标 |

---

## v4 根因分析摘要

TGPE v4 在所有种子/变体上均劣于 v3 baseline:
- v3 CTM IC=0.142 → v4 Curriculum IC=0.065
- v3 Ensemble IC=0.049 → v4 TimeGate IC=0.028

**五大根因**:
1. Curriculum dropout 在 test-prep 训练中污染模型 (P0)
2. test-prep 模型独立于 walk-forward (P0)
3. TimeGate 未学习 (α/β/γ 移动 <0.6%)
4. 窗口数虚增 (96% 重叠)
5. GBDT 频繁坍缩污染集成

---

## 运行命令

```bash
# 每台服务器上:
cd /root/Glaubenskrieg
export PYTHONPATH=/root/Glaubenskrieg:/root/Hoffnung/build:/root/Hoffnung/build/python:$PYTHONPATH
bash scripts/launch_v5.sh
```

---

## 预期实验

1. 单独进行GBDT实验
2. 单独进行CTM实验
3. 进行CTM+GBDT融合实验 (TimeGate + Finetune)
