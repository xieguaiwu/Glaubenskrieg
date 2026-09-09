# CTM + GBDT Ensemble Deployment Guide

> 撰写日期：2026-06-02 | 覆盖范围：CTM (Mamba SSM) + GBDT (Hoffnung C++ Tree) 联合训练与部署
> 硬件基线：CPU 可跑推理，GPU (RTX 3090/4090) 推荐用于训练

---

## 目录

1. [系统架构概述](#1-系统架构概述)
2. [前置要求](#2-前置要求)
3. [编译 Hoffnung (GBDT C++ 核心)](#3-编译-hoffnung-gbdt-c-核心)
4. [环境配置](#4-环境配置)
5. [数据管道](#5-数据管道)
6. [训练工作流](#6-训练工作流)
7. [模型序列化与导出](#7-模型序列化与导出)
8. [推理部署](#8-推理部署)
9. [国内云 GPU 部署](#9-国内云-gpu-部署)
10. [验证与冒烟测试](#10-验证与冒烟测试)
11. [故障排除](#11-故障排除)
12. [参考资料](#12-参考资料)

---

## 1. 系统架构概述

Glaubenskrieg 采用 **CTM (Conv-Temporal-Mamba) + GBDT (Gradient Boosted Decision Tree)** 双模型架构。两个模型独立训练后通过 IC 加权融合输出最终信号。

```
OHLCV 数据 (CSV)
    │
    ▼
特征工程 (features.py, 9 个技术指标)
    │
    ├──────────────────────┬──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
 CTM (PyTorch SSM)     GBDT (C++ Tree)     [可选: P3 联合训练]
    │                      │                      │
    │   CTM 输出           │   GBDT 输出          │  融合输出
    ▼                      ▼                      ▼
 ┌─────────────────────────────────────────────────────┐
 │              IC 加权信号融合 (fused_signal)           │
 │  fused = w_ctm × ctm_pred + (1-w_ctm) × gbdt_pred   │
 └─────────────────────────────────────────────────────┘
    │
    ▼
 排序 → 选股信号 → [交易执行 → retail_deployment_guide.md]
```

### 集成级别

| 级别 | 名称 | 训练方式 | 启用方式 |
|------|------|---------|---------|
| **P0** | Ensemble 融合 | CTM + GBDT 独立训练，IC 加权合成 | `--ensemble` |
| **P1** | 特征融合 | CTM 隐藏态 → GBDT 特征输入 | 自动（P0 内置） |
| **P2** | 损失桥接 | GBDT 优化 RankIC/复合损失 | `--gbdt-loss rankic` |
| **P3** | 三阶段联合训练 | CTM 预热 → GBDT 拟合 → 调制器微调 | `--p3` |

> 详见 `docs/fusion_roadmap.md`。

---

## 2. 前置要求

### 软件

| 组件 | 最低版本 | 说明 |
|------|---------|------|
| Python | ≥ 3.10 | 推荐 3.12 |
| PyTorch | ≥ 2.0 | 训练用 GPU 版，推理可用 CPU 版 |
| CMake | ≥ 3.18 | 编译 Hoffnung C++ 核心 |
| C++ 编译器 | C++17 | GCC ≥ 9 / Clang ≥ 10 |
| OpenMP | 可选 | 加速 GBDT 直方图构建 |

### Python 依赖

```bash
# 核心依赖（Glaubenskrieg）
pip install torch numpy pandas pyyaml scipy

# 可选：Alpaca 交易执行
pip install alpaca-py>=0.13.0
```

### 仓库

两个项目需要克隆到本地：

```bash
# Glaubenskrieg - CTM 模型
/path/to/Glaubenskrieg/

# Hoffnung - GBDT C++ 核心
/path/to/Hoffnung/
```

### 硬件

| 场景 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CTM 训练 (small/large) | 任意 GPU 4GB+ 或 CPU | RTX 3090 24GB |
| CTM 训练 (loop_large) | RTX 3090 24GB | RTX 4090 / A100 |
| GBDT 训练 | CPU 即可（树模型） | CPU 多核（OpenMP） |
| 每日推理 (500 stocks) | CPU 4 核 | CPU 8 核 (Hetzner AX102) |

---

## 3. 编译 Hoffnung (GBDT C++ 核心)

GBDT 的树构建和推理由 C++ 实现（通过 pybind11 暴露给 Python）。

### 3.1 编译步骤

```bash
cd /path/to/Hoffnung

# 创建构建目录
mkdir -p build && cd build

# 配置 CMake（Release 模式优化性能）
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译（利用多核加速）
make -j$(nproc)
```

### 3.2 编译产物

```
build/
├── gbdt_python.cpython-*-x86_64-linux-gnu.so   # Python 绑定模块（~500KB）
├── libgbdt_core.so                               # C++ 共享库（~250KB）
├── python/gbdt/                                  # Python 包副本
│   ├── __init__.py                               # GBDTTrainer 桥接类
│   └── losses.py                                 # 量化损失函数
└── test_tree                                     # 单元测试
```

### 3.3 验证编译

```bash
cd /path/to/Hoffnung/build
python3 -c "
import sys
sys.path.insert(0, 'python')
from gbdt import GBDT, GBDTConfig, GBDTTrainer
from gbdt.losses import mse_loss, rankic_loss, huber_loss
print('✅ C++ GBDT 模块加载成功')
print(f'   可用损失: mse, huber, rankic, quantile, sharpe, composite')
"
```

### 3.4 注意事项

- **LibTorch 依赖**：CMake 需要 `find_package(Torch)`，系统需安装 PyTorch 的 C++ 版本（`pip install torch` 即可提供）
- **GBDT 本身运行在 CPU**：树模型推理不需要 GPU，但编译时需要 LibTorch
- **Python 版本匹配**：编译时的 Python 版本必须与运行时的 Python 版本一致（均为 3.12+）

---

## 4. 环境配置

### 4.1 方案 A：pip 安装（推荐）

```bash
# Glaubenskrieg
cd /path/to/Glaubenskrieg
pip install -e .

# Hoffnung
cd /path/to/Hoffnung
pip install -e .
```

### 4.2 方案 B：PYTHONPATH 配置

```bash
export PYTHONPATH="/path/to/Glaubenskrieg:/path/to/Hoffnung/build/python:${PYTHONPATH}"
```

### 4.3 方案 C：动态路径设置

Glaubenskrieg 的 `scripts/train.py` 会自动在 `sys.path` 中添加 Hoffnung 编译路径。如果放在非标准位置，设置环境变量：

```bash
export GBDT_BUILD_DIR=/path/to/Hoffnung/build
```

### 4.4 验证集成

```bash
python3 -c "
# 验证 CTM 导入
from src.model.ctm_model import CTMStockModel
print('✅ CTM 模型导入成功')

# 验证 GBDT 集成
try:
    from gbdt.losses import rankic_loss
    print('✅ GBDT loss 函数导入成功')
    print('   完整集成可用')
except ImportError:
    print('⚠️ GBDT 不可用，CTM-only 模式仍可运行')
"
```

---

## 5. 数据管道

### 5.1 输入格式

OHLCV CSV 文件，至少包含以下列：

```csv
date,open,high,low,close,volume
2020-01-02,100.0,101.5,99.8,100.5,1000000
2020-01-03,100.5,102.0,100.2,101.8,1200000
...
```

### 5.2 特征工程

`src/data/features.py` 从 OHLCV 计算 9 个技术指标：

| 特征 | 说明 | 窗口 |
|------|------|------|
| simple_return | 简单收益率 | 1 日 |
| log_return | 对数收益率 | 1 日 |
| sma_5 | 5 日均线 | 5 日 |
| sma_20 | 20 日均线 | 20 日 |
| rsi_14 | 相对强弱指标 | 14 日 |
| bollinger_position | 布林带位置 | 20 日 |
| volume_ratio | 成交量比率 | 20 日 |
| realized_vol_21 | 已实现波动率 | 21 日 |
| bias | 乖离率 | 5 日 |

### 5.3 序列构建

- **序列长度**：63 个交易日（默认，约 3 个月）
- **归一化**：滚动 z-score，252 日回溯窗口
- **目标值**：`forward_return = close[t+1]/close[t] - 1`

### 5.4 训练/验证分割

采用 Walk-Forward 验证以保证时序完整性：

```
Train[t0, t1] → Purge[t1, t1+126] → Val[t1+126, t2] → ...
```

**purge_period=126**：清除训练窗口最后 126 日，防止数据泄露。

### 5.5 数据来源

| 数据源 | 覆盖市场 | 费用 | 备注 |
|--------|---------|------|------|
| yfinance | 全球股票/ETF | 免费 | Yahoo Finance，适合回测 |
| Alpaca Data API v2 | 美股 | $99/月 | 实时数据 |
| J-Quants API | 日股 | 免费 / ¥4,950/月 | JPX 官方 |
| 自定义 CSV | 任何 | — | 自行准备 |

---

## 6. 训练工作流

### 6.1 CTM-Only（基线）

最简单的训练方式，不依赖 GBDT：

```bash
cd /path/to/Glaubenskrieg

# 默认配置（small, CPU）
PYTHONPATH=. python scripts/train.py --config configs/default.yaml

# GPU 训练（large 配置）
PYTHONPATH=. python scripts/train.py \
  --config configs/scale_large.yaml \
  --device cuda

# RecurrentCTM（Loop Mamba）
PYTHONPATH=. python scripts/train.py \
  --config configs/scale_loop.yaml
```

### 6.2 CTM + GBDT Ensemble (P0)

使用 `--ensemble` 标志启用 GBDT 集成：

```bash
# 基本用法
PYTHONPATH=. python scripts/train.py \
  --config configs/default.yaml \
  --ensemble

# 自定义 GBDT 超参数
PYTHONPATH=. python scripts/train.py \
  --config configs/default.yaml \
  --ensemble \
  --gbdt-trees 200 \       # 树的数量（默认 100）
  --gbdt-depth 6 \         # 树最大深度（默认 6）
  --gbdt-lr 0.05 \         # 学习率（默认 0.1）
  --gbdt-loss mse \        # 损失函数（默认 mse）
  --gbdt-subsample 0.8 \   # 行采样比例
  --gbdt-colsample 0.8     # 列采样比例
```

训练完成后自动输出每个窗口的 CTM Sharpe、GBDT IC 和 Ensemble Sharpe。

### 6.3 RankIC 损失 (P2)

使用可微 RankIC 损失优化 GBDT，直接最大化排序相关性：

```bash
PYTHONPATH=. python scripts/train.py \
  --config configs/default.yaml \
  --ensemble \
  --gbdt-loss rankic \      # 可微 RankIC 损失
  --gbdt-trees 100 \
  --gbdt-depth 4
```

> **注意**：`rankic_loss` 和 `composite` 损失依赖 `GBDTTrainer`（Python 桥接类），需要先完成 Hoffnung 编译。编译不可用时自动降级为 MSE 损失并进行 manual boosting loop。

### 6.4 三阶段联合训练 (P3)

P3 模式将 CTM 和 GBDT 通过可微 Cross-Attention 调制器联合微调：

```bash
# 基本 P3 训练
PYTHONPATH=. python scripts/train.py \
  --config configs/default.yaml \
  --multi-asset --n-assets 10 \
  --ensemble --p3 \
  --modulator-epochs 10 \
  --modulator-lr 1e-4 \
  --gbdt-loss rankic \
  --gbdt-trees 100 --gbdt-depth 4
```

**三阶段流程**：
1. **CTM 预热**：正常训练 CTM 模型（无 GBDT）
2. **GBDT 拟合**：使用 CTM 提取的特征训练 GBDT
3. **调制器微调**：冻结 CTM + GBDT，微调 Cross-Attention 调制器

### 6.5 多资产 + Ensemble

对多个股票同时训练，利用 Cross-Asset Attention：

```bash
PYTHONPATH=. python scripts/train.py \
  --config configs/scale_portfolio.yaml \
  --ensemble \
  --gbdt-loss rankic
```

### 6.6 加载预训练权重的融合训练

支持从已保存的 CTM 和 GBDT 检查点加载权重，跳过训练阶段直接进行融合推理：

```bash
# Ensemble 推理模式：加载已训练好的 CTM + GBDT → 直接融合
PYTHONPATH=. python scripts/train.py \
  --config configs/default.yaml \
  --multi-asset --n-assets 200 \
  --ensemble-inference \
  --init-ctm-ckpt models/ctm_model/ctm_model.pt \
  --init-gbdt-json models/gbdt_model/gbdt_model.json \
  --output results/ensemble_inference.json

# Ensemble 训练 + 预热：从已有 CTM 检查点开始继续训练
PYTHONPATH=. python scripts/train.py \
  --config configs/default.yaml \
  --multi-asset --n-assets 200 \
  --ensemble \
  --init-ctm-ckpt models/ctm_model/ctm_model.pt \
  --gbdt-trees 200 --gbdt-depth 6 \
  --save-dir models/ensemble_warmstart/
```

参数说明：
| CLI 参数 | 说明 |
|----------|------|
| `--init-ctm-ckpt` | 预训练 CTM 检查点路径（.pt 文件） |
| `--init-gbdt-json` | 预训练 GBDT JSON 模型路径 |
| `--skip-ctm-training` | 跳过 CTM 训练阶段，直接使用预训练权重 |
| `--skip-gbdt-training` | 跳过 GBDT 训练阶段，直接使用预训练权重 |
| `--ensemble-inference` | 纯推理模式：加载两个模型 → 融合 → 输出指标 |

> 详见 `docs/pretrained_ensemble_guide.md`。

### 6.7 保存训练结果

```bash
PYTHONPATH=. python scripts/train.py \
  --config configs/default.yaml \
  --ensemble \
  --save-dir models/experiment_1
```

### 6.7 配置参考

| 配置 | 模型 | 参数量 | GPU VRAM | 推荐场景 |
|------|------|--------|---------|---------|
| `default.yaml` | CTM (d=64,L=3) | ~92K | ~500MB | CPU/入门 |
| `scale_large.yaml` | CTM (d=128,L=6) | ~335K | ~1.1GB | 任意 GPU |
| `scale_loop.yaml` | RecurrentCTM (loop=5) | ~360K | ~1.6GB | 中等规模 |
| `scale_loop_large.yaml` | RecurrentCTM (loop=8) | ~2.8M | ~6GB | **需要 RTX 4090/A100** |
| `scale_portfolio.yaml` | MultiAsset (100 stocks) | ~100K+ | ~2-4GB | 多资产组合 |

---

## 7. 模型序列化与导出

### 7.1 保存格式

使用 `--save-dir` 后，输出目录结构：

```
models/experiment_1/
├── ensemble_ctm.pt              # CTM 权重 (torch.save)
├── ensemble_ctm_config.yaml     # CTM 模型配置
├── ensemble_gbdt.json           # GBDT 树结构 (纯文本 JSON)
├── metrics.json                 # 训练指标（IC, Sharpe 等）
├── config.yaml                  # 完整训练配置
└── predictions/                 # （可选）推理结果
    └── window_0.csv
```

- **CTM 权重** `*.pt`：标准 PyTorch `state_dict`，跨平台兼容
- **GBDT JSON** `*.json`：纯文本树结构，C++/Python 均可加载
- **配置** `*.yaml`：模型超参数，推理时恢复架构必需

### 7.2 加载已训练模型

推理脚本自动加载两种格式：

```bash
# Ensemble 推理
python scripts/infer.py \
  --ctm-ckpt models/experiment_1/ensemble_ctm.pt \
  --gbdt-json models/experiment_1/ensemble_gbdt.json \
  --config configs/default.yaml \
  --data data/new_data.csv \
  --output results/predictions.csv

# CTM-only 推理（无 GBDT）
python scripts/infer.py \
  --ctm-ckpt models/experiment_1/ensemble_ctm.pt \
  --config configs/default.yaml \
  --data data/new_data.csv \
  --output results/predictions.csv
```

### 7.3 跨平台迁移

- `.pt` 文件在 CPU/GPU 间通用（需调用 `model.load_state_dict(torch.load(..., map_location=...))`）
- `.json` 文件完全可移植，不依赖特定硬件或 PyTorch 版本
- `--config` 必须与训练时一致（`model_dim`、`n_layers` 等改变后权重不兼容）

---

## 8. 推理部署

### 8.1 CPU 推理（推荐每日批量）

CTM 在 CPU 上推理约 685ms/stock（d=128, loop=5），每日 500 只股票约 6 分钟。

```bash
# 每日收盘后批量推理
PYTHONPATH=. python scripts/infer.py \
  --ctm-ckpt models/production/ctm_best.pt \
  --gbdt-json models/production/gbdt_best.json \
  --config configs/default.yaml \
  --data data/daily/$(date +%Y-%m-%d).csv \
  --output results/signals/$(date +%Y-%m-%d).csv
```

### 8.2 GPU 推理（低延迟）

需要实时信号时使用 GPU：

```bash
PYTHONPATH=. python scripts/infer.py \
  --ctm-ckpt models/production/ctm_best.pt \
  --gbdt-json models/production/gbdt_best.json \
  --config configs/default.yaml \
  --data data/latest.csv \
  --output results/signals/today.csv \
  --device cuda \
  --batch-size 256
```

**推理延迟参考**（500 只股票）：

| 配置 | CPU (685ms/stock) | GPU (3ms/stock) |
|------|-------------------|-----------------|
| 100 只 | ~69 秒 | ~0.3 秒 |
| 500 只 | ~5.7 分钟 | ~1.5 秒 |
| 2000 只 | ~23 分钟 | ~6 秒 |

### 8.3 部署方式

**最低成本方案**：CPU 云服务器 + cron 定时任务

```bash
# crontab -e（美东时间每日 16:30 执行）
30 16 * * 1-5 /path/to/venv/bin/python /path/to/Glaubenskrieg/scripts/infer.py \
  --ctm-ckpt /path/to/models/ctm_best.pt \
  --gbdt-json /path/to/models/gbdt_best.json \
  --config /path/to/configs/default.yaml \
  --data /path/to/data/daily.csv \
  --output /path/to/results/signals.csv \
  >> /path/to/logs/infer.log 2>&1
```

**Docker 方案**（可选）：

```dockerfile
FROM python:3.12-slim
RUN pip install torch numpy pandas pyyaml scipy
COPY Glaubenskrieg/ /app/Glaubenskrieg/
COPY Hoffnung/build/ /app/Hoffnung/build/
ENV PYTHONPATH="/app/Glaubenskrieg:/app/Hoffnung/build/python"
WORKDIR /app/Glaubenskrieg
CMD ["python", "scripts/infer.py", "--ctm-ckpt", "models/ctm_best.pt", ...]
```

---

## 9. 国内云 GPU 部署

以下以 AutoDL 为例，说明国内云 GPU 上的完整部署流程。

### 9.1 创建实例

1. 登录 [autodl.com](https://www.autodl.com)，完成学生认证（85 折）
2. 选择 **GPU 云服务器** → 社区镜像
3. 搜索 **PyTorch 2.1.0 + CUDA 12.1** 官方镜像
4. 选择实例规格：

| GPU | 价格 | 显存 | 适合配置 |
|-----|------|------|---------|
| RTX 3090 | ¥1.32/h（学生价 ¥1.12） | 24GB | 所有配置 |
| RTX 4090 | ¥1.98/h（学生价 ¥1.68） | 24GB | loop_large B=64 |
| A100 80G | ¥5.98/h | 80GB | 批量实验 |

### 9.2 环境初始化

```bash
# 克隆仓库
git clone /path/to/Glaubenskrieg
git clone /path/to/Hoffnung

# 编译 Hoffnung
cd Hoffnung && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 验证 GBDT
cd /path/to/Glaubenskrieg
export PYTHONPATH="/path/to/Glaubenskrieg:/path/to/Hoffnung/build/python:$PYTHONPATH"
python3 -c "from src.model.ctm_model import CTMStockModel; from gbdt.losses import rankic_loss; print('✅ 环境就绪')"
```

### 9.3 上传数据

```bash
# 选项 A：通过 AutoDL 文件存储
# 网页端上传 → 数据位于 /root/autodl-tmp/

# 选项 B：网盘下载
cp /root/autodl-tmp/stock_prices.csv data/
```

### 9.4 训练

```bash
# 使用 ensemble 模式，GPU 训练
export PYTHONPATH="/path/to/Glaubenskrieg:/path/to/Hoffnung/build/python:$PYTHONPATH"

python scripts/train.py \
  --config configs/scale_large.yaml \
  --device cuda \
  --ensemble \
  --gbdt-loss rankic \
  --gbdt-trees 200 \
  --gbdt-depth 6 \
  --save-dir /root/autodl-tmp/models/experiment_1

# 完成后
# 评价指标在 /root/autodl-tmp/models/experiment_1/metrics.json
```

### 9.5 下载模型

训练完成后，从网页端下载 `models/` 目录，或通过网盘同步到本地推理服务器。

### 9.6 关机不计费

AutoDL 支持**关机不计费**：停止实例后 GPU 费用停止，仅收取少量存储费（系统盘约 ¥0.10/日）。适合仅在训练时开机，训练完立即关机。

### 9.7 其他国内平台价格参考

| 平台 | RTX 4090 | RTX 3090 | A100 80G | 关机不计费 |
|------|----------|----------|----------|-----------|
| **潞晨云** | ¥1.00/时 | ¥0.82/时 | — | 部分支持 |
| **智星云** | ¥1.50/时 | ¥1.00/时 | ¥4.90/时 | ✅ |
| **矩池云** | ¥1.54/时 | ¥1.29/时 | ¥5.60/时 | ✅ 启动不计费 |
| **AutoDL** | ¥1.98/时 | ¥1.32/时 | ¥5.98/时 | ✅ |
| **恒源云** | ~¥2.00/时 | ~¥1.71/时 | ¥3.04(40G) | ✅ 秒级计费 |

> **推荐**：学生用 AutoDL（85 折后 3090 仅 ¥1.12/时），追求低价用潞晨云（¥1.00/时的 4090）。

---

## 10. 验证与冒烟测试

### 10.1 CTM-Only 快速测试

```bash
cd /path/to/Glaubenskrieg

# 生成测试数据或使用已存在的 CSV
PYTHONPATH=. python scripts/train.py \
  --config configs/default.yaml \
  --device cpu \
  --save-dir /tmp/test_ctm_only 2>&1

# 验证输出
ls /tmp/test_ctm_only/
# 期望输出：config.yaml  metrics.json

python3 -c "
import json
with open('/tmp/test_ctm_only/metrics.json') as f:
    d = json.load(f)
print(f'✅ 训练完成，mean_sharpe={d.get(\"mean_sharpe\", \"N/A\"):.4f}')
"
```

### 10.2 Ensemble 快速测试（需要 GBDT）

```bash
PYTHONPATH=/path/to/Glaubenskrieg:/path/to/Hoffnung/build/python:$PYTHONPATH \
python scripts/train.py \
  --config configs/default.yaml \
  --device cpu \
  --ensemble \
  --gbdt-trees 50 --gbdt-depth 4 \
  --save-dir /tmp/test_ensemble 2>&1

# 验证输出
ls /tmp/test_ensemble/
# 期望：ensemble_ctm.pt  ensemble_gbdt.json  metrics.json  config.yaml

python3 -c "
import json
with open('/tmp/test_ensemble/metrics.json') as f:
    d = json.load(f)
print(f'✅ Ensemble 训练完成')
print(f'   mean_sharpe: {d.get(\"mean_sharpe\", \"N/A\"):.4f}')
print(f'   mean_gbdt_ic: {d.get(\"mean_gbdt_ic\", \"N/A\")}')
print(f'   mean_ensemble_sharpe: {d.get(\"mean_ensemble_sharpe\", \"N/A\")}')
"
```

### 10.3 推理验证

```bash
# 使用刚刚训练的模型进行推理
PYTHONPATH=. python scripts/infer.py \
  --ctm-ckpt /tmp/test_ensemble/ensemble_ctm.pt \
  --gbdt-json /tmp/test_ensemble/ensemble_gbdt.json \
  --config configs/default.yaml \
  --data data/stock_prices.csv \
  --output /tmp/test_ensemble/predictions.csv

# 验证输出
python3 -c "
import pandas as pd
df = pd.read_csv('/tmp/test_ensemble/predictions.csv')
print(f'✅ 推理完成：{len(df)} 行预测')
print(f'   列: {list(df.columns)}')
assert 'fused_signal' in df.columns, '缺少 fused_signal 列'
assert df['fused_signal'].notna().all(), '存在 NaN 值'
print('✅ 信号验证通过')
"
```

---

## 11. 故障排除

### 11.1 GBDT 模块导入错误

```
ModuleNotFoundError: No module named 'gbdt_python'
```

**原因**：Hoffnung 未编译或 PYTHONPATH 未包含编译产物。

**解决**：
```bash
# 确认编译存在
ls /path/to/Hoffnung/build/gbdt_python*.so

# 确认路径正确
export PYTHONPATH="/path/to/Hoffnung/build/python:$PYTHONPATH"
python3 -c "from gbdt import GBDTTrainer"
```

### 11.2 Walk-Forward 窗口不足

```
No walk-forward windows fit data
```

**原因**：数据量不足。默认 `train_window=1000, val_window=200, purge_period=126`，最少需要约 2000 行数据。

**解决**：
- 增加数据量（至少 3 年以上日频数据）
- 或减小 `train_window` / `val_window` 配置

### 11.3 CUDA 内存不足

```
CUDA out of memory
```

**解决**：
```bash
# 减小 batch_size
python scripts/train.py --config default.yaml \
  --override "trainer.batch_size=8"

# 减小序列长度
# 修改 config 中 seq_len: 63 → 30

# 使用更小的配置
# default.yaml (d=64) 替代 scale_large.yaml (d=128)
```

### 11.4 NaN 损失

```
loss: nan
```

**原因**：数值不稳定，常见于学习率过高或梯度爆炸。

**解决**：
- 降低学习率（从 3e-4 降到 1e-4）
- 检查数据是否包含 NaN/Inf
- 启用梯度裁剪（`grad_clip: 1.0` 默认已启用）

### 11.5 GBDT 降级警告

```
WARNING: gbdt_python not available, falling back to MSE-only manual boosting
```

**原因**：Hoffnung C++ 模块未找到，`ensemble_trainer.py` 自动降级为纯 Python 的 manual boosting loop。

**影响**：
- 功能 **不受影响**，但仅支持 MSE 损失
- `--gbdt-loss rankic` 和 `--gbdt-loss composite` 将不可用
- 训练速度和稳定性略低于 C++ 版本

---

## 12. 参考资料

| 文档 | 内容 | 位置 |
|------|------|------|
| **CTM 架构指南** | 完整数学推导、SSM 公式、反向传播 | `docs/ctm_architecture_guide.pdf` |
| **融合路线图** | P0-P3 集成级别设计、代码变更 | `docs/fusion_roadmap.md` |
| **GPU 性能指南** | 基准测试、内存估算、并行扫描配置 | `docs/performance.md` |
| **实盘部署指南** | Broker API、风险控制、每日运行时间线 | `docs/retail_deployment_guide.md` |
| **Hoffnung 文献库** | GBDT 量化投资论文合集 | `Hoffnung/README.md` |
| **循环 Mamba 设计** | RecurrentCTM (Loop Mamba) 架构 | `docs/loop_mamba_design.md` |

### 快速命令参考

| 操作 | 命令 |
|------|------|
| CTM-only 训练 | `python scripts/train.py --config configs/default.yaml` |
| Ensemble 训练 | `+ --ensemble --gbdt-loss rankic` |
| P3 联合训练 | `+ --p3 --modulator-epochs 10` |
| **Ensemble 推理（预训练权重）** | `+ --ensemble-inference --init-ctm-ckpt model.pt --init-gbdt-json model.json` |
| **CTM 预热 Ensemble** | `+ --ensemble --init-ctm-ckpt model.pt` |
| 推理 (CPU) | `python scripts/infer.py --ctm-ckpt best.pt --config default.yaml --data data.csv --output preds.csv` |
| 推理 (GPU) | `+ --device cuda --batch-size 256` |
| 编译 GBDT | `cd Hoffnung/build && cmake .. && make -j$(nproc)` |
| 验证 GBDT | `python3 -c "from gbdt import GBDT; print('OK')"` |

---

*本指南配套 `docs/retail_deployment_guide.md` 使用：本文档覆盖 ML 训练/推理管道，零售部署指南覆盖 Broker 交易执行。*
