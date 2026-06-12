# ⚔️ Glaubenskrieg — Mamba + C++ GBDT 量化投资系统

<p align="center">
  <a href="README.md">🇬🇧 English</a>
</p>

**统一的量化研究系统，融合 Mamba 状态空间模型、因果时序卷积和自研高性能 C++ GBDT 引擎，用于股票收益与波动率预测。**

Glaubenskrieg 集成了两大引擎：
- **CTM（Conv-Temporal-Mamba）** — 严谨的 PyTorch 系统，包含选择性 SSM、季节趋势分解和滚动验证
- **Hoffnung** — 自研 C++17 梯度提升决策树库，支持量化专用损失函数和 Python 绑定

## 快速开始

```bash
# 安装 Python 依赖
pip install -e .

# 训练 CTM 模型
python scripts/train.py --config configs/default.yaml --device cuda

# CTM vs LightGBM vs GARCH 基准对比
python scripts/baseline_compare.py --seeds 42,123,456,789,1024

# 回测
python scripts/backtest.py --predictions results/predictions.csv

# --- Hoffnung C++ GBDT ---
cd Hoffnung
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
make -j$(nproc)
cd ../python && pip install -e .
```

```python
# Hoffnung GBDT 使用示例
from gbdt import GBDTRegressor, RankICLoss
model = GBDTRegressor(n_trees=200, max_depth=6, loss=RankICLoss())
model.fit(X_train, y_train)
scores = model.predict(X_test)
```

## 系统架构

```
                    ╔═══════════════════════════════════╗
                    ║       Glaubenskrieg 系统          ║
                    ╚═══════════════════════╤═══════════╝
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            ▼                               ▼                               ▼
   ┌─────────────────┐           ┌─────────────────────┐       ┌─────────────────────┐
   │  CTM (PyTorch)  │           │  Hoffnung (C++17)   │       │  基线模型            │
   │  Mamba SSM      │           │  自研 GBDT 引擎     │       │  LightGBM / GARCH   │
   │  CausalConv     │           │  基于直方图分裂     │       │  Ridge / XGBoost    │
   │  SeasonalTrend  │ ──────►   │  逐叶生长           │ ────► │                     │
   │  Bi-Mamba       │ 集成融合  │  量化专用损失       │ 对比   │                     │
   │  多任务输出头   │           │  (RankIC, Sharpe)   │        │                     │
   └────────┬────────┘           └──────────┬──────────┘       └─────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
            ┌──────────────────────────────┐
            │   投资组合优化器              │
            │   均值方差 / 风险平价         │
            │   / 体制自适应                │
            └──────────────────────────────┘
```

### CTM 引擎（`src/`、`scripts/`）

| 组件 | 描述 |
|------|------|
| `CausalConv1d` | 深度可分离因果卷积，提取局部时序模式 |
| `SeasonalTrendDecomp` | 可学习的趋势+季节+残差分解 |
| `MambaBlock` | 选择性 SSM（Gu & Dao 2023），输入依赖的状态转移 |
| `Bi-Mamba` | 前向+反向 Mamba 传递实现双向上下文 |
| `Ensemble` | CTM + LightGBM 堆叠集成，带时间门控融合 |
| `Curriculum Trainer` | 递进式训练：简单→困难样本，MSE→Sharpe→IC Loss |

### Hoffnung GBDT 引擎（`Hoffnung/`）

| 组件 | 描述 |
|------|------|
| `C++17 核心` | LibTorch 后端，直方图分裂，逐叶生长 |
| `Python 绑定` | pybind11，无缝对接 numpy/torch |
| `量化专用损失` | RankIC、方向夏普、Huber、MAE、MSE |
| `多输出` | 联合预测收益率和波动率 |
| `OpenMP 并行` | 直方图构建 |

## 特性

- **滚动验证**（Purged Walk-Forward），杜绝前视偏差
- **三重标签法**（Triple-Barrier）用于监督收益预测
- **分数阶差分**，在不完全差分的情况下保持平稳性
- **小波去噪**，从噪声价格数据中提取信号
- **多资产**截面+时序注意力机制
- **波动率预测**，带 GARCH(1,1) 基线和 QLIKE 损失
- **投资组合优化**：均值方差、风险平价、体制自适应
- **SHAP 解释器**，用于特征重要性分析
- **C++ GBDT**，含自定义量化损失函数，高性能集成

## 项目结构

```
├── src/                    CTM PyTorch 源码
│   ├── data/               数据集、特征工程、标签、滚动划分
│   ├── model/              CTM、Mamba 模块、注意力、集成、损失函数
│   ├── train/              训练器（标准、高级、递进式、集成）
│   ├── utils/              指标（IC、Sharpe、QLIKE）、序列化、SHAP
│   └── execution/          实盘交易经纪商适配器（Alpaca）
├── scripts/                训练、推理、回测、基准测试脚本
├── Hoffnung/               C++ GBDT 引擎（独立子项目）
│   ├── src/                C++ 源码（gbdt.cpp、builder.cpp、bindings）
│   ├── include/gbdt/       公开 C++ 头文件
│   ├── python/gbdt/        Python 绑定（GBDTRegressor、losses、pipeline）
│   ├── tests/              Python 和 C++ 测试套件
│   └── CMakeLists.txt      CMake 构建配置
├── configs/                YAML 配置文件
├── tests/                  Pytest 测试套件（20+ 测试模块）
├── data/                   训练数据
├── results/                实验结果、基准测试、回测
└── checkpoints/            训练好的模型权重
```

## 训练

```bash
# 仅 CTM
python scripts/train.py --config configs/default.yaml --device cuda

# 多随机种子
python scripts/train.py --config configs/default.yaml --seeds 42,123,456

# 递进式训练
python scripts/train.py --config configs/scale_loop.yaml --curriculum

# GBDT 基线
python scripts/train_gbdt_only.py --data data/features.csv
```

## 评估

```bash
# 完整基准测试：CTM vs LightGBM vs GARCH
python scripts/baseline_compare.py --seeds 42,123,456,789,1024

# 回测 + 组合构建
python scripts/backtest.py --predictions results/predictions.csv --capital 1000000

# GARCH 基线
python scripts/garch_baseline.py --data data/returns.csv
```

## 环境要求

- Python ≥ 3.10, PyTorch ≥ 2.0, numpy, pandas, scipy, pyyaml, LightGBM
- 可选：`alpaca-py`（实盘交易），`shap`（可解释性）
- Hoffnung C++ GBDT 需要：CMake ≥ 3.18, LibTorch, pybind11, OpenMP

## 许可证

MIT License。

## 参考文献

- Gu & Dao (2023): *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Shi (2024): *MambaStock: Selective SSM for Stock Prediction*
- Gu, Kelly & Xiu (2020): *Empirical Asset Pricing via Machine Learning*
- López de Prado (2018): *Advances in Financial Machine Learning*
- Friedman (2001): *Greedy Function Approximation: A Gradient Boosting Machine*
- Chen & Guestrin (2016): *XGBoost: A Scalable Tree Boosting System*

<p align="center">
  <a href="README.md">🇬🇧 English</a>
</p>
