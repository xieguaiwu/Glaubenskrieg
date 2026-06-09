# ⚔️ Glaubenskrieg — Mamba 量化投资系统

<p align="center">
  <a href="README.md">🇬🇧 English</a>
</p>

**Mamba 状态空间模型 + GBDT 集成，用于股票收益与波动率预测。**

一个严谨的 PyTorch 系统，结合选择性状态空间模型（Mamba）、因果时序卷积、季节性趋势分解和梯度提升树，用于多资产金融时序预测。

## 快速开始

```bash
pip install -e .

# 训练 CTM 模型
python scripts/train.py --config configs/default.yaml --device cuda

# 推理
python scripts/infer.py --ckpt checkpoints/best.pt --data data/features.csv

# 回测
python scripts/backtest.py --predictions results/predictions.csv

# 与基线对比
python scripts/baseline_compare.py --ctm results/ctm.json --gbdt results/gbdt.json
```

## 架构

```
OHLCV → CausalConv → SeasonalTrendDecomp → MambaBlock×N
                                                ↓
                                   [Bi-Mamba 反向传播]
                                                ↓
                                   多任务输出头
                                   ├── 收益预测 (IC Loss)
                                   ├── 波动率预测 (QLIKE)
                                   └── 方向分类
                                                ↓
                                   GBDT 集成（可选）
                                                ↓
                                   投资组合优化 → 信号
```

| 组件 | 描述 |
|------|------|
| `CausalConv1d` | 深度可分离因果卷积，提取局部时序模式 |
| `SeasonalTrendDecomp` | 可学习的趋势+季节+残差分解 |
| `MambaBlock` | 选择性 SSM（Gu & Dao 2023），输入依赖的状态转移 |
| `Bi-Mamba` | 前向+反向 Mamba 传递实现双向上下文 |
| `Ensemble` | CTM + LightGBM 堆叠集成，带时间门控融合 |
| `Curriculum Trainer` | 递进式训练：简单→困难样本，MSE→Sharpe→IC Loss |

## 特性

- **滚动验证**（Purged Walk-Forward），杜绝前视偏差
- **三重标签法**（Triple-Barrier）用于监督收益预测
- **分数阶差分**，在不完全差分的情况下保持平稳性
- **小波去噪**，从噪声价格数据中提取信号
- **多资产**截面+时序注意力机制
- **波动率预测**，带 GARCH(1,1) 基线和 QLIKE 损失
- **投资组合优化**：均值方差、风险平价、体制自适应
- **SHAP 解释器**，用于特征重要性分析

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

## 项目结构

```
src/
├── data/           # 数据集、特征工程、标签、滚动划分
├── model/          # CTM、Mamba 模块、注意力、集成、损失函数
├── train/          # 训练器（标准、高级、递进式、集成）
├── utils/          # 指标（IC、Sharpe、QLIKE）、序列化、SHAP
└── execution/      # 实盘交易经纪商适配器（Alpaca）

scripts/
├── train.py        # 主训练入口
├── infer.py        # 推理 / 预测
├── backtest.py     # 滚动回测
├── baseline_compare.py    # 多模型对比
├── portfolio_optimizer.py # 投资组合构建
├── garch_baseline.py      # GARCH(1,1) 波动率模型
├── volatility_backtest.py # 波动率回测
├── download_sp500.py      # 数据下载
├── enhanced_features.py   # 特征工程
└── synthetic_data.py      # 合成数据测试

configs/            # YAML 配置文件
tests/              # Pytest 测试套件（20+ 测试模块）
```

## 环境要求

- Python ≥ 3.10, PyTorch ≥ 2.0, numpy, pandas, scipy, pyyaml, LightGBM
- 可选：`alpaca-py`（实盘交易），`shap`（可解释性）

## 许可证

MIT License。

## 参考文献

- Gu & Dao (2023): *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Shi (2024): *MambaStock: Selective SSM for Stock Prediction*
- Gu, Kelly & Xiu (2020): *Empirical Asset Pricing via Machine Learning*
- López de Prado (2018): *Advances in Financial Machine Learning*

<p align="center">
  <a href="README.md">🇬🇧 English</a>
</p>
