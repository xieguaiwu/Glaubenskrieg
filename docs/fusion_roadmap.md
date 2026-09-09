# Glaubenskrieg + Hoffnung：融合路线图

> 撰写日期：2026-05-23 | 范围：P0–P3 融合规划
> 当前状态：P0 (Ensemble) + P1 (特征融合) 已完成 | RecurrentCTM (Loop Mamba) 可用

---

## 已实现：P0 + P1 概述

```
┌──────────────────────────────────────────────────────────────┐
│                    当前融合状态                                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  P0 Ensemble ──── IC加权融合 (已验证)                          │
│  ┌────────────────────────────────────────────────────┐      │
│  │  CTM (Mamba SSM) + GBDT (C++ Tree)               │      │
│  │  → 独立训练，IC加权合成 → fused_signal              │      │
│  │  → --ensemble 命令行开关 + 6个GBDT超参               │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
│  P1 特征融合 ──── CTM隐藏态→GBDT特征 (已验证)                   │
│  ┌────────────────────────────────────────────────────┐      │
│  │  CTM.extract_features() → hidden_states            │      │
│  │  → pool(both=last+mean) → concat(原始聚合6种)      │      │
│  │  → 全部喂入GBDT训练                                 │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## P2：损失桥接（Loss Bridging）

### 目标
让 GBDT 直接优化量化指标（Sharpe Ratio / RankIC / 复合损失），而非 MSE。

### 当前状态
- `rankic_loss_gbdt_style` 已实现（可微排名近似 → 梯度 → Hessian）
- 手动 boosting 循环已通过验证（`trainer=manual_loop`）
- Python GBDTTrainer 因 `gbdt/__init__.py` 损坏无法加载

### 路线

```
短期修复 (1天) ─── 修复 Hoffnung Python 包 ── 启用 GBDTTrainer
  Fix: gbdt/losses.py 缺少 log_loss 和 rankic_loss 导出
  → 修复后 from gbdt import GBDTTrainer 可正常加载
  → 直接使用 GBDTTrainer(config, loss_fn=rankic_loss) 训练
  → 比 manual_loop 更稳定（含早停/线搜索/验证集追踪）

中期增强 (1周) ─── CTM复合损失 → GBDT 优化目标
  CTM损失 = α·MSE + β·(1-Sharpe) + γ·Directional + δ·Pinball
                            ↓
  GBDT损失 = CTM_composite_loss(y_true, y_pred)
                            ↓
  实现: loss_bridge.py → compute_gradients → GBDT.fit_one_tree()
  
  这使GBDT的树分裂直接优化与CTM完全相同的目标函数

长期 (2周+) ─── Sharpe作为GBDT原生损失
  C++ core 的 GradientBridge 扩展: 实现 sharpe_gradient_hessian()
  减少 Python↔C++ 桥接开销，提升训练速度
```

### 代码变更清单
| 文件 | 变更 | 估算 |
|------|------|------|
| `Hoffnung/python/gbdt/losses.py` | 补全 `log_loss`, `rankic_loss` 导出 | 0.5天 |
| `Glaubenskrieg/src/train/loss_bridge.py` | 新建: CTM复合损失 → GBDT梯度桥 | 1天 |
| `Hoffnung/src/gbdt.cpp` | 新增 `sharpe`, `rankic` 原生损失 | 2天 |

### 预期收益
- 切换到 `GBDTTrainer` + `rankic_loss` 后, GBDT 直接优化排序指标
- 对比 MSE 训练的 GBDT，RankIC 预期提升 2-5%（参考东方证券 DFQ-XGB 报告）

---

## P3：完全融合（CTM ← GBDT 双向桥接）

### 目标
将 GBDT 的树结构嵌入 CTM 作为「可微截面专家模块」，实现端到端联合训练。

### 架构

```
Input(B,N,T,D)     ← N只股票 × T时间 × D因子
     │
     ▼
┌─────────────────────────────────────┐
│ CTM 时序编码器 (共享权重)              │
│ CausalConv → MambaBlock×L → RMSNorm │  ← 时序建模
│ 每只股票独立处理                        │
└──────────────┬──────────────────────┘
               │ (B,N,T,d_model)
               ▼
┌─────────────────────────────────────┐
│ GBDT 截面交叉专家模块                  │
│                                     │
│  对每个时间步 t:                      │
│    X_t = (B,N,d_model)              │  ← 截面数据
│    for tree in ensemble:            │
│      leaf_output = traverse(X_t)    │
│    gbdt_out = Σ leaf * step_size    │  ← 树集成输出
│                                     │
│  可微化: 对分裂做软路由 (sigmoid)      │
│  或: GBDT输出作为CrossAttention偏置   │
└──────────────┬──────────────────────┘
               │ (B,N,T,d_model)
               ▼
┌─────────────────────────────────────┐
│ CrossAssetAttention                  │  ← 股票间关系
│ + Output Heads (回归+分类)           │
└─────────────────────────────────────┘
               │
           Output(B,N,T,C)
```

### 两种实现路径

#### 路径 A：软路由树 (Neural Tree)
```
每个内部节点: gate = σ(w·x + b) ∈ (0,1)
路径概率 = Π(gate_left or gate_right)
叶子输出 = 加权平均所有叶子 * 路径概率
```

- **优点**: 完全可微，可以与 CTM 联合 BP
- **缺点**: 训练不稳定，叶子数受限
- **参考**: NODE (Neural Oblivious Decision Ensembles), TabNet
- **框架**: 纯 PyTorch 实现，不依赖 Hoffnung C++

#### 路径 B：GBDT 偏置注意力
```
GBDT 输出 → 转换为 CrossAssetAttention 的偏置项：
  attn_bias = α · GBDT_correlation_matrix
  原始注意力 = Q·K^T/√d + attn_bias
```

- **优点**: 保持 C++ GBDT 效率，改动最小
- **缺点**: GBDT 不参与 BP，只能周期性重训练
- **推荐**: 短期可实现，快速验证

### 实验设计
```
对比组:
  A. CTM-only (基线)
  B. CTM + GBDT Ensemble (P0, 已完成)
  C. CTM + GBDT 特征融合 (P1, 已完成)
  D. CTM + GBDT 损失桥接 (P2)
  E. CTM-GBDT 联合模型 (P3)

评估维度:
  - RankIC / ICIR / 多头超额收益
  - 夏普比率 / 最大回撤
  - 训练时间 / 推理延迟
  - 模型间预测相关性
```

---

## P4：工程化落地

### 1. 因子归因与解释性 ✅ (2026-05-23)

```
当前状态: CTM + GBDT 特征重要性可通过 explain() 获取
已实现:
  ┌─ GBDT 特征重要性 (频率/增益/覆盖) → explain() 方法
  ├─ SHAP 值: TreeSHAP — 待实现
  └─ CTM 注意力可视化 — 待实现

实现: ensemble_trainer.py 已增加 explain() 方法
      训练完成后自动调用 explain() 输出 top-k 因子
      结果保存在 metrics.json 的 feature_importance 字段
```

### 2. 模型序列化与部署 ✅ (2026-05-23)

```
格式:
  CTM: torch.save(state_dict) + YAML config ✅
  GBDT: GBDT.to_json() → 纯文本 JSON ✅
  Ensemble: JSON + PT 双格式 ✅
  训练时使用 --save-dir <path> 自动保存

推理管线:
  python infer.py --ctm-ckpt best.pt --gbdt-json model.json ✅
  → 输出 fused_signal + 因子归因 + 置信度
```

### 3. 回测集成 ✅ (2026-05-23)

```
对接 backtrader / zipline:
  ┌─ 每日: CTM推演 → GBDT推演 → IC加权融合 — scripts/infer.py
  ├─ 换仓: 根据 fused_signal 排序 → topK 持仓 — scripts/backtest.py
  └─ 归因: 每日记录 CTM_IC / GBDT_IC / Fused_IC

单独回测:
  python scripts/backtest.py --predictions results/predictions.csv

一步到位: 推理 → 回测
  python scripts/backtest.py --infer --infer-args "--ctm-ckpt ... --gbdt-json ..."
```

---

## 优先级矩阵

| 等级 | 项目 | 难度 | 收益 | 工作量 | 状态 |
|------|------|------|------|--------|------|
| **P0** | 修复 Hoffnung Python 包 | ⭐ | 高 | 0.5天 | ✅ 完成 |
| **P1** | `loss_bridge.py` + CTM复合损失→GBDT | ⭐⭐ | 高 | 1天 | ✅ 完成 |
| **P2** | 因子归因 + explain()管线 | ⭐⭐ | 高 | 1天 | ✅ 完成 |
| **P3** | 模型序列化 + 推理管线 | ⭐⭐ | 高 | 1天 | ✅ 完成 |
| **P4** | 三阶段融合 (P3) | ⭐⭐⭐ | 中高 | 1周 | ✅ v1完成 |
| **P5** | 预训练权重 Ensemble 加载 | ⭐⭐ | **高** | 1天 | ✅ **完成** |
| **P6** | SHAP / TreeSHAP 解释 | ⭐⭐ | 中 | 2天 | ⏳ 待做 |
| **P7** | GBDT 原生 Sharpe 损失 (C++ core) | ⭐⭐⭐ | 中 | 2天 | ⏳ 待做 |
| **P8** | 回测集成 (backtrader/zipline) | ⭐⭐⭐ | 高 | 3天 | ⏳ 待做 |
| **P9** | P3 完全融合 — 软路由树 | ⭐⭐⭐⭐ | 未知 | 2周 | 🔬 研究 |

---

## 下一步建议（按顺序）

```
第1步: 修复 Hoffnung Python 包
  → 补全 losses.py 缺失的导出
  → 验证 from gbdt import GBDTTrainer 可用
  → 切换 ensemble_trainer 到 GBDTTrainer 路径
  → 对比 manual_loop vs GBDTTrainer 的稳定性和速度

第2步: 实现 loss_bridge.py
  → CTM 复合损失 → compute_gradients → GBDT 梯度
  → 使 GBDT 直接优化 Shar peratio + Directional accuracy
  → 在 A 股真实数据上对比 MSE vs 复合损失的 GBDT IC

第3步: 因子归因管线
  → GBDT 特征重要性 → 每日输出 top-10 因子
  → CrossAssetAttention 热力图 → 行业/风格暴露监控
  → 对比 CTM-only, GBDT-only, Ensemble 的暴露差异

第4步: P3 实验 (GBDT偏置注意力)
  → GBDT 输出转换 → CrossAssetAttention 偏置
  → 与 P0 Ensemble 对比预测相关性
  → 如果相关性 < 0.7 → 融合价值极高
```

---

## RecurrentCTM 兼容性

`RecurrentCTM`（Loop Mamba）是 `CTMStockModel` 的子类，与融合管线**完全兼容**：

| 融合路径 | 兼容性 | 说明 |
|----------|--------|------|
| P0 Ensemble | ✅ | `--n-loop 5` + `--ensemble` 启用循环 GBDT 集成 |
| P1 特征融合 | ✅ | `extract_features()` 返回循环细化后的隐藏状态 |
| P2 损失桥接 | ✅ | `encode()` 已重写，含完整循环，`cond` 条件注入支持 |
| P3 完全融合 | ⚠️ | 需验证 `MultiAssetCTM` + `RecurrentCTM` 的组合 |

使用方式：
```bash
# RecurrentCTM-only
python scripts/train.py --config configs/scale_loop.yaml

# RecurrentCTM + GBDT Ensemble
python scripts/train.py --config configs/scale_loop.yaml --ensemble

# 或覆盖循环次数
python scripts/train.py --config configs/default.yaml --n-loop 5 --ensemble
```

### P3 路径 B 实现状态 (2026-05-23)

已完成：
- ✅ `FusedMultiHeadCrossAttention` (多头注意力, H=4)
- ✅ `GBDTModulator` (MLP 偏置调制器)
- ✅ `RecurrentCTM` 循环内融合支持
- ✅ `MultiAssetCTM` fused attention 标志 + GBDT 预测注入
- ✅ **三阶段训练流程** (`src/train/p3_trainer.py`): CTM预热 → GBDT拟合 → 调制器微调

待完成（真实训练环境就绪后）：
- [ ] 自适应循环深度（基于隐藏状态收敛的提前停止）
- [ ] 多专家 GBDT (K=3, 动量/反转/波动率)

---

## TGPE：时间门控渐进集成 (v4 → v5)

> 状态: v4 已完成 (结论: 劣于 v3 baseline) → v5 修复中

### TGPE 三变体

| 变体 | CLI | 描述 | v4 结果 |
|------|-----|------|---------|
| A: TimeDecayGate | `--time-gate --finetune-gate` | 可学习 α,β,γ 控制 CTM→GBDT 时变权重 | IC=0.028 (劣于 v3 0.049) |
| B: Time-Amp GBDTModulator | `--time-amp --p3` | GBDT 偏置随 τ 线性放大 | 未充分测试 |
| C: Curriculum Dropout | `--curriculum` | 训练时渐进归零 CTM 后期预测 | IC=0.065 (劣于 v3 0.142) |

### v4 五大致命缺陷 (已在 v5 修复)

1. Curriculum dropout 在 test-prep 中 80% 归零 → 模型坍缩
2. test-prep 模型随机初始化 → 与 walk-forward 无关
3. step_size=21 → 96% 窗口重叠，3× 计算量无益
4. Gate 微调仅 10 epoch, lr=1e-4 → 参数移动 <0.6%
5. GBDT 坍缩时仍执行 gate 微调

### v5 关键改进

- Gate 微调: 50 epoch, lr=5e-4, GBDT 质检通过后执行
- step_size=63 (4 窗口)
- test-prep warm-start 从最佳 walk-forward 窗口
- 移除 test-prep 中的 curriculum dropout
- 新硬件: V100 + RTX 3090 (原生 CUDA, 25-50× Ascend 910B 速度)

---

## 附录：当前管线使用方式

```bash
# CTM-only
python scripts/train.py --config configs/default.yaml

# CTM + GBDT Ensemble (P0 + P1)
python scripts/train.py --config configs/default.yaml  \
  --ensemble                                              \
  --gbdt-trees 200 --gbdt-depth 6 --gbdt-lr 0.05         \
  --gbdt-loss mse                                         \
  --gbdt-subsample 0.8 --gbdt-colsample 0.8

# CTM + GBDT RankIC (P2 就绪后)
python scripts/train.py --config configs/default.yaml  \
  --ensemble                                              \
  --gbdt-trees 100 --gbdt-depth 4                        \
  --gbdt-loss rankic

# P3 三阶段融合 (CTM预热 → GBDT拟合 → 调制器微调)
python scripts/train.py --config configs/default.yaml       \
  --multi-asset --n-assets 10                              \
  --ensemble --p3                                          \
  --modulator-epochs 10 --modulator-lr 1e-4                \
  --gbdt-loss rankic                                       \
  --gbdt-trees 100 --gbdt-depth 4

# 训练 + 保存模型
python scripts/train.py --config configs/default.yaml       \
  --ensemble --save-dir models/experiment_1

# 推理: 使用已训练模型预测新数据
python scripts/infer.py                                     \
  --ctm-ckpt models/experiment_1/ensemble_ctm.pt           \
  --gbdt-json models/experiment_1/ensemble_gbdt.json       \
  --config configs/default.yaml                             \
  --data data/new_data.csv                                  \
  --output results/predictions.csv
```

### 接入 Hoffnung 的前提

```bash
# 必须已编译 Hoffnung C++ 核心
cd /path/to/Hoffnung/build
cmake .. && make -j$(nproc)

# train.py 会自动在 sys.path 中加入该路径
# 如果编译产物放在非标准位置，设置环境变量:
export GBDT_BUILD_DIR=/path/to/Hoffnung/build
```
