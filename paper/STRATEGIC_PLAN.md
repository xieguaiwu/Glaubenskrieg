# Glaubenskrieg 系统工程论文 — 战略执行计划

> **日期**: 2026-06-07  
> **目标**: 将 Glaubenskrieg 从“实验堆”转化为可发表的系统工程论文  
> **当前状态**: 数据/代码/基线/诊断/大纲均已具备，缺 GPU 实验和论文撰写  
> **关键前提**: 2 台 GPU 服务器当前不可访问 — 必须先解决（见 §0）

---

## §0: 前置条件 — GPU 服务器访问修复

### 当前状态
| 服务器 | IP:Port | GPU | SSH 状态 |
|--------|---------|-----|:--------:|
| Server1 | 223.109.239.36:24224 | V100 ×2 | ❌ Permission denied |
| Server2 | 223.109.239.11:15512 | RTX 3090 ×2 | ❌ Permission denied |
| 本地 | — | 无 GPU | — |

**根因**: `~/.ssh/id_ed25519.pub` 未在两台服务器上授权。需要：
1. 联系服务器管理员添加 SSH 公钥，或
2. 获取密码并通过 sshpass 或交互式登录，或
3. 使用其他认证方式（跳板机等）

**阻塞影响**: 所有 DL 训练实验（CTM、LSTM、GRU、Transformer、TCN、合成数据、消融实验）**均需 GPU**。LightGBM/GARCH 基线可在 CPU 运行。

**建议**: 将“修复服务器访问”作为 **Day 0 任务**，在实验开始前完成。

---

## §1: 总工作量估算

### 按阶段分解

| 阶段 | 任务数 | 活跃人天 | 日历天 (并行后) | GPU 依赖 |
|:-----|:------:|:--------:|:--------------:|:--------:|
| §0 前置 | 1 | 0.5 | 0.5 | — |
| §3 Phase 1: 实验 | 9 | 6–8 | 4–5 | 100% |
| §4 Phase 2: 修复 | 4 | 2.5–3 | 2–3 | 50% |
| §5 Phase 3: 撰写 | 8 | 7–9 | 5–7 | — |
| **合计** | **22** | **16–20.5** | **12–16** | |

### 详细估算

| # | 任务 | 人天 | 说明 |
|:--:|------|:----:|------|
| E1 | 合成数据生成器 | 0.5 | AR(1)/正弦/因子模型数据生成脚本 |
| E2 | 合成数据实验 (S1-S4) | 1.5 | 运行 CTM+LSTM+GRU+Linear 在合成数据上 |
| E3 | 基线实现 (LSTM/GRU/Transformer/TCN) | 1.5 | 统一接口，复用 walk-forward 管线 |
| E4 | 基线实验运行 | 1.0 | 在 HK+US 数据上运行 4 个基线 |
| E5 | CTM 波动率实验 | 1.0 | 切换 target 为 realized_vol_21 |
| E6 | 消融实验 (7→3 个关键消融) | 0.5 | w/o Conv1d, w/o CrossAttn, w/o GBDT |
| E7 | T 值扫描 | 0.5 | CTM vs LSTM vs GRU, T∈{10,20,40,63,120,240} |
| E8 | 参数效率分析 | 0.3 | 只需分析，不需新实验（数据已存在） |
| E9 | 结果聚合脚本 | 0.5 | JSON→LaTeX 表自动化 |
| F1 | 数值稳定性修复 | 1.0 | 对数空间 SSM + 梯度裁剪 |
| F2 | TimeDecayGate 文档化 | 0.3 | 不修复，作为经验教训呈现 |
| F3 | FFN 消融 (CrossAssetAttention) | 0.3 | 线性投影替代 33K FFN |
| F4 | GBDT 双向融合 (交替训练) | 1.0 | 可选 — 决策树决定是否执行 |
| W1 | Sec 1-2: 引言+相关工作 | 1.5 | 大纲已有，需润色 |
| W2 | Sec 3: 方法论+架构 | 1.5 | 架构描述+设计决策 |
| W3 | Sec 4: 实验设置 | 1.0 | 插入实验配置 |
| W4 | Sec 5: 实验结果 | 1.5 | 插入所有表格+解读 |
| W5 | Sec 6: 诊断分析 | 1.0 | 5 步诊断法 |
| W6 | Sec 7: 经验教训 | 1.0 | 7 条教训+34 bug 分类 |
| W7 | Sec 8: 结论+未来工作 | 0.5 | |
| W8 | 参考文献+图表+排版 | 1.0 | LaTeX 模板+图表 |

---

## §2: 三阶段计划

### Phase 1: 补充实验 (Day 1–5，GPU 必须)

```
Day 1–2: 基础设施 + 并行实验启动
├── [本地] 合成数据生成器 (E1)
├── [本地] 基线模型实现 (E3: LSTM/GRU/Transformer/TCN)
├── [Server1+V100] 合成数据实验 (E2: S1-S4)
└── [Server2+3090] CTM 波动率实验 (E5) + T值扫描准备 (E7)

Day 2–3: 实验运行中
├── [Server1] 基线实验 US 数据 (E4: LSTM/GRU)
├── [Server2] 基线实验 HK 数据 (E4: Transformer/TCN)
├── [Server1] 消融实验 (E6: 3 个 CTM 变体)
└── [Server2] T 值扫描 (E7)

Day 3–4: 结果收集 + 聚合
├── [本地] 收集所有 JSON 结果
├── [本地] 结果聚合脚本 (E9)
└── [本地] 参数效率分析 (E8: 无需新实验)

Day 4–5: 初判决策 (见 §7 决策树)
└── 根据 CTM 波动率 + 合成数据结果 → 确定论文最终定位
```

**Phase 1 完成标志 (Milestone M1)**:
- [ ] 合成数据: CTM 恢复已知 IC 的精度曲线（4 个实验全部完成）
- [ ] 基线对比: LSTM/GRU/Transformer/TCN 在 HK+US 的 Test IC + Sharpe
- [ ] CTM 波动率: CTM vs LightGBM vs GARCH 的 QLIKE 对比
- [ ] 消融: 3 个 CTM 变体 (w/o Conv1d, w/o CrossAttn, w/o GBDT) 的 IC 差异
- [ ] 所有结果已聚合为 LaTeX-ready 表格

### Phase 2: 修复工程 (Day 5–7，部分 GPU)

```
Day 5–6:
├── [Server1] 数值稳定性修复 (F1: 对数空间 SSM)
├── [本地] FFN 消融 (F3: 线性投影替代)
├── [本地] TimeDecayGate 文档化 (F2: 不修代码)
└── [决定] GBDT 双向 (F4): 仅当 Phase 1 显示 GBDT-Modulator 有微小增益时执行

Day 6–7:
├── [Server1] 用修复后的数值稳定版 CTM 重跑关键实验
└── [本地] 修复文档化 (方法论 §3.x)
```

**Phase 2 完成标志 (Milestone M2)**:
- [ ] NaN 传播根因已确定并消除（日志确认 0 NaN）
- [ ] 对数空间 SSM 通过 243 个测试
- [ ] FFN 简化 vs 原始 FFN 的 IC/Speed 对比
- [ ] TimeDecayGate 失败分析写入 §7 经验教训

### Phase 3: 论文撰写 (Day 7–14，全本地)

```
Day 7–8:   Sec 3 (方法论+架构) + Sec 4 (实验设置)
Day 8–9:   Sec 5 (实验结果 — 依赖 Phase 1 数据)
Day 9–10:  Sec 6 (诊断分析) + Sec 7 (经验教训)
Day 10–11: Sec 1 (引言) + Sec 2 (相关工作)
Day 11–12: Sec 8 (结论) + 全篇润色
Day 12–13: 参考文献格式化 + LaTeX 图表
Day 13–14: 内部审阅 + 修改
```

**Phase 3 完成标志 (Milestone M3)**:
- [ ] 全文初稿 (~6000–8000 词)
- [ ] 所有表格/图表嵌入
- [ ] 参考文献完整 (BibTeX)
- [ ] 通过 Grammarly / 语言检查
- [ ] 内部一致性审查（表格数字与正文一致）

---

## §3: 并行化执行方案

### 两服务器分工

```
┌─────────────────────────────────────────────────┐
│ Server1 (V100 ×2, 32GB VRAM)                    │
│ ─────────────────────────────────────────────── │
│ 擅长: 大批量训练、多实验排队                     │
│ 任务:                                            │
│   1. LSTM + GRU 基线 (HK + US 数据)              │
│   2. 消融实验 ×3 (w/o Conv1d, w/o CrossAttn,    │
│      w/o GBDT)                                   │
│   3. T 值扫描: CTM T∈{10,20,40,63,120,240}       │
│   4. 数值稳定性修复后的重跑                       │
├─────────────────────────────────────────────────┤
│ Server2 (RTX 3090 ×2, 24GB VRAM)                │
│ ─────────────────────────────────────────────── │
│ 擅长: 快速迭代、小批量实验                       │
│ 任务:                                            │
│   1. 合成数据实验 (S1-S4) 全部                   │
│   2. Transformer + TCN 基线 (HK + US)            │
│   3. CTM 波动率预测实验                           │
│   4. GBDT 交替训练实验 (如果执行 F4)             │
└─────────────────────────────────────────────────┘
```

### 并行批次的依赖关系

```
Batch A (无依赖，Day 1 启动):
  A1: 合成数据生成器 [本地]
  A2: 基线模型实现 [本地]
  A3: CTM 波动率实验 [Server2]

Batch B (依赖 A1 完成，Day 1-2):
  B1: 合成数据实验 S1-S4 [Server2]
  B2: 基线实验 LSTM/GRU [Server1]

Batch C (依赖 A2 完成，Day 2):
  C1: 基线实验 Transformer/TCN [Server2]
  C2: 消融实验 [Server1]
  C3: T 值扫描 [Server1]

Batch D (依赖 B+C 完成，Day 4):
  D1: 结果聚合 + 初判决策 [本地]
  D2: 数值稳定性修复 [Server1]

关键路径: A1 → B1 → D1 (约 3 天)
非关键路径并行: A3, B2, C1, C2, C3
```

### 实验估算耗时

| 实验 | GPU | 每个种子 | 种子数 | 数据量 | 估算总时间 |
|------|:---:|:-------:|:-----:|--------|:--------:|
| CTM baseline (已有) | V100 | ~20min | 5 | HK 200×750d | 2h |
| LSTM baseline | V100 | ~8min | 3 | HK 200×750d | 25min |
| GRU baseline | V100 | ~6min | 3 | HK 200×750d | 20min |
| Transformer baseline | 3090 | ~12min | 3 | HK 200×750d | 40min |
| TCN baseline | 3090 | ~10min | 3 | HK 200×750d | 30min |
| 消融 ×3 变体 | V100 | ~15min | 3 | HK 200×750d | 2.5h |
| T 值扫描 ×6 | V100 | ~15min | 1 | 合成 | 1.5h |
| 合成数据 S1-S4 | 3090 | ~5min | 3×4 | 小型合成 | 1h |
| CTM 波动率 | 3090 | ~25min | 3 | HK 200×750d | 1.5h |
| **总计全串行** | | | | | **~10h** |
| **两服务器并行** | | | | | **~4h** |

> **注**: 实际瓶颈不在 GPU 时间（总计 10 GPU-hours），而在实现+调试+超参调优。DL 部分约 2–2.5 人天。

---

## §4: 最高杠杆任务 (HIGHEST IMPACT PER UNIT TIME)

按“对论文接受概率的提升 / 人天投入”排序：

### 第一梯队 (必须做，投入产出比极高)

| 排名 | 任务 | 人天 | 对论文的影响 |
|:----:|------|:----:|------------|
| **🥇 1** | **CTM 波动率 (E5)** | 1.0 | **论文命运决定者**。如果 CTM > LGB，论文有正面结果；如果 CTM ≤ LGB，论文只能靠方法论叙事。此实验不可跳过。 |
| **🥈 2** | **合成数据 (E1+E2)** | 2.0 | **架构合法性证明**。没有合成数据验证，"CTM 设计正确"无证据；有合成数据，即使 OHLCV 无信号，架构本身成立。 |
| **🥉 3** | **基线实现 (E3+E4)** | 2.5 | **论文可发表性的最低门槛**。没有神经网络基线，论文就是"我们的 97K 模型 = Ridge 451 参数"。有 LSTM/GRU/Transformer 对比才能声称穷举测试。 |
| 4 | **数值稳定性 (F1)** | 1.0 | **审稿人必杀技防御**。15+ NaN 检查点若不修复，审稿人可直接以"实现质量低"拒稿。修复后可作为工程挑战的正面案例。 |

### 第二梯队 (应该做，投入产出比高)

| 排名 | 任务 | 人天 | 对论文的影响 |
|:----:|------|:----:|------------|
| 5 | **消融实验 ×3 (E6)** | 0.5 | 为"系统工程记录"提供证据。w/o Conv1d 的消融特别重要，直接回答"为什么需要卷积前端"。 |
| 6 | **结果聚合自动化 (E9)** | 0.5 | 防止手动复制粘贴错误。在论文反复修改时节省大量时间。一次投入，多次复用。 |
| 7 | **T 值扫描 (E7)** | 0.5 | 回答"为什么选 Mamba 而非 GRU"。如果 CTM 在 T≥120 时优于 GRU，证明架构选择有依据。 |

### 第三梯队 (可剪裁)

| 排名 | 任务 | 人天 | 对论文的影响 |
|:----:|------|:----:|------------|
| 8 | 参数效率分析 (E8) | 0.3 | 可与 LLG bound 关联，增加理论深度。但不需要新实验，仅分析已有数据。 |
| 9 | GBDT 双向融合 (F4) | 1.0 | 如果交替训练没有增益，"我们尝试了双向融合但无效"也是诚实的发现。但投入产出比低。 |
| 10 | FFN 简化 (F3) | 0.3 | 微小消融，仅作脚注。 |
| 11 | TimeDecayGate 修复 (F2) | 0.3 | 不修复。直接定位为"经验教训：固定指数衰减形式过于严格"。 |

---

## §5: 最小可行论文 (MVP)

如果时间和资源极度紧张（例如 GPU 服务器修复延迟），以下是**绝对不可裁剪**的部分：

### 必须保留 (MVP Core)

| 组成部分 | 理由 |
|----------|------|
| **CTM 波动率实验** | 唯一可能产生正面结果的实验。如果连这个都砍了，论文全是负结果。 |
| **合成数据 S1+S2** | 至少证明架构能学习 AR(1) 和正弦信号。S3(因子模型)、S4(已知 IC)可简化为文字讨论。 |
| **LSTM + GRU 基线** | 最小神经网络基线套件。Transformer 和 TCN 可降级为"未来工作"。 |
| **数值稳定性修复** | 代码质量门控。不修复 = 拒稿风险。 |
| **消融: w/o Conv1d** | 回答审稿人第一个问题："为什么要卷积前端？" |
| **5 步诊断法 + 7 条经验教训** | 这是方法论贡献的核心，不依赖任何实验结果。 |
| **LightGBM + GARCH 波动率** | 已有结果。最可靠的正面发现。 |

### 可以剪裁 (Nice-to-have)

| 剪裁项 | 替代策略 |
|--------|---------|
| Transformer + TCN 基线 | 文字提及："由于 GPU 约束未运行，作为未来工作" |
| T 值扫描 | 文字讨论："Mamba 的选择基于其线性复杂度和对长序列的理论优势" |
| 消融 ×7 → ×3 | 保留 3 个最关键消融 |
| GBDT 双向融合 | 在 §7 经验教训中作为"已知局限"讨论 |
| CrossAssetAttention FFN 简化 | 不证明无效就不需要简化；如证明无效则化为经验教训 |
| 参数效率分析 | 在 §6 诊断分析中定性讨论 |
| TimeDecayGate 参数扫描 | 直接定性为"gate 设计失败 = 固定衰减形式太严格" |
| 美股 516 股票全量实验 | 已有 US full baseline 结果 (LGB QLIKE=-7.1, IC=0.007)，不再跑 DL |

### MVP 工作量

| 阶段 | MVP 人天 | vs 完整版 |
|------|:-------:|:--------:|
| Phase 1 | 4–5 | −3 天 |
| Phase 2 | 1.5 | −1.5 天 |
| Phase 3 | 5–6 | −2 天 |
| **合计** | **10.5–12.5** | **−6.5 天** |

MVP 日历时间: **8–10 天** (含 GPU 服务器修复时间)

---

## §6: 基础设施需求

### 6.1 需要的代码组件

| 组件 | 文件 | 状态 | 工作量 |
|------|------|:----:|:------:|
| 合成数据生成器 | `scripts/generate_synthetic.py` | ❌ 需新建 | 0.5d |
| DL 基线模型 | `src/model/baselines.py` | ❌ 需新建 | 1.5d |
| 统一基线运行器 | `scripts/run_baselines.py` | ⚠️ 扩展 `run_newdata_baselines.py` | 0.5d |
| 消融配置 | `configs/ablation/*.yaml` | ❌ 需新建 | 0.2d |
| CTM 波动率训练 | 复用 `train_volatility.py` | ⚠️ 需修改 | 0.3d |
| 结果聚合器 | `scripts/aggregate_results.py` | ❌ 需新建 | 0.5d |
| 对数空间 SSM | `src/model/mamba_block.py` | ⚠️ 需修改 | 0.5d |

### 6.2 统一接口设计

**目标**: 所有模型（CTM, LSTM, GRU, Transformer, TCN）共享相同接口，插入现有 walk-forward 管线。

```python
# src/model/baselines.py — 统一接口
class BaselineModel(nn.Module):
    """
    n_assets: int        # 股票数
    d_model: int         # 隐藏维度 (统一为 64)
    n_layers: int        # 层数 (统一为 2，匹配 CTM)
    output_dim: int      # 每资产预测维度 (统一为 1)
    """
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, N, D) → output: (B, T, N, output_dim)
        ...
```

所有 baseline 的参数数应控制在 **40K–60K** 范围（匹配 CTM 的 50K），确保对比公平。

### 6.3 Docker 环境

现有 `Dockerfile` 位于项目根目录。需要：
- 确保 PyTorch 2.x + CUDA 在容器中可用
- 预装 `lightgbm`, `arch`, `scikit-learn`
- 挂载数据卷 `/data` → `new_data/data/`

### 6.4 结果存储规范

```
results/
├── phase1/
│   ├── synthetic/
│   │   ├── s1_ar1.json
│   │   ├── s2_sine.json
│   │   ├── s3_factor.json
│   │   └── s4_known_ic.json
│   ├── baselines/
│   │   ├── lstm_hk.json
│   │   ├── lstm_us.json
│   │   ├── gru_hk.json
│   │   ├── transformer_us.json
│   │   └── tcn_hk.json
│   ├── ctm_volatility.json
│   ├── ablation/
│   │   ├── no_conv1d.json
│   │   ├── no_crossattn.json
│   │   └── no_gbdt.json
│   └── t_scan.json
└── phase2/
    ├── numerical_stability/
    │   ├── before_fix.json
    │   └── after_fix.json
    └── ffn_ablation.json
```

每个 JSON 文件遵循统一 schema：
```json
{
  "metadata": {"experiment": "...", "date": "...", "seed": 42, "gpu": "V100"},
  "config": {"model": "ctm", "n_assets": 50, ...},
  "results": {
    "per_window": [{"window": 0, "ic": 0.019, "sharpe": -0.03}, ...],
    "aggregate": {"mean_ic": 0.006, "std_ic": 0.007, "mean_sharpe": -0.028}
  }
}
```

---

## §7: 决策树 — 实验路径分叉

### 第一分支: CTM 波动率实验 (Phase 1 结束，Day 4–5)

```
CTM 波动率 QLIKE vs LightGBM QLIKE
│
├── CTM >> LightGBM (QLIKE 低 0.5+)
│   │
│   ├── 合成数据 CTM 恢复 IC ≈ 真实 IC (误差 <20%)
│   │   └── 🟢 PATH A: 架构论文 (最强)
│   │       标题: "Conv-Temporal-Mamba: When and Why SSM Architecture
│   │              Adds Value for Financial Time Series"
│   │       叙事: CTM 在信号存在时始终加值（合成+波动率双验证），
│   │             OHLCV 收益无信号是数据问题，不是架构问题
│   │       目标期刊: ICAIF / Journal of Financial Data Science
│   │
│   ├── 合成数据 CTM 无法恢复已知 IC (误差 >30%)
│   │   └── 🟡 PATH B: 方法论论文 (有正面结果)
│   │       标题: "Glaubenskrieg: A Systematic Study of Deep Learning
│   │              for Financial Volatility Prediction"
│   │       叙事: 重定向为波动率预测论文；收益预测部分缩为附录；
│   │             CTM 在波动率上有效但架构需要超参调优
│   │       目标期刊: ICAIF workshop / Quantitative Finance
│   │
│   └── 合成数据未完成 (时间不够)
│       └── 🟡 PATH B' (同 B，但缺合成验证)
│
├── CTM ≈ LightGBM (QLIKE 差异 <0.1)
│   │
│   ├── 合成数据 CTM 恢复 IC ≈ 真实 IC
│   │   └── 🟡 PATH C: 方法论论文 (标准版)
│   │       标题: "Glaubenskrieg: A Systems Engineering Approach to
│   │              Multi-Asset Financial Prediction"
│   │       叙事: 当前大纲。CTM 在合成数据上有效但在真实数据上 
│   │             LightGBM 已足够。DL 不必要但方法论有价值。
│   │       目标期刊: ICAIF / PAKDD / JFS (practitioner track)
│   │
│   └── 合成数据 CTM 无法恢复已知 IC
│       └── 🔴 PATH D: 诊断论文 (困难)
│           标题: "Systematic Failure Diagnosis of SSM Architectures
│                  for Financial Time Series"
│           叙事: 深挖为什么架构失败——T=63 太短？SSM 不适合
│                 金融数据？需要更基础的理论分析。
│           风险: 高。可能需要额外的理论工作。
│
├── CTM < LightGBM (QLIKE 更差，差异 >0.2)
│   │
│   └── 🔴 PATH E: 纯方法论/负结果论文
│       标题: "Six Paradigms, Zero Signal: A Comprehensive Negative
│              Result in ML-Based Stock Prediction"
│       叙事: 完全放弃架构创新定位，聚焦"我们穷举了 6 种方法、
│             3 个市场，结论是 OHLCV 日频无信号"。删除所有 CTM
│             特有的架构讨论（GBDTModulator、TimeDecayGate 等），
│             保留 CTM 作为"6 种方法之一"。
│       目标期刊: EACL/EMNLP (negative results track) / JDS
│       额外工作: 需大幅重写 Sec 3 架构部分
│
└── 实验无法执行 (GPU 不可用)
    └── 🔴 PATH F: CPU-only 论文 (极限裁剪)
        标题: "GARCH Beats Gradient Boosting for Volatility:
               Evidence from 877 Cross-Market Stocks"
        叙事: 仅报告已完成的 LightGBM + GARCH + Ridge 结果。
             去掉所有 DL 内容，重新定位为波动率建模基准论文。
        目标期刊: Finance Research Letters (短文，~2500 词)
        工作量: 仅需 3–5 天 (重新撰写，无新实验)
```

### 第二分支: 基线对比结果 (Phase 1, Day 3–4)

```
神经网络基线 vs Linear Ridge
│
├── 所有基线 IC ≈ 0（LSTM ≈ GRU ≈ Transformer ≈ TCN ≈ Ridge）
│   └── ✅ 无需改变论文叙事。支持"OHLCV 日频无信号"结论。
│       增强论文可信度："从 451 参数到 60K 参数，所有模型的
│       Test IC 在统计上无法区分。"
│
├── 某个 baseline IC 显著 > 0（如 LSTM IC=0.05, p<0.01）
│   └── ⚠️ 重大转向。该 baseline 成为论文的主要竞争者。
│       必须重新评估 CTM 的价值。可能定位为：
│       "We find LSTM outperforms Mamba SSM for daily financial 
│        prediction (T=63), suggesting SSM architectures are not
│        universally optimal for short financial sequences."
│       这本身是一个可发表的新发现。
│
├── Transformer 显著 > 所有模型
│   └── ⚠️ "Cross-temporal attention 比 SSM 更适合金融数据"
│       是颠覆性发现。论文价值大幅提升。
│
└── 所有 baseline 都崩溃 (NaN/不收敛)
    └── ❌ 展示金融数据的训练难度 → 作为额外经验教训呈现。
        但需要基本调试，大概率不是这种情况。
```

### 第三分支: 合成数据 CTM 失败 (Phase 1, Day 2–3)

```
合成数据 CTM 无法恢复已知 IC (即使 SNR 很高)
│
├── 诊断: 是 Mamba 本身？还是训练协议？
│   ├── 测试 LSTM 在相同合成数据上 → 如果 LSTM 也不行，是训练协议问题
│   └── 测试 CTM 在更长序列 (T=200) → 如果恢复，是 T=63 的问题
│
├── 如 LSTM 可恢复但 CTM 不能 → Mamba SSM 本身就坏了
│   └── 🔴 严肃问题。论文必须揭露这个漏洞。
│       可能定位为："Mamba SSM Degrades on Short Financial Sequences"
│
├── 如 LSTM 也不能 → 训练协议有 bug
│   └── 修复训练协议（shuffle? 早停？学习率？）
│       这反而是有价值的调试文档
│
└── 如 CTM 在 T=200 可恢复 → 确认是 T=63 过短
    └── 呼应弱点 #1。论文叙事："CTM requires T ≥ 120 for meaningful 
       SSM state; daily OHLCV windows (T=63) are too short."
```

---

## §8: 风险注册表

| # | 风险 | 概率 | 影响 | 缓解策略 |
|:--:|------|:----:|:----:|---------|
| R1 | GPU 服务器持续不可用 | 高 | 🔴 阻塞 | 退至 PATH F (CPU-only)；或申请云 GPU (Colab Pro+ $10) |
| R2 | CTM 波动率 = LightGBM | 中 | 🟡 削弱 | 退至 PATH C (方法论论文)；提升诊断分析部分比重 |
| R3 | CTM 波动率 < LightGBM | 低‑中 | 🔴 重写 | 退至 PATH E (纯负结果)；删除架构创新定位 |
| R4 | 合成数据 CTM 无法恢复 IC | 低 | 🔴 严重 | 深挖调试；T 值扫描；或放弃架构定位 |
| R5 | 基线模型无法收敛 | 低 | 🟡 延迟 | 超参扫描；简化模型；或降级为文字讨论 |
| R6 | 数值稳定性修复后性能更差 | 低 | 🟡 延迟 | 回退原始实现 + 梯度裁剪作为最小修复 |
| R7 | 论文撰写时间超预期 | 中 | 🟡 延迟 | 并行写作 + 模板复用；目标短文 (~5000 词) |
| R8 | 审稿人认定"负结果无价值" | 中 | 🔴 拒稿 | 提前准备 rebuttal：引用 Nature 2025 可复现性危机、McElfresh 176 数据集 |

---

## §9: 论文目标与投稿策略

### 投稿优先级

| 优先级 | 目标 | 类型 | 字数 | 难度 | 适合路径 |
|:------:|------|------|:---:|:----:|:--------:|
| 1 | **ICAIF 2026** (ACM Int'l Conf on AI in Finance) | 会议 | ~6000 | 高 | A, B, C |
| 2 | **Journal of Financial Data Science** | 期刊 | ~8000 | 中高 | A, B, C |
| 3 | **PAKDD 2026** (Industry Track) | 会议 | ~5000 | 中 | C, D |
| 4 | **ICAIF Workshop** (挑战赛/短文) | 工作坊 | ~3000 | 低中 | B, C, E |
| 5 | **Finance Research Letters** | 短文期刊 | ~2500 | 中 | F |
| 6 | **EMNLP Negative Results Track** | 会议 | ~4000 | 低 | E |

### 时间线 (倒推)

```
目标: ICAIF 2026 (投稿截止假定 2026-09-01)

2026-06-07: 战略计划制定 ✅ (今日)
2026-06-08: GPU 服务器修复 (§0)
2026-06-12: Phase 1 实验完成 (M1)
2026-06-14: Phase 2 修复完成 (M2)
2026-06-21: Phase 3 初稿完成 (M3)
2026-06-28: 内部审阅 + 修改
2026-07-15: 提交 arXiv 预印本
2026-08-15: 最终版完成
2026-09-01: 投稿截止
```

---

## §10: 立即行动清单 (本周，2026-06-07 → 2026-06-13)

### 🔴 阻塞项 (必须解决才能继续)

- [ ] **修复 GPU 服务器访问** — 尝试：
  - `ssh-copy-id root@223.109.239.36 -p 24224` (如果知道密码)
  - `ssh-copy-id root@223.109.239.11 -p 15512`
  - 联系管理员添加公钥 `~/.ssh/id_ed25519.pub`
  - 备选: Google Colab Pro+ (≈$10/月, T4 GPU)

### 🟡 可并行启动 (无需 GPU)

- [ ] 编写 `scripts/generate_synthetic.py` (合成数据生成器)
- [ ] 编写 `src/model/baselines.py` (LSTM/GRU/Transformer/TCN 实现)
- [ ] 编写 `scripts/aggregate_results.py` (结果→LaTeX 聚合器)
- [ ] 精读 `paper/NEGATIVE_RESULTS_REPORT.md` 确认数据一致性
- [ ] 精读 `paper/CLASSIC_LITERATURE_COMPARISON.md` 确认文献对比准确

### 🟢 可立即完成 (已有数据)

- [ ] 参数效率分析 (E8): 从已有结果计算 P/T ratio，与 LLG bound 对比
- [ ] TimeDecayGate 失败分析 (F2): 撰写 §7 经验教训文字
- [ ] 更新 `paper/00_paper_outline.md` 与当前战略对齐

---

*本计划将随着实验推进（尤其是 Phase 1 结束时的决策点）动态更新。*
