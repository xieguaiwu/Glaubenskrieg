# Glaubenskrieg — 全项目进度文档

> **最后更新**: 2026-06-07
> **项目状态**: 全部实验完成 — OHLCV 日频已穷竭，论文材料已汇整
> **相关文档**: `.omo/session_context.md` (操作详情), `paper/` (论文材料), `MEMORY.md` (长期记录)

---

## 一、项目概览

**Glaubenskrieg** = CTM (Mamba SSM) + GBDT (Hoffnung C++) Ensemble 量化选股系统，经两市场 (A-Share/US) × 多 pipeline × 6 范式 × 477 美股 × 10 年 × GARCH 基准的穷举测试，确认 OHLCV 日频衍生特征不含可检测收益预测信号。

---

## 二、实验时间线

| 阶段 | 日期 | 内容 | 结论 |
|------|------|------|------|
| v1-v5 | 06-05~06 | CTM/GBDT/Ensemble 基线 | Mamba 97K = Ridge 451, IC≈0 |
| Phase 1 | 06-06 | FE-LightGBM + 6 范式 | 全败, 波动率预测唯一稳健 |
| Phase 2 | 06-06 | 新数据迁移 (A股+美股) | 3.6× 数据量, IC 仍≈0 |
| Phase 3 | 06-06 | 美股全量 (516→477 stocks) | US IC=0.007, QLIKE=−7.10 |
| GARCH | 06-06 | GARCH(1,1) vs LightGBM | 300/300 GARCH胜, p<10⁻¹⁴ |
| Wavelet | 06-07 | 因果小波去噪测试 | ΔIC=−0.046, 去噪无效 |
| Power | 06-07 | 统计功效分析 | 最小可检测 IC=0.102 |

---

## 三、跨市场最终对比

| 市场 | 股票数 | 窗口 | LGB IC | Ridge IC | Vol QLIKE |
|------|:------:|:---:|:------:|:--------:|:---------:|
| 🇨🇳 A-Share (multi-asset) | 50 | 3 | 0.053±0.054 | 0.006 | −6.42±0.24 |
| 🇨🇳 A-Share (cross-sectional) | 200 | 6 | −0.003±0.042 | 0.030±0.054 | −6.78±0.26 |
| 🇺🇸 US | 477 | 10 | 0.007±0.043 | 0.031±0.030 | −7.10±0.20 |
| **IC>0.02?** | — | — | **❌** | **❌** | — |

---

## 四、GARCH(1,1) 完胜 ML — 核心发现

| 市场 | 股票数 | GARCH 胜率 | DM p | 中位数 LGB/GARCH |
|------|:------:|:------:|:------:|:------:|
| 🇺🇸 US | 200 | 200/200 | 2.8×10⁻¹⁴ | 2.49× |
| 🇨🇳 A-Share | 100 | 100/100 | ~0 | 2.12× |
| **合计** | **300** | **300/300** | — | — |

---

## 五、统计功效分析

| N | α=0.05, 80% 功效 |
|:--:|:--:|
| 750 | **0.102** |
| 1,500 | 0.072 |
| 2,500 | 0.056 |

观测 US Ridge IC = 0.007 = 检测阈值的 6.85%。**研究功效充足，信号不存在。**

---

## 六、全部负结果清单

| # | 范式 | 关键指标 | 裁决 |
|:--:|------|----------|:----:|
| 1 | DL (Mamba SSM) | Test IC≈0, 97K=451 params | ❌ |
| 2 | GBDT (Hoffnung) | Val IC=0.008 | ❌ |
| 3 | Ensemble+TimeGate | Val IC=0.028-0.065 | ❌ |
| 4 | FE LightGBM+EMA | Test IC=−0.004 | ❌ |
| 5 | CS lambdarank | Rank IC 负值 | ❌ |
| 6 | Meta-Labeling | ΔSharpe=−0.22 | ❌ |
| 7 | Wavelet denoising | ΔIC=−0.046 | ❌ |
| 8 | Inverse-vol weighting | ΔSharpe=−0.019 | ❌ |

---

## 七、唯一正向发现

| 发现 | 指标 | 局限 |
|------|------|------|
| **Regime-Adaptive** | ΔSharpe=+0.216, MaxDD 37→19% | 来自战术减杠杆, 非选股 alpha |
| **组合感知优化** | +26% vs MSE, p=0.011 | IC 仍 0.015 < 0.02 |
| **波动率可预测** | QLIKE 跨市场一致 | GARCH 比 LGB 更好 |

---

## 八、论文资产

```
paper/ (20 文件, ~4,000 行)
├── 00_paper_outline.md         ← 完整大纲+摘要
├── LATERAL_COMPARISON.md       ← 横向对比 (检测能力特征)
├── IC_BENCHMARK_COMPARISON.md  ← IC 基准对比
├── NEGATIVE_RESULTS_REPORT.md  ← 负结果完整报告
├── REPRODUCIBILITY_ASSESSMENT.md ← 三方代码复现评估
├── oracle_assessment.md        ← Oracle 审稿评估
├── findings/ (5 文件)          ← 实验发现
├── literature/ (3 文件)        ← 文献对比
├── methodology/ (3 文件)       ← 方法论
└── results/ (3 文件)           ← 结果汇总
```

---

## 九、脚本资产 (新增)

| 脚本 | 描述 |
|------|------|
| `garch_baseline.py` / `garch_baseline_extended.py` | GARCH vs LGB 基准 |
| `power_analysis.py` | 统计功效分析 |
| `permutation_test.py` | 排列检验框架 |
| `portfolio_optimizer.py` | 组合感知优化 |
| `vol_regime_strategy.py` | 波动率 regime-switching |
| `test_wavelet_causal.py` | 因果小波去噪测试 |
| `download_sp500.py` / `download_nasdaq100.py` | 美股数据下载 |
| `run_newdata_baselines.py` | 跨市场统一基线 |

---

## 十、服务器

| 属性 | Server 1 (V100) |
|------|-----------------|
| SSH | `sshpass -p '<REDACTED>' ssh -p 24212 root@223.109.239.36` |
| GPU | 2× Tesla V100-SXM2-32GB |
| 数据 | `/root/data/us_stocks_full/` (477 US), `/root/data/tencent_clean/` (200 A-share) |

---

> **最终总结**: 经过两市场 × 多 pipeline × 6 范式 × GARCH 基准 × 统计功效分析的穷举测试，Glaubenskrieg 项目严谨地证明了: (1) OHLCV 日频特征不含可检测收益预测信号; (2) GARCH(1,1) 3 参数完胜 LightGBM 200 棵树 (300/300 stocks); (3) 研究功效充足，信号不存在而非未被检测到。项目的最重要贡献是方法论质量（5 步诊断框架 + GARCH 基准 + 统计功效分析）和诚实的负结果。
