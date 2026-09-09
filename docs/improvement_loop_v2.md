---
name: iterative-improvement-loop-v2
version: 2.0.0
description: >-
  第二代迭代改进循环参考文档。主规范见 improvement-loop.md。
supersedes: improvement-loop.md (v1)
---

# 迭代改进循环 v2 — 参考文档

> **主规范文件**: [`improvement-loop.md`](../improvement-loop.md)
> 本文件是 docs/ 中的参考副本。所有更新请直接编辑 `improvement-loop.md`。

本文件保存方向 B-D 的核心改进总结，供快速参考。

---

## 核心改进 (Directions B-D)

### B: Agent 差异化

| 角色 | Agent 类型 | 成本 |
|------|-----------|------|
| 架构审查 (Pass A) | `oracle` | 高 |
| 一致性审查 (Pass B) | `explore` (×3并行) | 低 |
| 接口契约审查 (Pass C) | `explore` (×2并行) | 低 |
| 测试覆盖审查 (Pass D) | `explore` | 低 |
| 代码修复 | `deep` 或 `unspecified-high` | 中 |
| 回归扫描 | `unspecified-low` | 低 |

### C: 4 道并行审查 Pass

```
Pass A (Oracle):   架构审查 → 模块边界、接口稳定性、可扩展性
Pass B (explore):  一致性审查 → 跨文件模式匹配、sibling模块对比
Pass C (explore):  接口契约审查 → caller/callee签名、配置字段消费
Pass D (explore):  测试覆盖审查 → 修改→测试映射、回归风险评估
```

### D: 修复后回归扫描

```
Fix Phase → Regression Scan → Test Suite → Next Review
```

---

## 缺陷根因分类标签 (10种)

| 标签 | 描述 |
|------|------|
| `cross-file-inconsistency` | 跨文件模式不一致 |
| `interface-contract` | 模块接口签名/行为不匹配 |
| `silent-fallback` | 配置参数被静默忽略 |
| `dry-violation` | 代码重复/未复用已有工具 |
| `error-handling-gap` | 错误处理不完整/不一致 |
| `numerical-stability` | 数值稳定性问题 |
| `vectorization-opportunity` | 可向量化但用循环 |
| `dead-code` | 未使用代码/参数/配置 |
| `doc-clarity` | 命名误导/注释缺失 |
| `test-contract-drift` | 接口改动但测试未同步 |

---

## 相关文档索引

| 文件 | 内容 |
|------|------|
| [`improvement-loop.md`](../improvement-loop.md) | **主规范** — 完整执行流程、所有阶段细节 |
| [`defect_patterns.md`](defect_patterns.md) | 缺陷模式知识库 — 每个模式的出现次数、checklist、风险文件 |
| [`ai_handover_guide.md`](ai_handover_guide.md) | AI交接指南 — 新session入口文档 |
| [`improvement_loop_report.md`](improvement_loop_report.md) | 历史修复清单 — 26个已修复问题的完整记录 |
