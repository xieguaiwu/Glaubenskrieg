---
name: iterative-improvement-loop
version: 2.0.0
description: 第二代迭代改进循环——差异化Agent审查×4、专项审查Pass、修复后回归扫描、缺陷模式知识库
supersedes: improvement-loop.md (v1)
upgrade-reason: |
  v1 缺陷:
  (1) Review Agent无历史记忆, 同类模式缺陷反复遗漏
  (2) Fix Agent引入新缺陷无专项检查
  (3) 跨文件系统性问题(占60%)比局部问题更难捕获
  (4) 审查质量高度依赖prompt, 无结构化保障
triggers:
  - "改进循环"
  - "迭代优化"
  - "持续重构"
  - "improvement loop"
  - "refactoring loop"
  - "修复循环"
  - "迭代改进"
inputs:
  - name: target_path
    description: 要改进的目标代码库路径（目录或文件）
    required: true
  - name: max_iterations
    description: 最大循环次数，防止无限循环
    required: false
    default: 5
  - name: review_focus
    description: 审查重点覆盖
    required: false
    default: "architecture, consistency, contracts, coverage, quality"
  - name: improvement_goal
    description: 改进目标描述，为空则由审查自主发现
    required: false
    default: ""
  - name: defect_patterns_file
    description: 已知缺陷模式清单路径(跨session持久化). 新session启动时加载此文件注入review prompt.
    required: false
    default: "docs/defect_patterns.md"
tools:
  - read_file
  - list_dir
  - grep_files
  - file_search
  - write_file
  - edit_file
  - apply_patch
  - exec_shell
  - task_shell_start
  - task_shell_wait
  - agent_spawn
  - agent_result
  - agent_wait
  - agent_cancel
  - agent_send_input
  - checklist_write
  - checklist_update
  - update_plan
  - diagnostics
---

# 迭代改进循环 v2 (Differentiated Improvement Loop)

对目标代码库执行自动化的闭循环：**修改 → 4道并行审查Pass(A-D) → 分类(Triage) → 修复 + 提交标记 → 回归扫描 → 下一轮**，直到审查判断无问题后退出。

适用于：
- 代码重构后的质量审查
- 功能实现后的系统性检查
- 大型重构的分轮验证
- 新功能后的可扩展性和边界情况检查

---

## 核心设计

### v1→v2 核心改进 (Directions B-D)

**B: Agent 差异化 —— 审查与修复分离**

| 角色 | Agent 类型 | 特性 | 成本 |
|------|-----------|------|------|
| **架构审查** (Pass A) | `oracle` | 只读、高IQ、最严格 | 高 |
| **一致性审查** (Pass B) | `explore` (×3并行) | 上下文grep、跨文件模式匹配 | 低 |
| **接口契约审查** (Pass C) | `explore` (×2并行) | 调用者/被调用者签名比对 | 低 |
| **测试覆盖审查** (Pass D) | `explore` | 修复→测试映射检查 | 低 |
| **代码修复** | `deep` 或 `unspecified-high` | 目标导向执行 | 中 |
| **回归扫描** | `unspecified-low` | 专注副作用检测 | 低 |

**关键约束**: Review和Fix使用不同session_id，避免confirmation bias。

**C: 专项审查 Pass —— 取代单一泛化审查**

Review 阶段不再是单一 agent 做"全面审查"，而是 4 个并行 pass：

```
┌─ Iteration N Review Phase ─────────────────────────────┐
│                                                         │
│  Pass A: Oracle (架构审查)    并行    Pass B-D: explore │
│   模块边界、接口稳定性        ───▶    一致性/契约/覆盖    │
│   可扩展性、整体设计                   跨文件模式匹配     │
│   加载defect_patterns.md                                    │
│                                                         │
└──────────────────────┬──────────────────────────────────┘
                       ▼
             汇总审查结果 → 全Pass无问题→完成
                          有问题→进入Fix阶段
```

**D: 修复后回归扫描**

Fix 完成后不直接进入下一轮审查，先执行回归扫描：
```
Fix Phase → Regression Scan → Test Suite → Next Review
```

回归扫描检测：测试契约漂移、同类模式同步、副作用检查。

---

## 执行流程

```
┌──────────────────────────────────────────────────────────────┐
│  阶段 0: 初始化                                               │
│  - 加载缺陷模式清单 (defect_patterns_file)                     │
│  - 读 target_path 了解项目结构                                 │
│  - 记录当前 git commit hash，用于后续对比                       │
│  - 获取 max_iterations、review_focus、improvement_goal        │
│  - 设置迭代计数器 iteration = 0                                │
│  - 通过 checklist_write 创建可视化进度追踪                      │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段 1: 审查 (Review) — 4 Pass 并行执行                      │
│                                                              │
│  ┌────────────────────┐    ┌─────────────────────────────┐   │
│  │  Pass A: Oracle    │    │  Pass B-D: explore agents  │   │
│  │  (架构审查)        │    │  (一致性/契约/覆盖)          │   │
│  │  同步, 串行        │    │  并行, background=true      │   │
│  └────────┬───────────┘    └──────────┬──────────────────┘   │
│           │                          │                        │
│           └──────────┬───────────────┘                        │
│                      ▼                                        │
│             汇总审查结果                                        │
│             - 全Pass无问题 → REVIEW_PASSED, 跳阶段5            │
│             - 有问题 → 汇总问题列表, 进入阶段2                  │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段 2: 分批 (Triage)                                        │
│  - 按严重程度排序: critical > major > minor                   │
│  - 分批策略:                                                 │
│    1-5 个问题 → 全部修复                                     │
│    6-15 个问题 → 先修 critical + major                       │
│    16+ 个问题 → 每轮修最多 10 个                              │
│  - 记录每个问题的根因类型标签(见缺陷模式分类)                   │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段 3: 修复 (Fix)                                           │
│                                                              │
│  - 启动 deep 或 unspecified-high agent                       │
│  - 每个 fix agent 负责一个 batch                              │
│  - 流程: 读文件→确认上下文→edit_file/apply_patch→验证语法→提交 │
│  - 不修改与问题无关的代码                                      │
│  - 修复中将根因类型记录到 commit message                       │
│    格式: "iter-improve-loop: iteration N fix [pattern:标签]"    │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段 4: 回归扫描 (Regression Scan)                            │
│                                                              │
│  启动 regression-scan agent (unspecified-low):                │
│                                                              │
│  1. 测试契约检查:                                              │
│     被修改文件的对应测试是否有效? 需要更新参数/期望值吗?         │
│     修改了签名 → grep 所有调用者确认已同步                      │
│                                                              │
│  2. 同类模式检查:                                              │
│     本次修改的模式是否存在于其他文件? 需要同步吗?               │
│     例: 改mamba_block.py → 检查mamba_parallel.py/loop_ctm.py  │
│                                                              │
│  3. 副作用检查:                                                │
│     新的 import 是否引入循环依赖?                               │
│     删除的 export 是否仍有外部调用者?                           │
│                                                              │
│  4. 测试套件:                                                  │
│     python -m pytest tests/ -v --tb=short                    │
│                                                              │
│  - 如果回归扫描发现问题 → 回到阶段3修复回归问题                 │
│  - 如果通过 → iteration += 1, 回到阶段1                       │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
                      iteration >= max_iterations?
                             │
                      ┌──────┴──────┐
                      ▼              ▼
                   是 / REVIEW_PASSED  否 → 回到阶段1
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段 5: 完成                                                 │
│  - 输出最终报告                                              │
│  - 更新缺陷模式清单 (defect_patterns_file)                    │
│  - 列出剩余问题(如有)                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 各阶段详细说明

### 阶段 0：初始化

1. 读取 `target_path` 了解项目结构
2. 读取 `defect_patterns_file`（默认 `docs/defect_patterns.md`）：
   - 如果文件存在，将内容注入后续 review agent 的 prompt
   - 如果文件不存在，创建骨架文件
3. 记录当前 git commit hash（用于后续对比）
4. 获取 `max_iterations`、`review_focus`、`improvement_goal`
5. 设置 `iteration = 0`
6. 通过 `checklist_write` 创建可视化进度追踪

### 阶段 1：审查 — 4 道并行 Pass

#### Pass A：架构审查 (Oracle)

启动 Oracle agent（只读，不修改代码）。提示结构：

```
你是一个架构审查 agent。审查 {target_path} 的代码。

已知缺陷模式(来自历史迭代):
{加载 defect_patterns.md 的内容}

审查范围:
1. 模块划分: 职责是否清晰? 有无循环依赖?
2. 接口稳定性: public API 是否向后兼容?
3. 可扩展性: 添加新功能需要改多少文件?
4. 整体设计: 是否有更好的架构方案?

请特别关注与已知缺陷模式相关的新问题。

输出格式:
- 如果无问题: "REVIEW_PASSED: true"
- 如果有问题: 每个问题一行:
  [SEVERITY: critical/major/minor]
  [FILE: relative_path:line_number]
  [DESCRIPTION]
  [SUGGESTION]
  [PATTERN_TAG: 根因分类标签]
```

#### Pass B：一致性审查 (explore × 3, 并行, background=true)

```
搜索 {target_path} 下所有类似模式:
1. 找到同类模块/函数的所有变体
2. 比较实现是否一致
3. 重点检查 defect_patterns.md 中 cross-file-inconsistency 清单

检查方向:
- NaN/错误处理是否在所有变体中一致?
- 相同功能是否使用了相同的工具函数?
- 配置键命名是否统一?
```

#### Pass C：接口契约审查 (explore × 2, 并行, background=true)

```
验证 {target_path} 中所有跨模块调用:
1. 对每个 .py 提取 public 函数签名
2. 用 grep 找到所有调用者
3. 对比 caller 传参 vs callee 签名是否匹配

特别检查:
- loss_config 字段是否被所有消费者正确读取
- trainer 返回类型是否被调用者正确处理
- 配置 key 变更是否已更新所有 YAML
- dataclass/配置类的所有字段是否被生产代码使用
```

#### Pass D：测试覆盖审查 (explore × 1, background=true)

```
映射当前 iteration 的修改到测试:
1. 获取 git diff 中修改的文件列表
2. 对每个修改的文件, 找到对应测试文件
3. 检查测试是否覆盖了修改逻辑
4. 输出缺失测试覆盖的修改点

已知覆盖热力图(加载 docs/defect_patterns.md 中的 coverage 部分):
- fused_attention.py: 0% → 修改必须加测试
- features.py: 57% → 3个函数未测试
- _walk_forward_utils.py: 0% → 无边界条件测试
- metrics.py: 0% → sharpen_ratio_torch 无单元测试
```

#### 汇总审查结果

所有 Pass 完成后，汇总结果：
- **全 Pass 无问题** → `REVIEW_PASSED: true`, 跳阶段 5
- **有问题** → 合并所有 Pass 的问题列表去重，进入阶段 2

#### 特殊情形处理

| 情形 | 处理 |
|------|------|
| Pass A (Oracle) 超时或失败 | 重新启动一次，仍失败则标记结果不可靠并询问用户 |
| Pass B-D (explore) 超时 | 单独重启，不影响已完成的 Pass |
| 问题过多 (>20 条) | 仅选取 critical + major 级别处理，其余留下一轮 |
| `iteration >= max_iterations` | 强制退出循环 |
| 第二轮起 | 做**增量审查**：只检查被修改的文件 + 相关接口，减少 agent 调用成本 |

### 阶段 2：分批 (Triage)

按严重程度排序并分批：

| 严重程度 | 定义 | 优先级 |
|----------|------|--------|
| **critical** | 直接 bug、空指针、安全漏洞、数据丢失风险 | 优先处理 |
| **major** | 架构缺陷、接口设计问题、复杂度过高、异常处理缺失 | 优先处理 |
| **minor** | 命名不当、注释缺失、代码风格、轻微冗余 | 可延后 |

分批策略：
- 1-5 个问题 → 全部修复
- 6-15 个问题 → 先修 critical + major
- 16+ 个问题 → 每轮修最多 10 个

**记录每个问题的根因类型标签**（见[缺陷模式分类](#缺陷根因分类标签)章节），用于 commit message 和 defect_patterns_file。

### 阶段 3：修复 (Fix)

对每个批次，启动一个修改 subagent：

```
你是一个代码修改 subagent。请根据以下问题列表修复 {target_path} 的代码。

问题列表：
[{问题1 — 带 PATTERN_TAG}]
[{问题2 — 带 PATTERN_TAG}]

要求：
1. 每个修改请使用 read_file 先确认代码上下文，再用 edit_file 修改
2. 修改前 git 快照：git add -A && git commit -m "wip: before fix batch"
3. 修改后用 git commit 记录变更，格式：
   git commit -m "iter-improve-loop: iteration {N} fix [{pattern-tag}] {description}"
4. 不要修改与问题无关的代码
5. 如果一个问题需要多个文件联动修改，请一次性完成
6. 使用 agent 类型：deep 或 unspecified-high
```

**验证修复：**
- 修改的文件是否存在语法错误（运行 lsp_diagnostics）
- 抽查 1-2 个修改点确认变更真正落地
- 如有语法错误，在同一 subagent 内修复

### 阶段 4：回归扫描 (Regression Scan)

修复完成后启动回归扫描 agent（`unspecified-low` 类型，不同 session_id）：

```
你是一个回归扫描 agent。检测以下问题:

1. 测试契约检查:
   - 修改了 src/model/losses.py → 检查 tests/test_losses.py 是否需要更新
   - 修改了 src/train/* → 检查 tests/test_trainer.py 是否需要更新
   - 修改了函数签名 → 检查所有调用者是否已同步

2. 同类模式检查:
   - 如果修改了 mamba_block.py → 检查 mamba_parallel.py 是否有相同模式需同步
   - 如果修改了一个 trainer → 检查其他 trainer 是否有相同模式需同步

3. 副作用检查:
   - 新的 import 是否引入循环依赖?
   - 删除的 export 是否仍有外部调用者?

4. 运行测试:
   python -m pytest tests/ -v --tb=short
```

- 如果回归扫描发现问题 → 回到阶段 3 修复回归问题
- 如果通过 → `iteration += 1`，回到阶段 1

### 阶段 5：完成

循环结束后输出：

```
=== 迭代改进循环完成 ===
循环次数: {n}
目标路径: {target_path}

=== 修改摘要 ===
{列出主要变更}

=== 审查结论 ===
{审查结论}

=== 变更文件 ===
- {路径} (修改数)
```

**更新缺陷模式文件：** 将本轮新发现的**根因类型**追加到 `defect_patterns_file` 中。

---

## 缺陷模式知识库 (Defect Pattern Knowledge Base)

### 目的

每次新 session 清空 context 后，review agent 没有上次迭代的记忆。缺陷模式知识库跨 session 持久化已知缺陷模式，补偿"fresh context"带来的记忆损失。

### 生命周期

```
新session开始 → 加载defect_patterns.md → 注入review prompt
                                               ↓
                                          Review和Fix → 标记根因类型到commit message
                                               ↓
session结束 → 将新root cause追加到defect_patterns.md
```

### 缺陷根因分类标签

| 标签 | 描述 | 示例 |
|------|------|------|
| `cross-file-inconsistency` | 跨文件模式不一致 | NaN处理在不同Mamba变体中不同 |
| `interface-contract` | 模块接口签名/行为不匹配 | loss_bridge不读lambda_sharpe |
| `silent-fallback` | 配置参数被静默忽略 | typo的key用默认值无警告 |
| `dry-violation` | 代码重复/未复用已有工具 | 两个trainer各有walk-forward实现 |
| `error-handling-gap` | 错误处理不完整/不一致 | 有的模块raise有的warn |
| `numerical-stability` | 数值稳定性问题 | NaN传播、零方差除、极端值 |
| `vectorization-opportunity` | 可向量化但用循环 | rolling_normalize的Python循环 |
| `dead-code` | 未使用代码/参数/配置 | 未使用的 output_heads、GBDTFeatureConfig |
| `doc-clarity` | 命名误导/注释缺失 | lr_warmup_steps实际是epochs |
| `test-contract-drift` | 接口改动但测试未同步 | 参数变必传后测试挂 |

### 知识库文件格式

```markdown
# Defect Pattern Knowledge Base
> Auto-generated from improvement loop iterations

## Pattern: cross-file-inconsistency
- Occurrences: 9+
- Occurrence: IT-1: NaN handling across 3 Mamba variants
- Occurrence: IT-1: Three separate Sharpe implementations
- Occurrence: IT-1: lr_warmup_steps renamed across 10 files
- Checklist: When modifying a shared utility, check ALL sibling files
- Files at risk: src/model/mamba_*.py, src/train/*trainer*.py

## Pattern: interface-contract
- Occurrences: 4
- Checklist: After changing function signature, grep ALL callers
- Files at risk: src/train/loss_bridge.py → callers
```

---

## Git 提交规范

每次修复的 commit message 必须包含缺陷类型标签：

```
iter-improve-loop: iteration {N} fix [{pattern-label}] {short description}
```

示例:
```
iter-improve-loop: iteration 1 fix [cross-file-inconsistency] NaN handling across Mamba variants
iter-improve-loop: iteration 2 fix [interface-contract] forward is_multi_asset to validate()
```

好处：
- 未来的 review agent 可通过 `git log --oneline --grep='pattern:'` 查询历史缺陷分布
- 缺陷模式文件可直接从 git log 生成统计

---

## 反模式与经验教训 (Anti-Patterns)

### 不要做 ❌

1. **审查和修复用同一个 agent session** → 修复 agent 会产生确认偏误
2. **修复后直接进入下一轮审查** → 应该先做回归扫描，否则 fix side effects 被遗漏
3. **泛化 prompt 做审查** → "review the code" 效果远差于结构化多 pass
4. **每个 iteration 都做全量审查** → 第二轮起应做增量审查（只检查修改的文件 + 相关接口）
5. **把所有问题塞给一个 fix agent** → 16+ 问题应分批，每批不超过 10 个

### 一定要做 ✅

1. **每个新 session 加载 defect_patterns.md** → 让 review agent 知道历史缺陷模式
2. **Oracle 做最贵的审查** → 架构和系统性缺陷必须用最高质量模型
3. **explore 做便宜的并行扫描** → 一致性/契约检查是 grep 工作
4. **Fix 后跑回归测试** → 非可选项，是必须的验证步骤
5. **记录根因类型** → 为未来的 session 积累知识

---

## Subagent 管理

1. **超时处理**：审查 subagent 设置合理超时（建议 120 秒），超时则重新启动一次
2. **上下文隔离**：每个 subagent 使用独立 session，不混用 review 和 fix 的 session_id
3. **结果验证**：修改 subagent 返回后，抽查 1-2 个修改点确认变更落地
4. **串行约束**：审查和修复是串行的——下一轮必须等上一轮审查结果出来后再决定
5. **Pass B-D 并行**：一致性/契约/覆盖审查之间互不依赖，可同时启动

---

## Git 安全网

每次修改前执行 git 快照 (`git add -A`)，每次修改后提交 (`git commit`)。确保所有变更都可追溯、可回滚。

连续 3 次修复失败时：
1. **STOP** 所有编辑
2. **REVERT** 到上次已知正常的 git 状态 (`git checkout .`)
3. **DOCUMENT** 尝试了什么、失败了什么
4. **CONSULT** Oracle
5. 如果 Oracle 也无法解决 → **ASK 用户**

---

## 防止无限循环

1. **`max_iterations` 硬上限**：达到后强制退出
2. **问题数量递减检查**：连续两轮问题不减反增 → 退出并报告
3. **相同问题重复出现**：同一问题连续两轮都出现且未被有效修复 → 标记为"顽固问题"，跳出循环
4. **空循环保护**：没有做任何修改却通过审查 → 算通过，不循环

---

## 适用边界

| 场景 | 适用性 |
|------|--------|
| 代码重构 | ✅ 核心场景 |
| 功能实现后质量检查 | ✅ 高度适用 |
| 修复 bug 后验证 | ✅ 适用 |
| 新增 API 端点 | ✅ 可用 |
| 纯文档修改 | ⚠️ 只检查结构和链接 |
| 二进制/配置文件 | ❌ 不适用 |
| 首次代码编写 | ❌ 不是从零生成，而是改进已有代码 |

---

## 与用户互动

1. **首次启动时**：输出当前代码状态的摘要（文件数、行数、粗略评估）
2. **发现问题时**：列出问题并简要说明每个问题的风险
3. **强制退出时**：解释为什么退出（达到上限 / 问题不减反增 / 顽固问题），给后续建议
4. **用户可随时中断**：一旦用户介入给出新指令，停止当前循环按新指令执行

---

## 输出格式

### 每轮迭代摘要

```
━━━ 迭代 {n}/{max_iterations} ━━━
发现 {m} 个问题：
  [critical] file.py:42 — ...
  [major]    mod.py:15 — ...
  [minor]    file.py:88 — ...
已修复 {k} 个问题，进入下一轮审查
```

### 循环结束报告

```
=== 迭代改进循环完成 ===
循环次数: {n}
目标路径: {target_path}

=== 修改摘要 ===
- (带 pattern 标签的变更列表)

=== 审查结论 ===
REVIEW_PASSED: true/false — {说明}

=== 变更文件 ===
- {path} ({count} 处修改)

=== 剩余问题 ===
{如有}
```

---

## 与 v1 的兼容性

v2 是 v1 的超集：
- ✅ 保留 v1 的 Iteration 循环架构（Review → Fix → Review）
- ✅ 保留 v1 的问题分批策略
- ✅ 保留 v1 的安全网（max_iterations、问题递减、顽固问题检测）
- ✅ 保留 v1 的输出格式模板
- ✅ 保留 v1 的 subagent 管理规则
- 🆕 新增：差异化 agent 路由 (Oracle/explore/deep)
- 🆕 新增：四道专项审查 Pass (A-D)
- 🆕 新增：修复后回归扫描 (Regression Scan)
- 🆕 新增：缺陷模式持久化知识库 (defect_patterns.md)
- 🆕 新增：Git 提交规范 (pattern标签)
- 🆕 新增：反模式与经验教训

可以直接替代 v1。所有 v1 触发词仍然有效。
