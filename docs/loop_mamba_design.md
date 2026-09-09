# Loop Mamba — Recurrent CTM Design (v2)

> Universal Transformer 风格迭代细化架构，应用于 CTMStockModel
> 
> 更新日期：2026-05-23 | 实现完成，已完成 2 轮架构增强

## 核心思路

MambaBlock 的 `forward: (B,T,d_model) → (B,T,d_model)` 保持维度不变，输出可直接再馈入输入门。循环迭代时 Mamba 的 Δ/Β/C 参数由各轮输入重新计算 → 每一轮产生不同的 SSM 动力学。

**关键设计决策**：第 0 轮使用完整 `_encode_core`（含 `input_proj`），后续轮次使用 `_encode_blocks`（共享的 conv→Mamba→双向流水线，跳过 `input_proj`）。这解决了原始设计中的维度不匹配问题：`_encode_core` 返回 `(B,T,model_dim)` 但输入 `x` 是 `(B,T,input_dim)`，`h + x` 残差在维度不等时无法运算。

## 架构图

```
x_raw ─→ [_encode_core] ─→ h₀ ──[迭代0: 含input_proj]──→ x₀
          ↑ input_proj→conv→Mamba×N→[bidir]
                            │
          ╔══ 迭代 1..K-1: _encode_blocks (共享权重, 无input_proj) ═══╗
          ║  ┌─────────────────────────────────────────────────────┐  ║
          ║  │ conv → [Decomp] → Mamba×N → [bidir]                │  ║
          ║  │     ↓                                               │  ║
          ║  │ hᵢ ─── (+) ─→ LayerNorm ─→ Dropout(pᵢ) ─→ xᵢ      │  ║
          ║  └─────────────────────────────────────────────────────┘  ║
          ╚══════════════════════════════════════════════════════════╝
                            │
          x_K─1 ─→ [_encode_blocks] ─→ h_K (最终轮: 无残差/LN/Dropout)
                            │
                        [heads]
                            │
                       output(B,T,4)
```

**渐进式 Dropout**：p 从 `loop_dropout`（默认 0.1）退火至 `loop_dropout/3`（~0.03），早期高 dropout 鼓励探索，后期低 dropout 帮助细化。

## 实现 (RecurrentCTM 子类)

```python
class RecurrentCTM(CTMStockModel):
    def __init__(self, ..., n_loop_iters=5, loop_dropout=0.1):
        super().__init__(..., return_hidden=True, ...)
        if n_loop_iters < 1:
            raise ValueError(...)
        self.n_loop_iters = n_loop_iters
        self.loop_norm = nn.LayerNorm(model_dim)
        self.loop_dropout = nn.Dropout(loop_dropout)

    def _encode_loop(self, x):
        """委托给共享的 _encode_blocks（conv→Mamba→bidir，无 input_proj）"""
        return self._encode_blocks(x)

    def encode(self, x, cond=None):
        """重写基类 encode()：运行完整循环而非单次 _encode_core"""
        x = self.input_proj(x)
        if cond is not None:
            x = x + cond
        for i in range(self.n_loop_iters):
            h = self._encode_blocks(x)
            x = self.loop_dropout(self.loop_norm(h + x)) if i < self.n_loop_iters - 1 else h
        return x

    def forward(self, x, return_hidden=False):
        x = self._encode_core(x)       # 迭代0: 完整编码 (含input_proj)
        for i in range(1, self.n_loop_iters):
            h = self._encode_blocks(x)  # 迭代1+: 共享块 (无input_proj)
            if i < self.n_loop_iters - 1:
                # 渐进式 Dropout
                decay = 1.0 - 0.7 * (i-1) / max(1, self.n_loop_iters-2)
                p = self.loop_dropout.p * decay
                x = F.dropout(self.loop_norm(h + x), p=p, training=self.training)
            else:
                x = h
        out_reg = self.head_regression(x)
        out_cls = self.head_classification(x)
        output = torch.cat([out_reg, out_cls], dim=-1)
        return (output, x) if return_hidden else output
```

## 架构修复 — 2 轮改进循环

### 第 1 轮修复 (代码质量)

| 问题 | 严重程度 | 修复方式 |
|------|----------|----------|
| `_encode_loop` 缺少 conv 后 NaN 检测 | minor | 从 `_encode_core` 继承到 `_encode_blocks` |
| `n_loop_iters < 1` 无校验 | minor | 添加 `raise ValueError` |
| `--n-loop` CLI 优先级反转（配置覆盖CLI） | minor | 改为 CLI 优先（与其他参数一致） |
| `--scale` 缺少 `"loop"` 选项 | minor | 添加到 choices |
| `build_model_params` 缺失 `use_decomp/bidirectional` | minor | 补充到参数构造器中 |
| "Pre-LN" 术语不准确（实际是残差后 LN） | minor | 改为 "residual" |
| 缺少 bidirectional/decomp/return_hidden 测试 | minor | 添加 3 个新测试 |

### 第 2 轮修复 (架构 + 性能)

| 问题/增强 | 类型 | 方式 |
|-----------|------|------|
| `_encode_core` 与 `_encode_loop` 代码重复 | 架构修复 | 提取共享 `_encode_blocks` 到基类 `CTMStockModel` |
| `encode()` 绕过循环（MultiAssetCTM 组合隐患） | 架构修复 | 在 `RecurrentCTM` 重写 `encode()` 含完整循环 |
| 启用 `use_decomp=true` | 性能增强 | DMamba 季节性-趋势分解，零额外参数 |
| `n_loop_iters` 3→5 | 性能增强 | 等效深度 9→15，零额外参数 |
| 渐进式 Dropout | 性能增强 | 阶段探索→细化，p=0.10→0.03 |

## 训练配置

```yaml
model:
  n_loop_iters: 5           # 循环次数（1=禁用循环）
  loop_dropout: 0.1         # 循环间 dropout 基率（渐进退火至 ~0.03）
  use_decomp: true          # DMamba 季节性-趋势分解
  parallel_scan: true       # GPU 推荐；CPU 上 sequential 反而快 ~2x
  n_layers: 3
  model_dim: 128
  state_dim: 32
```

## 预期效果

| n_layers | n_loop | 等效深度 | 参数量 | 计算量 | Decomp |
|----------|--------|----------|--------|--------|--------|
| 3 | 1 (关) | 3 | 90K | 1× | — |
| 3 | 3 | 9 | 90K | 3× | — |
| 3 | **5** | **15** | **90K** | **5×** | **推荐** |
| 6 | 1 (关) | 6 | 716K | 1× | — |
| 6 | 3 | 18 | 716K | 3× | — |

*参数量基于 model_dim=64, state_dim=16。scale_loop.yaml 使用 model_dim=128, state_dim=32 → ~360K 参数。*

## 基准测试

| 规模 | 参数 | 前向 (ms) | 反向 (ms) | 总时间 |
|-------|-----------|-----------|------------|---------|
| small (基线) | 360,068 | 205 | 219 | 424ms |
| loop (n=3) | 360,324 | 543 | 911 | 1,454ms |
| **loop (n=5, 推荐)** | **360,324** | **685** | **812** | **1,497ms** |
| large (基线) | 716,420 | 366 | 542 | 908ms |
| portfolio | 159,319 | 1,123 | 13,727 | 14,850ms |

*CPU (Intel) + B=4 + T=63，使用 parallel_scan=True。loop n=5 较 n=3 仅 +26% 前向时间。*

## 实现状态

### Phase 1 完成 ✅
- [x] 创建 `src/model/loop_ctm.py` 含 `RecurrentCTM` 类
- [x] 添加测试 `tests/test_loop_ctm.py`（13 个测试）
- [x] `scripts/train.py` 添加 `--n-loop` CLI arg + `--scale loop` 选项
- [x] 创建 `configs/scale_loop.yaml`（n_loop_iters=5, use_decomp=true）
- [x] 验证梯度流 + NaN 检查（verify_scales — 全部 PASS）
- [x] benchmark loop vs non-loop

### Phase 2 完成 ✅
- [x] 提取 `_encode_blocks` 消除代码重复
- [x] 重写 `RecurrentCTM.encode()` 修复循环绕过
- [x] 渐进式 Dropout 正则化
- [x] 启用季节性-趋势分解
- [x] CLI 优先级修复（--n-loop 覆盖配置）

### 不推荐
- **Bidirectional**：因果预测中反向 Mamba 引入未来信息泄漏
- **增大 model_dim**：股票数据噪声高，~360K 参数已足够，增大容易过拟合。default.yaml 已将 model_dim 从 128 降至 64, state_dim 从 32 降至 16 以匹配 9 维输入特征（而非之前误标的 20 维）。
- **循环专属参数**：破坏参数效率，升级应优先考虑增加 n_layers

## 规模扩展 (Scaling Roadmap)

### 参数量扩展路径（激进版本）

| 配置 | model_dim | state_dim | n_layers | n_loop | 等效深度 | 估计参数量 | 前向(CPU) |
|------|-----------|-----------|----------|--------|----------|-----------|-----------|
| loop (当前) | 128 | 32 | 3 | 5 | 15 | ~360K | 685ms |
| loop_large (激进) | 256 | 64 | 6 | 8 | 48 | ~2.8M | ~5s (est.) |
| 扩展倍率 | 2× | 2× | 2× | 1.6× | 3.2× | ~7.8× | ~7× |

### 架构扩展（P3 融合预备）

已实现：
- `FusedMultiHeadCrossAttention`：多头交叉注意力 (H=4)，支持 GBDT 外部偏置注入
- `GBDTModulator`：将 GBDT 成对预测差值 → 注意力偏置 MLP
- `RecurrentCTM` 循环内融合：交叉注意力可在每次循环迭代内部执行
- `MultiAssetCTM` 扩展：`use_fused_attention` 标志 + `gbdt_preds` 参数

### 推荐训练设备

| 配置 | 最低 GPU | 推荐 GPU | 批大小 | 期望前向 (GPU) |
|------|---------|----------|--------|-------------|
| loop | RTX 3060 (12GB) | RTX 4090 | 32 | ~15ms |
| loop_large | RTX 4090 (24GB) | A100 80GB | 16 | ~40ms |
