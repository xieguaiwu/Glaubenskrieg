# Design Document: Time-Gated CTM→GBDT Progressive Fusion

> **Author**: Prometheus (architect) | **Date**: 2026-06-05
> **Scope**: Glaubenskrieg — 3 architectural variants for time-dependent CTM/GBDT blending
> **Reference**: MultiAssetCTM + GBDT ensemble, 63-step sequences, multi-asset forecasting

---

## Table of Contents

1. [Problem Statement & Design Philosophy](#1-problem-statement--design-philosophy)
2. [Architecture Baseline: Current MultiAssetCTM + GBDT Integration](#2-architecture-baseline)
3. [Variant A: Time-Step Exponential Decay Gate](#3-variant-a-time-step-exponential-decay-gate)
4. [Variant B: GBDT-Modulated Attention Bias with Time Amplification](#4-variant-b-gbdt-modulated-attention-bias-with-time-amplification)
5. [Variant C: Progressive Curriculum Dropout](#5-variant-c-progressive-curriculum-dropout)
6. [Cross-Variant Comparison](#6-cross-variant-comparison)
7. [Integration Strategy & Implementation Roadmap](#7-integration-strategy--implementation-roadmap)

---

## 1. Problem Statement & Design Philosophy

### 1.1 The Core Hypothesis

Stock prediction at different forecast horizons exhibits fundamentally different noise regimes:

- **Short horizons** (t=0–20): Local temporal patterns dominate. The Mamba SSM's selective scan excels at capturing these short-range dependencies. CTM hidden states are information-rich.

- **Long horizons** (t=40–62): Accumulated SSM state noise, compounded discretization error, and market regime drift degrade CTM signal quality. GBDT's stability (trained on aggregated window-level features) becomes advantageous.

The three variants below operationalize this hypothesis as architectural priors: **CTM leads early, GBDT anchors late, and the transition is learnable (Variants A/B) or enforced as a curriculum (Variant C).**

### 1.2 Design Constraints

| Constraint | Implication |
|---|---|
| Preserve Mamba/CTM architectural innovation | Variants must *augment*, not replace. The selective SSM backbone is the core differentiator. |
| Minimal parameter overhead | <10 for A, <100 for B, 0 for C |
| Backward compatible | All variants degrade gracefully to existing CTM-only behavior when GBDT predictions are unavailable |
| Differentiable end-to-end | Gates (A, B) must propagate gradients through the full CTM backbone |
| No GBDT backpropagation | GBDT predictions are pre-computed fixed inputs — treated as constant leaf tensors |

### 1.3 Sequence Convention

All variants operate on sequences of length `T = 63`. Time index `t ∈ {0, 1, ..., T-1}`. The normalized time coordinate is `τ = t / T ∈ [0, 1)`.

---

## 2. Architecture Baseline

### 2.1 Current Data Flow in `MultiAssetCTM.forward()`

```
Input x (B, N, T, D_in)
    │
    ▼
AssetEmbedding → cond_proj ──────────────────────┐
    │                                               │
    ▼                                               │
Input x reshaped to (B*N, T, D_in) ──→ input_proj ─┤
    │                                               │
    ▼                                               ▼
_encode_blocks (conv → [decomp] → MambaBlock×n_layers → [bidirectional])
    │
    ├── [if use_fused_attention: in-loop FusedMultiHeadCrossAttention]
    │         ▲
    │    GBDTModulator(gbdt_pairwise_diffs) → attn_bias added to QK^T
    │
    ▼
z (B, N, T, model_dim) ──→ head_regression → out_reg (B, N, T, 1)
                         ──→ head_classification → out_cls (B, N, T, 3)

Return: cat(flatten(out_reg), flatten(out_cls)) → (B, T, N*4)
```

### 2.2 Key Integration Points

| Integration Point | Location | What It Does |
|---|---|---|
| **A** — Output heads | `MultiAssetCTM.forward()` L256–267 | Produces `out_reg` and `out_cls` from hidden states |
| **B** — Cross-attention bias | `FusedMultiHeadCrossAttention.forward()` L191–192 | `GBDTModulator(gbdt_diff)` → pairwise additive bias |
| **C** — GBDT prediction injection | `MultiAssetCTM.forward()` L228–235 | `set_gbdt_predictions()` call before backbone encode |
| **D** — Loss computation | `composite_loss()` | Operates on flattened (B, T, N*channels) output |

---

## 3. Variant A: Time-Step Exponential Decay Gate

### 3.1 Concept

Two learnable scalars (α, β) define a sigmoidal time-weight curve that blends CTM and GBDT regression predictions at each time step:

```
gate(t) = σ(α − β · τ)      where τ = t / (T−1) ∈ [0, 1]
fused(t) = gate(t) · ctm_pred(t) + (1 − gate(t)) · gbdt_pred
```

- **α** controls the initial gate value at t=0: `gate(0) = σ(α)`. With α=2.0, gate≈0.88 (CTM dominates early).
- **β** controls decay rate: at t=T−1, `gate(T−1) = σ(α − β)`. With β=4.0, gate≈0.12 (GBDT dominates late).
- Both are **learnable** — the model can discover the optimal handoff schedule from data.
- GBDT predictions are pre-computed once per sequence and cached as `(B, N)` or `(B, N, T)`.

### 3.2 Forward Pass Pseudocode

```
# ──────────────────────────────────────────────
# MultiAssetCTM.forward() — gate insertion point
# ──────────────────────────────────────────────

# ... backbone + cross-attention produces z (B, N, T, model_dim) ...

out_reg = self.head_regression(z)    # (B, N, T, output_dim)
out_cls = self.head_classification(z) # (B, N, T, 3) — NOT gated

# ── Variant A: Time-Step Exponential Decay Gate ──
if self.use_time_gate and gbdt_preds is not None:
    # 1. Normalize gbdt_preds to (B, N, T) shape
    if gbdt_preds.dim() == 2:
        # (B*T, N) → (B, T, N) → (B, N, T)
        gbdt_3d = gbdt_preds.reshape(B, T, N).permute(0, 2, 1)
    elif gbdt_preds.shape[1] == N and gbdt_preds.shape[2] == T:
        gbdt_3d = gbdt_preds  # already (B, N, T)
    elif gbdt_preds.shape[1] == T and gbdt_preds.shape[2] == N:
        gbdt_3d = gbdt_preds.permute(0, 2, 1)  # (B, T, N) → (B, N, T)

    # 2. Compute gate values: σ(α − β · t/(T−1))
    tau = torch.arange(T, device=z.device, dtype=z.dtype) / (T - 1)  # (T,) in [0, 1]
    gate_t = torch.sigmoid(self.time_alpha - self.time_beta * tau)   # (T,)
    gate_t = gate_t.view(1, 1, T, 1)                                  # (1, 1, T, 1)

    # 3. Blend: gate * ctm + (1-gate) * gbdt
    gbdt_expanded = gbdt_3d.unsqueeze(-1)  # (B, N, T, 1)
    out_reg = gate_t * out_reg + (1.0 - gate_t) * gbdt_expanded

    # 4. (Optional) Log mean gate for monitoring
    # self._gate_history.append(gate_t.mean().item())

# ... reshape and return as before ...
```

**Key design decisions:**

1. **Gate applies only to regression output**, not classification. Classification logits (up/flat/down) encode directional semantics specific to CTM — blending with GBDT scalars would break softmax calibration.

2. **Gate is shared across all assets.** Two global parameters (α, β) enforce a consistent temporal handoff schedule. Per-asset gating (2·N params) is a trivial extension but risks overfitting on small datasets.

3. **Gate is purely time-dependent, not content-dependent.** This is intentional for Variant A — it encodes the "CTM early, GBDT late" prior as an architectural inductive bias. Content-dependent gating (conditioned on hidden state) is explored in Variant B.

4. **GBDT predictions are treated as constants.** No gradient flows through the GBDT branch — only the gate parameters (α, β) and the CTM backbone receive gradients.

### 3.3 Parameter Count Impact

| Parameter | Count | Description |
|---|---|---|
| `time_alpha` | 1 | Initial gate offset. Shape: scalar. |
| `time_beta` | 1 | Time decay rate. Shape: scalar. |
| **Total (global)** | **2** | |
| *Optional: per-asset extension* | *2·N* | *Separate α, β per stock* |

At model_dim=64, the total MultiAssetCTM parameter count is ~157K (with fused attention) or ~204K (with RecurrentCTM backbone). Two additional scalars represent a **0.001% increase.**

### 3.4 Training Strategy

**Phase 1: Joint end-to-end training (recommended)**
Both α, β and all CTM parameters are trained jointly in a single phase. The gate is differentiable, so gradients flow from the composite loss through the gate into the backbone.

- **Initialization**: α = 2.0, β = 4.0 (handoff from ~88% CTM at t=0 to ~12% CTM at t=62)
- **Optimizer**: Same AdamW schedule as CTM backbone (lr=3e-4, weight_decay=0.1)
- **No special gate learning rate**: Gate parameters are scalars — their gradient magnitude is naturally small
- **Gradient clipping**: Existing 1.0 clip norm is sufficient

**Phase 2: Gate-only fine-tuning (alternative for P3 pipeline)**
After Stage 1 (CTM warmup) and Stage 2 (GBDT training), freeze the backbone and fine-tune only α, β plus output heads for 5–10 epochs. This is the lower-risk path if gate behavior is unstable.

**Monitoring**: Log `gate_t.mean()` per epoch. Expected range: 0.2–0.8 on average. Values near 0 or 1 indicate mode collapse (gate always favors one model). If this occurs, add L2 regularization on `|α| + |β|` with weight 1e-4.

### 3.5 How It Preserves the CTM/Mamba Architecture

- **Zero Mamba modification.** The Mamba blocks, selective scan, causal conv, and bidirectional processing are untouched.
- **Zero cross-attention modification.** `FusedMultiHeadCrossAttention` and `GBDTModulator` operate independently — Variant A can coexist with or without them.
- **Output heads unchanged.** `head_regression` still produces `(B, N, T, output_dim)` from `z`. The gate is a post-hoc transform applied *after* the heads, not a replacement.
- **Classification head preserved.** Directional prediction remains pure CTM, maintaining the theoretical innovation of selective state-space modeling for temporal pattern recognition.

### 3.6 Expected Effect on Overfitting

| Factor | Effect |
|---|---|
| **2 parameters** | Negligible capacity increase — no overfitting risk from added complexity |
| **Strong inductive bias** | The sigmoid time-decay form encodes `∂gate/∂t ≤ 0` (gate monotonically decreases). This constrains the hypothesis space and acts as regularization. |
| **GBDT as constant input** | No gradient path through GBDT — the gate cannot "overfit to GBDT" by modifying GBDT parameters |
| **Interaction with existing dropout** | The existing `dropout=0.1` in Mamba blocks and output heads provides regularization that propagates through the gate |
| **Risk: gate collapse** | Without monitoring, the gate could converge to constant (α → ∞ or β → 0). Mitigation: log mean gate per epoch; add small L2 penalty on |α|+|β| if collapse is detected. |

### 3.7 Implementation in Existing Codebase

**File: `src/model/multiasset_ctm.py`**

1. **`__init__` additions** (L140–145 region, after `self.dropout` definition):
   ```python
   # New parameter in constructor signature:
   use_time_gate: bool = False,

   # In __init__ body:
   self.use_time_gate = use_time_gate
   if use_time_gate:
       self.time_alpha = nn.Parameter(torch.tensor(2.0))
       self.time_beta = nn.Parameter(torch.tensor(4.0))
   else:
       self.time_alpha = None
       self.time_beta = None
   ```

2. **`forward()` modification** (after L263 `out_cls` line, before L267 reshape):
   - Insert ~25 lines of gating logic (pseudocode above)
   - The `gbdt_preds` parameter is already accepted by `forward()` — no signature change needed
   - Handle both `(B, N, T)` and `(B*T, N)` gbdt_preds formats (already done elsewhere in forward)

3. **No changes to:**
   - `src/model/ctm_model.py` (CTMStockModel untouched)
   - `src/model/loop_ctm.py` (RecurrentCTM untouched)
   - `src/model/mamba_block.py` (MambaBlock untouched)
   - `src/model/fused_attention.py` (FusedMultiHeadCrossAttention untouched)
   - `src/train/p3_trainer.py` (gbdt_preds already passed; format already `(B, N, T)`)

4. **CLI activation** (in `scripts/train.py` or config system):
   ```python
   # Add to model_params dict:
   'use_time_gate': args.time_gate,
   ```

**Total code delta: ~30 lines in multiasset_ctm.py, ~3 lines in train.py.**

### 3.8 Edge Cases & Failure Modes

| Scenario | Handling |
|---|---|
| `gbdt_preds is None` (CTM-only mode) | Gate branch skipped entirely — forward pass is identical to current behavior |
| `T` varies across batches (padding) | Use `torch.arange(T, ...)` per-batch — gate adapts to actual sequence length |
| `gate(t)` becomes constant (α, β not learning) | Log mean gate per 100 steps; if |mean−0.5| > 0.45, warn "gate collapse detected" |
| NaN in gbdt_preds | Already guarded: `nan_to_num` in p3_trainer `_prepare_per_asset_gbdt_preds` |
| Multiple GPUs (DDP) | α, β are normal Parameters — DDP syncs them automatically |

---

## 4. Variant B: GBDT-Modulated Attention Bias with Time Amplification

### 4.1 Concept

Extend the existing `GBDTModulator` in `FusedMultiHeadCrossAttention` with a time-dependent amplification factor. The GBDT-derived pairwise attention bias grows linearly with the time index, making GBDT's cross-asset correlation structure increasingly dominant at later time steps:

```
mod_bias_raw[i,j] = GBDTModulator(gbdt_preds[i] − gbdt_preds[j])  # existing
amplification(τ)  = 1 + γ · τ                                      # new: time-dependent
mod_bias[i,j,τ]   = mod_bias_raw[i,j] × amplification(τ)          # applied per time step
attn_logits       = QK^T/√d_k + adj_bias + mod_bias               # final attention
```

- **γ** (single learnable scalar, default init = 1.0) controls how strongly GBDT's influence grows over time.
- At τ=0 (first step): `amplification = 1.0` — baseline GBDTModulator effect.
- At τ=1 (last step): `amplification = 1 + γ` — up to 2× GBDT influence if γ≈1.
- This preserves the **pairwise structure** of the GBDTModulator — stocks with similar GBDT predictions attend more strongly to each other (and this similarity signal intensifies at longer horizons).

### 4.2 Forward Pass Pseudocode

```
# ──────────────────────────────────────────────────────────────
# FusedMultiHeadCrossAttention.forward() — bias injection point
# ──────────────────────────────────────────────────────────────

# ... compute Q, K, V, attn = QK^T/√d_k + adj_bias ...

if self.use_gbdt_bias and self._gbdt_preds is not None:
    # 1. Compute raw pairwise bias (existing logic)
    mod_bias = self.gbdt_modulator(self._gbdt_preds)  # (B*T, N, N) or (B, N, N)

    # 2. Compute time amplification: 1 + γ · τ
    #    Need to know T. Either:
    #    (a) Store T from input shape during forward's needs_reshape branch
    #    (b) Infer from BT and B (if B known)
    #    For batched input (B, N, T, D): T = x.shape[2], BT = B*T
    tau = torch.arange(self._T, device=x.device, dtype=x.dtype) / (self._T - 1)  # (T,)
    amp = 1.0 + self.time_gamma * tau  # (T,)

    # 3. Expand amplification to match mod_bias shape
    #    mod_bias is (B*T, N, N) — need to multiply each of the BT entries
    #    by the corresponding time-step's amp value
    amp_expanded = amp.repeat_interleave(self._B, dim=0)  # (B*T,)
    amp_expanded = amp_expanded.view(-1, 1, 1)            # (B*T, 1, 1)
    mod_bias = mod_bias * amp_expanded

    # 4. Add to attention logits (existing logic)
    attn = attn + mod_bias

# ... softmax, dropout, matmul with V ...
```

**Key design decisions:**

1. **Time amplification is multiplicative, not additive.** This preserves the sign of the GBDTModulator bias — stocks with negative pairwise bias (predicted to move in opposite directions) remain negatively correlated, but the magnitude grows over time.

2. **γ is initialized to 1.0** (moderate growth), not 0.0 (no effect). This gives the model a non-trivial starting point. If γ→0 during training, the variant gracefully degrades to the existing GBDTModulator behavior.

3. **Requires awareness of T** inside the attention module. The current `FusedMultiHeadCrossAttention` flattens to `(B*T, N, D)` and loses sequence structure. We need to preserve T information. Two approaches:
   - **Approach (a)**: Store T as an instance variable during `__init__` or `forward()`
   - **Approach (b)**: Accept T as a parameter
   - **Recommended (a)**: Store `self._T` during forward when input is 4D, use it for bias computation

4. **Works with both 3D and 4D inputs.** For 4D `(B, N, T, D)`: B and T are known. For 3D `(B*T, N, D)`: need B and T externally, or infer from `mod_bias.shape[0]` and `self._B` (if stored via `set_gbdt_predictions`).

### 4.3 Parameter Count Impact

| Parameter | Count | Description |
|---|---|---|
| `time_gamma` | 1 | Time amplification scalar |
| **Total** | **1** | (GBDTModulator already exists — ~4,600 params for hidden_dim=64) |

The existing `GBDTModulator` parameter count for hidden_dim=64 is:
```
Linear(1→64):     64 + 64  = 128
LayerNorm(64):    64 + 64  = 128
Linear(64→64):  4096 + 64  = 4160
LayerNorm(64):    64 + 64  = 128
Linear(64→1):     64 + 1   = 65
                         Total = 4,673
```

Adding γ brings it to 4,674 — a **0.02% increase** over the existing modulator.

### 4.4 Training Strategy

**Integration with P3 Trainer (Stage 3):**

The P3 trainer already freezes the Mamba backbone and fine-tunes `FusedMultiHeadCrossAttention` (including `GBDTModulator`) plus output heads. Variant B adds γ to the fine-tuned parameter set — no change to the training flow.

- **Phase**: Stage 3 fine-tuning only (modulator_epochs=10, modulator_lr=1e-4)
- **γ initialization**: 1.0
- **γ learning rate**: Same as other cross-attention parameters (1e-4)
- **Monitoring**: Log `amp.mean()` across time steps. Expected: 1.0 → ~1.5 at τ=1 if γ≈0.5.
- **Risk**: γ growing too large (e.g., >10) could overwhelm QK^T attention scores. Mitigation: clamp γ to [0, 5] via `torch.clamp` after each optimizer step, or add L2 penalty on γ.

**Alternative: joint training (no freeze):**
If training from scratch with `use_fused_attention=True` and `use_time_amp=True`, all parameters including γ train jointly. This is feasible but harder to tune — the modulator and backbone may compete for signal.

### 4.5 How It Preserves the CTM/Mamba Architecture

- **Mamba backbone untouched.** The selective SSM processing is unchanged.
- **Cross-attention mechanism preserved.** QKV projections, multi-head splitting, and softmax attention operate identically. The only change is the *magnitude* of the existing bias term.
- **Additive bias structure maintained.** `mod_bias × amp(τ)` is still added to `attn_logits` — the softmax sees the same `(B*T, n_heads, N, N)` shape.
- **GBDTModulator MLP unchanged.** The pairwise difference computation and MLP architecture are identical to the current implementation.

### 4.6 Expected Effect on Overfitting

| Factor | Effect |
|---|---|
| **1 parameter** | Essentially zero capacity impact |
| **Linear time amplification** | Simple functional form — low risk of fitting noise in the time dimension |
| **γ initialized to 1.0** | Provides a meaningful prior (GBDT matters more later) without over-constraining |
| **Multiplicative on existing bias** | If GBDTModulator output is near-zero (no useful signal), amplification does nothing — the model can ignore GBDT when uninformative |
| **Risk: attention saturation** | Large γ could produce extreme softmax temperatures (very peaked attention). Mitigation: clamp γ to [0, 5] to keep amplification ≤ 6×. |

### 4.7 Implementation in Existing Codebase

**File: `src/model/fused_attention.py`**

1. **`FusedMultiHeadCrossAttention.__init__` additions:**
   ```python
   # New parameter:
   use_time_amp: bool = False,

   # In __init__ body:
   self.use_time_amp = use_time_amp
   if use_time_amp:
       self.time_gamma = nn.Parameter(torch.tensor(1.0))
   else:
       self.time_gamma = None

   # Store B, T for time amplification (populated during forward)
   self._B: Optional[int] = None
   self._T: Optional[int] = None
   ```

2. **`forward()` modification** (in the GBDT bias section, L185–212):
   - After computing `mod_bias` (L192), insert time amplification logic
   - Capture B and T from input shape when `needs_reshape=True`:
     ```python
     if needs_reshape:
         B, N_in, T, D = x.shape
         self._B = B
         self._T = T
     ```
   - Compute amp and multiply:
     ```python
     if self.use_time_amp and self.time_gamma is not None:
         T = self._T
         tau = torch.arange(T, device=x.device, dtype=x.dtype) / max(T - 1, 1)
         amp = 1.0 + torch.clamp(self.time_gamma, 0.0, 5.0) * tau  # (T,)
         B = self._B if self._B is not None else (mod_bias.shape[0] // T)
         amp_exp = amp.repeat_interleave(B, dim=0).view(-1, 1, 1)  # (B*T, 1, 1)
         mod_bias = mod_bias * amp_exp
     ```

3. **`MultiAssetCTM.__init__` passthrough:**
   ```python
   # Add parameter to pass through to FusedMultiHeadCrossAttention:
   time_amp: bool = False,
   # In cross_attn construction:
   self.cross_attn = FusedMultiHeadCrossAttention(
       ...,
       use_time_amp=time_amp,
   )
   ```

**No changes to:**
- `src/model/multiasset_ctm.py` forward pass (attention already called)
- `src/train/p3_trainer.py` (gradient flow unchanged)
- Any loss functions

**Total code delta: ~25 lines in fused_attention.py, ~5 lines in multiasset_ctm.py.**

### 4.8 Edge Cases & Failure Modes

| Scenario | Handling |
|---|---|
| 3D input `(B*T, N, D)` without stored T | Fall back to `T=1` (amp = 1.0 for all steps) — no amplification, equivalent to existing behavior |
| `γ → 0` during training | Amplification → 1.0 — degrades gracefully to existing GBDTModulator |
| `γ` very large | Clamp to [0, 5] to prevent attention logit saturation |
| Batch size varies across calls | `self._B` and `self._T` updated each forward — no stale state |
| DDP synchronization | γ is a normal Parameter — synced automatically |

---

## 5. Variant C: Progressive Curriculum Dropout

### 5.1 Concept

A training-only intervention: during each epoch, CTM predictions at later time steps are randomly zeroed out with probability proportional to both the epoch fraction and the time step index. This creates a curriculum that progressively forces the model to handle the absence of CTM predictions at long horizons — effectively teaching it to rely on GBDT as a fallback.

```
p_drop(t, epoch) = (epoch / n_epochs) × (t / T)

For each (batch, asset, time_step):
    If rand() < p_drop(t, epoch):
        out_reg[b, a, t] = 0  # kill CTM prediction
```

At inference time, **no dropout is applied** — the model produces its normal CTM predictions. The curriculum has taught the model's hidden representations to be robust to long-horizon CTM degradation, and the GBDT can serve as a post-hoc ensemble partner.

**Important nuance:** Dropping CTM predictions to zero during training is different from replacing them with GBDT. Zeroing forces the model to:
1. Not over-rely on CTM predictions at long horizons (they might be zero)
2. Front-load useful information into earlier time steps (where dropout probability is lower)
3. Produce hidden states that are "aware" of the uncertainty at long horizons

### 5.2 Training Loop Pseudocode

```
# ───────────────────────────────────────────────────────
# Inside train_epoch / _finetune_stage3 / similar function
# ───────────────────────────────────────────────────────

def train_epoch_with_curriculum(
    model, dataloader, optimizer, epoch, n_epochs, T=63
):
    model.train()
    epoch_frac = epoch / max(n_epochs - 1, 1)  # ∈ [0, 1]

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()

        # Standard forward pass (no gate, no special args needed)
        output = model(batch_x)  # (B, T, N*4)

        # ── Variant C: Progressive Curriculum Dropout ──
        if epoch_frac > 0:  # skip epoch 0 (warmup)
            B, T_total, C = output.shape
            N = model.n_assets
            output_dim = model.output_dim

            # Extract regression slice: (B, T, N*output_dim)
            num_reg = N * output_dim
            reg_output = output[..., :num_reg].reshape(B, T_total, N, output_dim)

            # Compute per-time-step dropout probability
            tau = torch.arange(T_total, device=output.device).float() / max(T_total - 1, 1)
            p_drop = epoch_frac * tau  # (T,) — higher dropout at later steps

            # Generate dropout mask: keep CTM with prob (1 - p_drop)
            mask = torch.bernoulli(
                (1.0 - p_drop).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                .expand(B, T_total, N, output_dim)
            )  # (B, T, N, output_dim) — 1=keep, 0=drop

            # Apply mask: zero out dropped predictions
            reg_output = reg_output * mask

            # Reassemble output (regression gated, classification unchanged)
            reg_flat = reg_output.reshape(B, T_total, num_reg)
            output = torch.cat([reg_flat, output[..., num_reg:]], dim=-1)

        loss = composite_loss(output, batch_y, ...)
        loss.backward()
        optimizer.step()
```

**Key design decisions:**

1. **Dropout applies only to regression output**, not classification. Consistent with Variant A.

2. **Dropout is applied after model forward, not inside the model.** This means the model's internal computation is unchanged — gradients flow through the dropped predictions back to the model, which learns which time steps are "unreliable" and adjusts hidden states accordingly.

3. **No dropout in epoch 0 (warmup).** The model first learns to predict all time steps normally, then the curriculum gradually introduces the dropout pattern.

4. **p_drop(t, epoch) ∈ [0, 1]**. At epoch 0: no dropout anywhere. At the final epoch: 
   - t=0: p=0 (CTM always kept)  
   - t=T/2: p≈0.5 (50% dropout)  
   - t=T-1: p≈1 (CTM almost always dropped)

5. **Alternative: replace with GBDT instead of zeroing.** For a stronger signal, replace dropped CTM predictions with GBDT predictions:
   ```python
   # Instead of reg_output = reg_output * mask:
   reg_output = mask * reg_output + (1 - mask) * gbdt_preds
   ```
   This is a configurable option — if `gbdt_preds` are available, we blend; otherwise, zero.

### 5.3 Parameter Count Impact

**Zero parameters.** Variant C is a training procedure modification with no architectural footprint. The saved model checkpoint is identical to a non-curriculum-trained model.

### 5.4 Training Strategy

| Phase | Duration | p_drop behavior |
|---|---|---|
| Warmup (epoch 0) | 1 epoch | p_drop = 0 everywhere (learn basic predictions) |
| Early curriculum (epochs 1–25%) | 25% of training | p_drop ramps up linearly: at epoch_frac=0.25, p_drop_max≈0.25 |
| Mid curriculum (epochs 25%–75%) | 50% of training | p_drop continues ramping: p_drop_max reaches ~0.75 |
| Late curriculum (epochs 75%–100%) | 25% of training | Full dropout: p_drop_max≈1.0 at final epoch |

**Learning rate interaction**: The dropout introduces noise — consider a slightly higher learning rate (1.2×) to compensate for reduced effective sample size at later time steps.

**Validation**: During validation, do NOT apply dropout. The model sees a clean forward pass. The goal is to evaluate whether the curriculum-trained model produces better hidden representations that, when combined with GBDT post-hoc, yield superior ensemble performance.

### 5.5 How It Preserves the CTM/Mamba Architecture

- **Complete preservation.** No architectural modification of any kind.
- The Mamba blocks, selective scan, cross-attention, output heads — all operate identically to the current implementation.
- The curriculum is implemented as a **data augmentation** applied to model outputs before loss computation, not as a model component.
- The saved model is interchangeable with a non-curriculum-trained model — same architecture, same checkpoint format.

### 5.6 Expected Effect on Overfitting

| Factor | Effect |
|---|---|
| **Regularization through noise** | Dropout on outputs acts as a form of structured noise injection — forces the model to be robust to prediction degradation at specific time steps |
| **Curriculum prevents early collapse** | Gradual introduction avoids the model "giving up" on long horizons entirely |
| **No capacity increase** | Zero new parameters — no overfitting from added complexity |
| **Risk: CTM learns to predict near-zero at late steps** | If p_drop→1 consistently at late steps, the model may learn to output small values there (since zero is the dropout value). Mitigation: cap p_drop_max at 0.8 so some CTM signal is always preserved |
| **Risk: gradient starvation at late steps** | When predictions are zeroed, no gradient flows through those time steps for the regression head. Mitigation: keep classification head unaffected (provides gradient signal) |

### 5.7 Implementation in Existing Codebase

**Two integration approaches:**

**Approach 1: In the trainer directly (simplest, recommended)**

Modify the training step in the trainer being used (e.g., `WalkForwardTrainerAdvanced`, `P3EnsembleTrainer._finetune_stage3`, or the main training loop in `scripts/train.py`):

```python
# In the train function or loop, after model.forward() but before loss:
if curriculum_dropout and epoch > 0:
    # Apply dropout mask as described in pseudocode above
    output = apply_curriculum_dropout(output, epoch, n_epochs, model.n_assets, model.output_dim)
```

A standalone helper function (~30 lines) in `src/train/` or `scripts/train.py`:

```python
def apply_curriculum_dropout(
    output: torch.Tensor,
    epoch: int,
    n_epochs: int,
    n_assets: int,
    output_dim: int,
    T: int = 63,
    max_dropout: float = 0.8,
    gbdt_preds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply progressive curriculum dropout to CTM regression predictions.

    With probability p = (epoch/n_epochs) * (t/T), zero out CTM predictions
    at time step t. Optionally replace with GBDT predictions.
    """
    epoch_frac = epoch / max(n_epochs - 1, 1)
    B, T_total, C = output.shape
    num_reg = n_assets * output_dim

    tau = torch.arange(T_total, device=output.device).float() / max(T_total - 1, 1)
    p_drop = torch.clamp(epoch_frac * tau, max=max_dropout)  # (T,)

    reg_output = output[..., :num_reg].reshape(B, T_total, n_assets, output_dim)
    mask = torch.bernoulli(
        (1.0 - p_drop).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        .expand(B, T_total, n_assets, output_dim)
    )

    if gbdt_preds is not None:
        # Blend: keep CTM where mask=1, use GBDT where mask=0
        gbdt_expanded = gbdt_preds.unsqueeze(-1)  # shape to (B, N, T, output_dim)
        reg_output = mask * reg_output + (1 - mask) * gbdt_expanded
    else:
        # Pure dropout: zero where mask=0
        reg_output = reg_output * mask

    reg_flat = reg_output.reshape(B, T_total, num_reg)
    return torch.cat([reg_flat, output[..., num_reg:]], dim=-1)
```

**Approach 2: As a nn.Module wrapper (more modular)**

Create a `CurriculumDropout` module that wraps the loss computation — less invasive to training code but requires model output to be unwrapped/wrapped.

**Recommendation: Approach 1** — the trainer already has epoch information, and the dropout is inherently a training-loop concern.

**Files to modify:**
| File | Change | Lines |
|---|---|---|
| `src/train/curriculum.py` (new) | `apply_curriculum_dropout()` helper | ~35 |
| `src/train/p3_trainer.py` | Call helper after model forward in `_finetune_stage3` | ~5 |
| `src/train/advanced_trainer.py` | Call helper in `train_epoch_advanced` | ~5 |
| **Total** | | **~45** |

### 5.8 Edge Cases & Failure Modes

| Scenario | Handling |
|---|---|
| Batch with T < 63 (variable-length sequences) | `p_drop` computed from actual `T_total` per batch — curriculum adapts |
| Validation (no dropout) | Pass `epoch=-1` or separate train/val code paths — dropout only during `.train()` |
| Max dropout too aggressive | Cap `max_dropout=0.8` — at least 20% CTM signal preserved even at full curriculum |
| No GBDT available | Fall back to pure zero-dropout (no replacement) |
| Model learns to output near-zero at late steps | Monitor mean absolute prediction per time step; if late-step predictions → 0, reduce curriculum intensity |

---

## 6. Cross-Variant Comparison

### 6.1 Summary Matrix

| Dimension | Variant A (Time Gate) | Variant B (Amp Attn) | Variant C (Curriculum) |
|---|---|---|---|
| **Mechanism** | Learned sigmoid time-gate at output | Learned time-amplified attention bias | Training-only dropout schedule |
| **Parameters** | 2 (α, β) | 1 (γ) + existing modulator | 0 |
| **Architecture change** | Yes — output blending | Yes — attention bias scaling | No — training procedure only |
| **Inference change** | Yes — gate active at inference | Yes — amplification active at inference | No — standard forward pass |
| **Backbone modification** | None | None | None |
| **Cross-attention modification** | None | Yes (bias magnitude) | None |
| **Output heads modification** | Yes (blended) | None | None (training only) |
| **Training complexity** | Low (joint training) | Medium (P3 Stage 3) | Low (helper function) |
| **GBDT at inference required** | Yes | Yes | No |
| **Degrades without GBDT** | Gracefully (gate skipped) | Gracefully (amp=1.0) | N/A (no GBDT dependency) |
| **Expected LOC delta** | ~30 | ~30 | ~45 |

### 6.2 Synergies & Interactions

| Combination | Synergy | Notes |
|---|---|---|
| **A + B** | Gate blends outputs; amp shapes attention | GBDT influences both the hidden representation (via attn) and the final prediction (via gate). Strongest integration. |
| **A + C** | Gate handles inference; curriculum shapes training | Curriculum teaches the CTM to produce gate-friendly hidden states. |
| **B + C** | Amp grows during training; curriculum forces robustness | Complementary — different mechanisms, combined effect on learning dynamics. |
| **A + B + C** | Full progressive pipeline | Highest complexity; recommended only after A and B are individually validated |

### 6.3 Which Variant to Prototype First?

**Recommended order: A → C → B**

1. **Variant A first**: Simplest fully-differentiable gate. Clear hypothesis (CTM early, GBDT late). Easy to debug (monitor α, β, gate_t). Validates the core assumption that time-dependent fusion improves performance over static ensembles.

2. **Variant C second**: Zero risk (no architecture change). Validates whether a curriculum that forces CTM to cede long-horizon predictions produces better representations. If C works, it provides evidence that the time-dependent prior is valid, motivating A and B.

3. **Variant B third**: Most invasive (modifies attention). Only worth pursuing if A shows that time-dependent GBDT integration is beneficial, and if cross-asset attention patterns are a meaningful source of predictive signal.

---

## 7. Integration Strategy & Implementation Roadmap

### 7.1 File-Level Changes Summary

```
Glaubenskrieg/
├── src/
│   ├── model/
│   │   ├── fused_attention.py        ← Variant B: +~25 lines
│   │   └── multiasset_ctm.py         ← Variant A: +~30 lines; B passthrough: +~5
│   └── train/
│       ├── curriculum.py             ← Variant C: NEW FILE ~35 lines
│       ├── advanced_trainer.py       ← Variant C: +~5 lines (integration)
│       └── p3_trainer.py             ← Variant C: +~5 lines (integration)
└── scripts/
    └── train.py                      ← CLI flags for all variants: +~10 lines
```

### 7.2 Backward Compatibility Guarantees

All variants are **opt-in** via constructor flags (`use_time_gate=False`, `use_time_amp=False`, `curriculum_dropout=False`). When all flags are False, the codebase behaves identically to the current implementation. Existing tests pass without modification.

### 7.3 Testing Strategy

| Test Type | Variant A | Variant B | Variant C |
|---|---|---|---|
| **Unit: gate range** | `test_gate_in_0_1`, `test_gate_monotonic` | `test_amp_positive`, `test_amp_at_zero` | `test_dropout_prob_range` |
| **Unit: shape preservation** | `test_output_shape_unchanged` | `test_attn_shape_unchanged` | `test_output_shape_unchanged` |
| **Unit: no-GBDT fallback** | `test_forward_without_gbdt` | `test_forward_without_gbdt` | N/A |
| **Unit: gradient flow** | `test_gate_params_get_grad` | `test_gamma_gets_grad` | N/A |
| **Integration: P3 pipeline** | `test_p3_with_time_gate` | `test_p3_with_time_amp` | `test_p3_with_curriculum` |
| **Integration: combined** | — | — | `test_combined_variants` (A+B) |

### 7.4 Expected Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Variant A implementation | 1–2 hours | `multiasset_ctm.py` modifications + unit tests |
| Variant A training run | 2–4 hours | Baseline vs. time-gate comparison on 3 seeds |
| Variant C implementation | 1 hour | `curriculum.py` helper + trainer integration |
| Variant C training run | 2–4 hours | Curriculum vs. baseline comparison |
| Variant B implementation | 1–2 hours | `fused_attention.py` modifications + unit tests |
| Variant B training run | 2–4 hours | With P3 pipeline, compare against A-only |
| Analysis & report | 2 hours | Cross-variant ablation, Diebold-Mariano tests |

---

## Appendix A: Design Rationale — Why Not Alternatives?

### A.1 Why not softmax over 2 models?

```
w_ctm(t), w_gbdt(t) = softmax(Linear(z[t]))
fused(t) = w_ctm(t) * ctm_pred(t) + w_gbdt(t) * gbdt_pred
```

**Rejected because**: Softmax enforces `w_ctm + w_gbdt = 1` with both always non-zero. A sigmoid gate allows the model to suppress both (gate≈0.5 → both contribute equally, not suppressed) or trust one fully (gate≈0 or ≈1). More importantly, softmax with learned parameters is content-dependent — the gate value depends on the hidden state, not just time. This makes the handoff schedule harder to interpret and more prone to overfitting on specific hidden state patterns.

### A.2 Why not per-layer gating?

Applying a gate after each MambaBlock (not just after all blocks):

```
for each MambaBlock output h_i:
    gate_i(t) = σ(α_i − β_i · τ)
    h_i' = gate_i(t) * h_i + (1 − gate_i(t)) * proj(gbdt_pred)
```

**Rejected for Variant A** (kept it simple with 2 params) but noted as a **natural extension**. Per-layer gating with 2·n_layers parameters (6 for n_layers=3) would allow different Mamba layers to integrate GBDT at different rates — early layers might use GBDT for structural priors while later layers use it for prediction refinement. This is worth exploring if Variant A shows promise.

### A.3 Why not learned step function?

```
gate(t) = σ(α · (t − t_switch) / temperature)
```

Where `t_switch` is a learnable hard-switch threshold. **Rejected**: The sigmoid already provides a smooth transition. A hard switch creates a non-smooth loss landscape (gradient near-zero except at the switch point) and may be harder to optimize. The exponential decay form `σ(α − βτ)` is mathematically equivalent to `σ(β · (α/β − τ))` which already encodes a soft-switch at `τ = α/β`.

---

## Appendix B: Initialization Guidelines

| Variant | Parameter | Init Value | Rationale |
|---|---|---|---|
| A | α | 2.0 | σ(2.0) ≈ 0.88 — CTM ~88% weight at t=0 |
| A | β | 4.0 | σ(2.0−4.0) ≈ σ(−2.0) ≈ 0.12 — CTM ~12% weight at t=T−1 |
| B | γ | 1.0 | amp(1.0)=2.0 — GBDT attention bias doubles by final step |
| C | — | — | No parameters to initialize |

---

*End of design document.*
