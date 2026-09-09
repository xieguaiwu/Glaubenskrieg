# Variant A: Time-Step Exponential Decay Gate — Feasibility Analysis

## 1. In MultiAssetCTM.forward(), where does gbdt_preds enter? What shape?

**Entry point:** `MultiAssetCTM.forward()` parameter `gbdt_preds: Optional[torch.Tensor]` (line 222)

**Shape:** The parameter accepts two shapes:
- `(B, N, T)` — batched, per-asset, per-timestep predictions
- `(B*T, N)` — pre-flattened version

**Flow path (when `use_fused_attention=True`):**
```
forward(x, gbdt_preds)
  │
  ├─ If gbdt_preds.dim() == 3:  reshape (B,N,T) → (B*T,N)
  ├─ cross_attn.set_gbdt_predictions(gbdt_preds_2d)   # cache on module
  └─ single_asset_model.encode(x, cond, cross_attn)     # encode loop
       └─ In RecurrentCTM.encode(), each loop iteration:
            cross_attn(reshaped_hidden)                  # GBDTModulator reads cached preds
            → GBDTModulator.forward(gbdt_preds)          # MLP(pred_i - pred_j) → bias matrix
            → Bias added to attention scores             # before softmax
```

**Critical finding:** `gbdt_preds` is only used when `use_fused_attention=True`. When False, the parameter is accepted but never referenced. The GBDT predictions influence the model **indirectly** through cross-asset attention biases — they never directly participate in the output prediction blending.

**In the current training pipeline** (ensemble_trainer.py), `gbdt_preds` is **never** passed during CTM training (`_train_ctm_window`). CTM is trained first, then GBDT is trained separately on CTM hidden states. Predictions are fused post-hoc via IC-weighted averaging in numpy (`evaluate_ensemble`).

**In P3 trainer** (p3_trainer.py), `gbdt_preds` IS passed during Stage 3 fine-tuning:
```python
output = model(batch_x, gbdt_preds=batch_gbdt)
```
But here `gbdt_preds` only flows through the fused attention modulator — it still doesn't participate in output-level blending.

---

## 2. At what point do we have both ctm_pred and gbdt_pred for blending?

We need to distinguish **three contexts**:

### Context A: Inside `MultiAssetCTM.forward()`
After the output heads (line ~290):
```python
out_reg = self.head_regression(z)   # (B, N, T, output_dim)
out_cls = self.head_classification(z) # (B, N, T, 3)
```
At this point:
- **ctm_reg_pred:** `(B, N, T, output_dim)` — freshly computed
- **gbdt_preds:** Available as function parameter, needs reshaping to `(B, N, T, 1)` for broadcasting
- **No shape mismatch:** `out_reg` is `(B, N, T, 1)` for single regression output → gbdt_preds `(B, N, T)` just needs `.unsqueeze(-1)`
- **Both available:** YES, at line ~290, both tensors are in scope

### Context B: In the training loop (ensemble_trainer.py)
After the CTM forward pass:
```python
ctm_output = ctm_model(val_data)      # (B, T, N*(output_dim+3))
ctm_pred_val = ctm_output[:, -1, :n_assets].reshape(-1).cpu().numpy()  # (B*N,)
```
And separately:
```python
gbdt_pred_val = gbdt_model.predict(X_gbdt_val)  # numpy, computed from hidden states
```
- **ctm_pred:** numpy, post-processed
- **gbdt_pred:** numpy, pre-computed
- **Fused post-hoc** via `evaluate_ensemble()`
- **No gradient flow** back to CTM parameters

### Context C: During P3 Stage 3 fine-tuning
```python
output = model(batch_x, gbdt_preds=batch_gbdt)
```
- gbdt_preds flows into fused attention (inside model)
- loss backpropagates through backbone + fused attention + heads
- But gbdt_preds only affects attention biases, not the output directly

### ⚠️ Key architectural constraint
In the **current walk-forward training pipeline**, CTM and GBDT are trained sequentially:
1. Train CTM (no GBDT access)
2. Train GBDT (no CTM gradient flow)
3. Fuse predictions in numpy

This means **Variant A cannot be trained end-to-end in one pass** unless we either:
- Pre-compute GBDT predictions before CTM training (cheating — uses future model)
- Use a separate gate-tuning phase after both models are trained (two-phase)
- Keep the gate as a post-hoc, validation-optimized component (simplest)

---

## 3. Can we insert a time-step-gated blending layer with <30 lines? Where?

**YES. The gate itself is ~12 lines.** The integration code is another ~15 lines. Total <30 lines.

### Recommended insertion: Inside `MultiAssetCTM.forward()`, between output heads and flattening

This is lines 286–296 in `multiasset_ctm.py`. The exact insertion point:

```python
# Current code (lines ~286-296):
out_reg = self.head_regression(z)       # (B, N, T, output_dim)
out_cls = self.head_classification(z)   # (B, N, T, 3)

# INSERT TIME GATE HERE (if self.use_time_gate and gbdt_preds is not None)

out_reg_b = out_reg.permute(0, 2, 1, 3).reshape(B, T, N * self.output_dim)
out_cls_b = out_cls.permute(0, 2, 1, 3).reshape(B, T, N * 3)
return torch.cat([out_reg_b, out_cls_b], dim=-1)
```

### Precise code sketch for the gate module:

```python
class TimeDecayGate(nn.Module):
    """Variant A: Exponential decay blending of CTM and GBDT predictions.

    Gate weight: w_ctm(t) = σ(α · exp(-β · t) + γ)
    where t = forecast horizon index (0 = 1-step, 1 = 2-step, etc.)
    σ = sigmoid, clamping to [0, 1].

    Trained either:
      - Jointly with CTM (when gbdt_preds available during training)
      - Post-hoc on validation data (freeze CTM, optimize α/β/γ)
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))   # scale
        self.beta = nn.Parameter(torch.tensor(0.3))    # decay rate
        self.gamma = nn.Parameter(torch.tensor(0.0))   # floor weight

    def forward(self, ctm_pred, gbdt_pred, time_indices=None):
        """Blend CTM and GBDT predictions.

        Args:
            ctm_pred: (B, N, T, 1) or (B, N, T, output_dim) CTM regression
            gbdt_pred: (B, N, T) GBDT scalar predictions per asset per step
            time_indices: (T,) optional explicit horizon indices

        Returns:
            blended: (B, N, T, output_dim) same shape as ctm_pred
            weights: (T,) CTM weight at each step (for monitoring)
        """
        T = ctm_pred.shape[2]
        device = ctm_pred.device

        if time_indices is None:
            t = torch.arange(T, device=device, dtype=torch.float32)
        else:
            t = time_indices.to(device=device, dtype=torch.float32)

        # w_ctm(t) = sigmoid(α · exp(-β · t) + γ)
        w_ctm = torch.sigmoid(self.alpha * torch.exp(-self.beta * t) + self.gamma)  # (T,)

        # Broadcast: (T,) → (1, 1, T, 1)
        w_ctm_expanded = w_ctm.view(1, 1, T, 1)

        # Ensure gbdt_pred has compatible shape
        if gbdt_pred.dim() == 3:
            gbdt_expanded = gbdt_pred.unsqueeze(-1)  # (B, N, T) → (B, N, T, 1)
        else:
            raise ValueError(f"Expected gbdt_pred dim=3 (B,N,T), got {gbdt_pred.dim()}")

        blended = w_ctm_expanded * ctm_pred + (1 - w_ctm_expanded) * gbdt_expanded
        return blended, w_ctm
```

### Integration into MultiAssetCTM.forward():

Add these changes (3 locations, ~15 lines total):

**Location 1 — `__init__` (add ~3 lines):**
```python
# After head_classification definition (line ~207):
self.use_time_gate = use_time_gate  # new bool parameter
if use_time_gate:
    self.time_gate = TimeDecayGate()
else:
    self.time_gate = None
```

**Location 2 — `__init__` parameter (add 1 line):**
```python
use_time_gate: bool = False,  # new parameter
```

**Location 3 — `forward()` blending (add ~10 lines), before the permute+reshape:**
```python
# ── Optional: Time-decay gate blending (Variant A) ──
if self.use_time_gate and gbdt_preds is not None:
    # gbdt_preds may be (B,N,T) or (B*T,N) — normalize to (B,N,T)
    if gbdt_preds.dim() == 2:  # (B*T, N)
        gbdt_preds_3d = gbdt_preds.reshape(B, T, N).transpose(1, 2)  # → (B, N, T)
    else:
        gbdt_preds_3d = gbdt_preds  # already (B, N, T)

    # Only blend regression outputs; classification logits are CTM-only
    out_reg, self._gate_weights = self.time_gate(out_reg, gbdt_preds_3d)
```

### ⚠️ Critical nuance: gbdt_preds shape normalization

Inside forward(), gbdt_preds can be `(B, N, T)` or `(B*T, N)`. For the fused attention path, it's reshaped to `(B*T, N)`. For the output gate, we need `(B, N, T)`. The shape normalizer handles both:

```python
def _normalize_gbdt_for_gate(self, gbdt_preds, B, N, T):
    """Ensure gbdt_preds is (B, N, T)."""
    if gbdt_preds.dim() == 3 and gbdt_preds.shape == (B, N, T):
        return gbdt_preds
    if gbdt_preds.dim() == 2:  # (B*T, N)
        return gbdt_preds.reshape(B, T, N).transpose(1, 2)
    raise ValueError(f"Unexpected gbdt_preds shape: {gbdt_preds.shape}")
```

---

## 4. Will LossWrapper and composite_loss still work?

**YES — with no changes required.**

The blended output has the **exact same shape** as before:
```python
# Before blending:
out_reg_b = out_reg.permute(0, 2, 1, 3).reshape(B, T, N * output_dim)  # (B, T, N)
out_cls_b = out_cls.permute(0, 2, 1, 3).reshape(B, T, N * 3)          # (B, T, N*3)
result = torch.cat([out_reg_b, out_cls_b], dim=-1)                     # (B, T, N*(1+3))

# After blending: same shapes, same concatenation order
blended_reg = ...         # (B, N, T, output_dim) → permute+reshape → (B, T, N*output_dim)
result = torch.cat([blended_reg_flat, out_cls_b], dim=-1)  # identical shape
```

**LossWrapper** slices predictions by channel:
- First `N * output_dim` channels = regression → `mse_loss`, `sharpe_loss`, `pinball_loss`
- Remaining `N * 3` channels = classification → `directional_loss`

Since the channel layout is preserved, `composite_loss` operates unchanged. The gradient flows through the blended regression predictions back through:
```
loss → blended_reg → gate(w_ctm * ctm_pred + (1-w_ctm) * gbdt_expanded)
  ├─ ctm_pred path: gate weight → head_regression → cross_attn → backbone
  └─ gbdt_pred path: (1-w_ctm) weight (gbdt_preds are detached, constant)
```

**Important:** `gbdt_preds` must be detached (no gradient). We only want gradients through `ctm_pred` and the gate parameters `(α, β, γ)`, not through the GBDT predictions themselves. This is naturally the case since `gbdt_preds` is a pre-computed tensor passed in from the trainer.

**Classification outputs are NOT blended** — they remain pure CTM. This is by design: the GBDT provides only regression signals; directional classification is a CTM-specific feature.

### Gradient flow verification:

| Parameter | Receives gradient? | Via |
|-----------|-------------------|-----|
| `head_regression` weights | ✅ Yes | `loss → blended → w_ctm * head(z)` |
| `head_classification` weights | ✅ Yes | `loss → out_cls → head(z)` |
| Backbone (Mamba, conv, etc.) | ✅ Yes | `loss → head(z) → z → backbone` |
| `TimeDecayGate.alpha` | ✅ Yes | `loss → blended → w_ctm → α` |
| `TimeDecayGate.beta` | ✅ Yes | `loss → blended → w_ctm → β` |
| `TimeDecayGate.gamma` | ✅ Yes | `loss → blended → w_ctm → γ` |
| gbdt_preds tensor | ❌ No | Detached input, no `requires_grad` |

---

## 5. Minimal invasive change: Module inside MultiAssetCTM vs. post-processing

### Assessment matrix:

| Approach | LOC | Model changes | Trainer changes | Joint training |
|----------|-----|---------------|-----------------|----------------|
| **A: Module inside MultiAssetCTM** | ~25 | ✅ Add gate sub-module + `__init__` param | ✅ Pass `gbdt_preds` during training | ✅ Yes (if gbdt_preds available) |
| **B: Wrapper nn.Module** | ~30 | ✅ New wrapper class | ✅ Replace model class in trainer | ✅ Yes |
| **C: Post-hoc in training loop** | ~20 | ❌ None | ✅ Add blending in eval loop | ❌ Gate not part of model graph |

### Recommendation: Approach A (Module inside MultiAssetCTM)

**Rationale:**
1. **Most cohesive** — gate belongs to the model, not the training loop
2. **Natural parameter lifecycle** — saved/loaded with state_dict, checkpointed with model
3. **Works in both training and inference** — no code duplication between train/eval paths
4. **Backward compatible** — `use_time_gate=False` (default) changes nothing
5. **Minimal trainer changes** — just pass `gbdt_preds` through when available

**However**, there is a **training pipeline constraint**: in the current walk-forward loop, CTM is trained before GBDT, so `gbdt_preds` is unavailable during CTM training. This means **Variant A cannot be trained jointly in a single pass** under the current sequential training paradigm.

### Two viable training strategies:

#### Strategy 1: Post-hoc gate tuning (simplest, ~20 extra lines)
After CTM and GBDT are both trained, freeze both and optimize only the gate:
```python
# Phase 3: Gate-only fine-tuning
ctm_model.use_time_gate = True
ctm_model.time_gate.alpha.requires_grad = True
# Freeze everything else
for p in ctm_model.parameters():
    p.requires_grad_(False)
for p in ctm_model.time_gate.parameters():
    p.requires_grad_(True)

# Train gate on validation set
for epoch in range(n_gate_epochs):
    output = ctm_model(val_x, gbdt_preds=gbdt_predictions)
    loss = criterion(output, val_targets)
    loss.backward()
    gate_optimizer.step()
```

#### Strategy 2: Joint CTM+Gate training with pre-computed GBDT preds
Pre-compute GBDT predictions on training data, then train CTM with gate jointly:
```python
# Pre-compute GBDT preds on training data
gbdt_preds_train = gbdt_model.predict(X_train)  # (N_samples, N_assets, T)

# Train CTM with gate, using frozen GBDT preds as auxiliary input
ctm_model.use_time_gate = True
for epoch in range(n_epochs):
    output = ctm_model(train_x, gbdt_preds=gbdt_preds_train)
    loss = criterion(output, train_targets)
    loss.backward()  # gradients through CTM AND gate params
    optimizer.step()
```

**Strategy 2 is more theoretically sound** because the gate weights are co-optimized with the CTM backbone. Strategy 1 is simpler but may lead to suboptimal coupling (CTM learned without knowing it would be blended).

---

## Concrete Implementation Plan

### Phase 1: New Module (`src/model/time_gate.py`) — NEW FILE

```python
"""Time-decay gate for Variant A: exponential horizon-based NN→GBDT blending.

From research_progressive_ensembling.md Pattern A:
    w_NN(h) = α · exp(-βh) + γ    (sigmoid-clamped to [0,1])
"""

import torch
import torch.nn as nn


class TimeDecayGate(nn.Module):
    """Horizon-dependent exponential decay gate for CTM ↔ GBDT blending.

    At forecast horizon t (0 = 1-step-ahead, increasing = further future),
    CTM weight decays as:  w(t) = σ(α · exp(-β · t) + γ)

    This biases the ensemble toward:
      - CTM for near-term forecasts (higher weight)
      - GBDT for long-term forecasts (lower weight)

    The sigmoid σ ensures w ∈ [0, 1] while maintaining smooth gradients.
    """

    def __init__(self, alpha_init=1.0, beta_init=0.3, gamma_init=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, ctm_pred, gbdt_pred, time_indices=None):
        """Blend predictions with learned time-dependent weights.

        Args:
            ctm_pred: (B, N, T, C) — CTM regression predictions
            gbdt_pred: (B, N, T) — GBDT scalar predictions per asset/step
            time_indices: (T,) optional explicit horizon indices

        Returns:
            blended: (B, N, T, C) — weighted average
            w_ctm: (T,) — CTM weight at each horizon (for monitoring)
        """
        T = ctm_pred.shape[2]
        device = ctm_pred.device

        t = (time_indices if time_indices is not None
             else torch.arange(T, device=device, dtype=torch.float32))

        weight = torch.sigmoid(
            self.alpha * torch.exp(-self.beta * t) + self.gamma
        )  # (T,)
        weight_4d = weight.view(1, 1, T, 1)  # broadcast over (B, N, C)
        gbdt_4d = gbdt_pred.unsqueeze(-1)     # (B, N, T) → (B, N, T, 1)

        return weight_4d * ctm_pred + (1 - weight_4d) * gbdt_4d, weight

    def extra_repr(self):
        return (f"alpha={self.alpha.item():.3f}, beta={self.beta.item():.3f}, "
                f"gamma={self.gamma.item():.3f}")
```

### Phase 2: Patch `MultiAssetCTM.__init__` — add 1 parameter + 3 lines

**File:** `src/model/multiasset_ctm.py`

**Change 1:** Add `use_time_gate` parameter to `__init__` signature (around line 132):
```python
use_time_gate: bool = False,  # NEW: Variant A time-decay gate
```

**Change 2:** After `self.head_classification` (around line 208), add:
```python
# ── Time-decay gate (Variant A) ──
if use_time_gate:
    from .time_gate import TimeDecayGate
    self.time_gate = TimeDecayGate()
else:
    self.time_gate = None
```

### Phase 3: Patch `MultiAssetCTM.forward()` — add blending after output heads

**File:** `src/model/multiasset_ctm.py`

**Change 3:** After `out_reg = self.head_regression(z)` / `out_cls = self.head_classification(z)` (~line 288), insert before the permute+reshape block:

```python
# ── Time-decay gate blending (Variant A) ──
if self.time_gate is not None and gbdt_preds is not None:
    gbdt_3d = self._normalize_gbdt_for_gate(gbdt_preds, B, N, T)
    out_reg, self._last_gate_weights = self.time_gate(out_reg, gbdt_3d)
```

And add the helper at the bottom of the class:

```python
def _normalize_gbdt_for_gate(self, gbdt_preds, B, N, T):
    """Reshape gbdt_preds to (B, N, T) for the time gate."""
    if gbdt_preds.dim() == 3:
        return gbdt_preds  # already (B, N, T)
    if gbdt_preds.dim() == 2:  # (B*T, N)
        return gbdt_preds.reshape(B, T, N).transpose(1, 2)
    raise ValueError(f"Unexpected gbdt_preds shape: {gbdt_preds.shape}")
```

### Phase 4: (Optional) P3 trainer extension for gate fine-tuning

Add Stage 3b to `P3EnsembleTrainer` that fine-tunes the time gate after fused attention fine-tuning:

```python
# In p3_trainer.py, after Stage 3 fused-attention fine-tuning:
if stage3_model.use_time_gate:
    # Phase 3b: Fine-tune only the gate parameters
    stage3_model = self._finetune_time_gate(
        model=stage3_model,
        train_data=train_data, train_targ=train_targ,
        gbdt_preds_train=train_per_asset,
        val_data=val_data, val_targ=val_targ,
        gbdt_preds_val=val_per_asset,
        n_gate_epochs=5, gate_lr=1e-3,
    )
```

### Total new code estimate

| Component | File | Lines |
|-----------|------|-------|
| `TimeDecayGate` class | `src/model/time_gate.py` (new) | ~45 |
| `__init__` param + init | `src/model/multiasset_ctm.py` | ~8 |
| Blending in `forward()` | `src/model/multiasset_ctm.py` | ~6 |
| `_normalize_gbdt_for_gate` helper | `src/model/multiasset_ctm.py` | ~7 |
| Tests | `tests/test_time_gate.py` (new) | ~50 |
| **Total** | | **~116 lines** |

Core gate logic is <25 lines; the rest is plumbing and tests.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Gate collapses to always-CTM (w≈1) | Medium | Monitor `_last_gate_weights`; clip α, β, γ ranges; add weight regularization |
| GBDT predictions unavailable during training | High | Always train CTM first, then tune gate post-hoc (Strategy 1); or pre-compute GBDT preds (Strategy 2) |
| NaNs in gate computation | Low | Sigmoid naturally bounds output; α, β, γ are scalars — minimal risk |
| Gate weights diverge in opposite direction (GBDT always wins) | Medium | Initialize α>0, β>0 to bias toward CTM at short horizons; add gate weight logging |
| Shape mismatch between gbdt_preds variants | Low | `_normalize_gbdt_for_gate` handles both (B,N,T) and (B*T,N); add assertion |

---

## Answers Summary

1. **gbdt_preds enters** as `forward()` parameter with shape `(B, N, T)` or `(B*T, N)`. Currently only used in the fused-attention path (indirectly via cross-attention biases), never for output blending.

2. **Both ctm_pred and gbdt_pred** are simultaneously available in `forward()` at the output-head stage (line ~288), where `out_reg` is `(B, N, T, output_dim)` and `gbdt_preds` can be reshaped to `(B, N, T)` for broadcasting.

3. **Yes, <30 lines of new code** can implement the gate. Insertion point: between output heads and the permute/reshape step in `MultiAssetCTM.forward()`.

4. **LossWrapper and composite_loss work unchanged** because the blending preserves the output shape `(B, T, N*(output_dim+3))` with identical channel layout. Only regression channels are blended; classification channels pass through untouched.

5. **Minimal invasive change:** Add `TimeDecayGate` as a `nn.Module` sub-module inside `MultiAssetCTM`, gated by `use_time_gate=False` (default). This is ~20 lines in the model + ~45 lines for the gate class. **Training constraint:** gbdt_preds must be pre-computed before training with the gate enabled (two-phase approach: train CTM → train GBDT → fine-tune gate).
