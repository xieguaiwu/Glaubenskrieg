# Defect Pattern Knowledge Base

> Auto-generated from improvement loop iterations.
> Purpose: Compensate for "fresh context per session" by preserving defect pattern memory across sessions.
> Load this file as part of review prompts in the next session.
> Last updated: 2026-05-25 (IT-7: overfitting mitigations, config fixes, 8 pre-existing tests resolved, docs updated)

---

## Pattern: cross-file-inconsistency

**Occurrences:** 15+
- **New (IT-7)**: `MockBroker.simulate_fill()` didn't update `_positions` — caused `rebalance_to_targets` to wipe account state on `update_from_broker()` call. Broke all 4 rebalance tests.
- **New (IT-7)**: `normalize_features()` std_safe correction (`np.where(std_arr < eps)`) failed for NaN because NaN < eps is False — fixed with `~np.isfinite(std_arr) | (std_arr < eps)`
- **New (IT-7)**: Test `test_defaults` expected `include_ctm_features=True` but IT-6 changed source default to `False` — test not updated
- **New (IT-6)**: `multiasset_ctm.py` had ZERO NaN handling for cross-attention/FF/heads — all other model files have warn+nan_to_num+counter
- **New (IT-6)**: `loop_ctm.encode()` had ZERO NaN checks while `forward()` had 4 — encode/forward DRY violation fixed by adding NaN checks to encode()
- **New (IT-5)**: `mamba_parallel.py` missing NaN check on gate `z` branch — `mamba_block.py` has it at step 1 check
- **New (IT-5)**: `mamba_parallel.py` has only 2 NaN detection sites vs 4 in `mamba_block.py` — gate branch and conv output after `_forward_steps_1_3` not checked
- **New (IT-5)**: `loop_ctm.py` had unprotected `B = B_N // N` integer division (fixed with assert)
- **New (IT-5)**: `p3_trainer.py` duplicates ~80 lines from `ensemble_trainer.py` `train_walk_forward` loop — fixes to base loop don't propagate to P3
- **New (IT-5)**: `fused_attention.py` used `assert BT % B_gbdt == 0` (lost in `-O` mode) — now `raise ValueError`
- **New (IT-5)**: `gbdt_features.py` `np.nanstd(..., ddof=0)` uses population std while all loss/metric code uses `ddof=1` sample std

**Occurrences:** 9+
- IT-0 (pre-session): NaN handling used `RuntimeError` in `mamba_parallel.py`, `warnings.warn` in `mamba_block.py` and `ctm_model.py` → inconsistent
- IT-1: Three separate Sharpe implementations (losses.py, trainer.py, ensemble_trainer.py) with different ddof
- IT-1: `lr_warmup_steps` → `lr_warmup_epochs` renamed across 10 files
- IT-2: `import math` unused in 4 trainer files
- IT-3: Direct imports replaced `try/except` in multiasset_ctm.py
- **New (post-IT-4)**: `loop_ctm.py` NaN detection warns but doesn't `nan_to_num` — all other Mamba variants do both (FIXED in IT-6: both encode() and forward() now have full warn+nan_to_num+counter)
- **New (post-IT-4)**: `ensemble.py:37-38` uses `np.std()` (population ddof=0), while `sharpe_ratio_torch` uses sample ddof=1
- **New (IT-4 explore)**: `advanced_trainer.validate_advanced()` delegates to `validate()` without passing `is_multi_asset` → triggers heuristic fallback
- **New (IT-4 explore)**: `EnsembleWalkForwardTrainer.train_walk_forward` missing `log_gradients` param that `WalkForwardTrainerAdvanced.train_walk_forward` has

**Checklist:**
- [ ] When modifying a shared utility/pattern, check ALL files that implement the same pattern
- [ ] When fixing NaN behavior in one Mamba variant, check ALL Mamba variants
- [ ] Stats functions: check ddof/correction consistency across numpy and torch variants
- [ ] Config params: check all consumers after adding/renaming a config key
- [ ] Identically-named params (e.g., `train_walk_forward`) should have identical signatures across trainers
- [ ] Delegation calls should forward all relevant parameters explicitly, not rely on heuristic fallback

**Files at risk:**
- `src/model/mamba_block.py`, `mamba_parallel.py`, `loop_ctm.py`, `ctm_model.py` (sibling SSM variants)
- `src/train/trainer.py`, `advanced_trainer.py`, `ensemble_trainer.py` (sibling trainers)
- `src/data/features.py` + `dataset.py` + `gbdt_features.py` (data pipeline siblings)

---

## Pattern: interface-contract

**Occurrences:** 4
- IT-1: `loss_bridge.ctm_composite_loss_for_gbdt()` ignores `lambda_sharpe` silently — config field defined but not consumed
- IT-1: `composite_loss()` auto-deduced `output_dim = features - 3`, assuming 3 classes — breaks for n-class != 3
- IT-1: `trainer.validate()` fragile heuristic `n_channels % 4 == 0` to detect multi-asset
- IT-3: `ensemble_trainer` returned dicts with no schema; callers needed `.get(key, default)` guards
- **New (post-IT-4)**: `advanced_trainer.LossWrapper` (lines 41-43) detects `n_assets` via `hasattr`, passes to `validate()` but `is_multi_asset` not explicitly forwarded → falls through to heuristic in trainer.py

**Checklist:**
- [ ] After changing a function signature, grep for ALL callers and verify params match
- [ ] Config fields: check they're actually READ by all consumers, not just defined
- [ ] Return type changes: verify all callers handle the new type
- [ ] `hasattr` heuristic → replace with explicit parameter where possible

**Files at risk:**
- `src/train/loss_bridge.py` → callers in `ensemble_trainer.py`, `scripts/train.py`
- `src/train/trainer.py` → `advanced_trainer.py` (delegation chain)
- `src/model/losses.py` → callers in all trainers

---

## Pattern: silent-fallback

**Occurrences:** 7+
- IT-1: `lambda_sharpe > 0` silently ignored in GBDT bridge
- IT-1: Config typos in YAML → default used with no warning (mitigated by `_warn_unknown_keys` in IT-3)
- IT-2: `BT // B_gbdt` integer division silently truncated
- IT-3: `hessians.abs()` silently masked negative values
- IT-4: `advanced_trainer.py` `.get("grad_norm", 0.0)` masks missing metrics when `log_gradients=False`
- **New (IT-5)**: `scripts/train.py:lr_warmup_epochs` fallback `5` vs ALL configs use `200` — if config key removed, warmup drops 40×
- **New (IT-5)**: `scripts/infer.py:seq_len` fallback `60` vs config default `63` — silent input length mismatch
- **New (IT-5)**: `src/model/losses.py:LossConfig.skip_l2_reg` — field exists but never settable from config/CLI (always defaults to False)
- **New (IT-5)**: `src/model/fused_attention.py` — `assert BT % B_gbdt == 0` stripped in `-O` mode (fixed: now `raise ValueError`)

**Checklist:**
- [ ] Search for `.get(key, default)` — is the fallback truly harmless, or masking a bug?
- [ ] Search for `.abs().clamp(...)` patterns — is information being discarded?
- [ ] Integer division `/` — could `//` silently truncate?
- [ ] Template file `configs/default.yaml` creates the **contract** for defaults — any `.get(key, fallback)` should match the YAML value
- [ ] `assert` in production code — use `raise ValueError` instead (asserts stripped in `-O`)

**Files at risk:**
- `scripts/train.py` (many `.get()` calls in config loading)
- `src/train/loss_bridge.py` (Hessian abs, Sharpe ignore)
- `src/model/fused_attention.py` (batch alignment division)

---

## Pattern: dry-violation

**Occurrences:** 3
- IT-1: `ensemble_trainer` re-implemented walk-forward window loop instead of using shared `walk_forward_windows()`
- IT-1: Model param construction duplicated 3x in `scripts/train.py` → extracted `_build_model_params()`
- IT-1: Sharpe ratio implemented 3x → extracted `sharpe_ratio_torch()`

**Checklist:**
- [ ] Before writing a new training loop / data transform / metrics function, check if a shared utility exists
- [ ] Look for similar function signatures in sibling files
- [ ] When fixing a bug in a utility function, check if the same bug exists in duplicated implementations

**Files at risk:**
- `src/train/*trainer*.py` (each trainer has the same structure)
- `scripts/train.py` + `scripts/verify_scales.py` (shared entry point patterns)

---

## Pattern: error-handling-gap

**Occurrences:** 3
- IT-1: `RuntimeError` vs `warnings.warn` for NaN — inconsistent across Mamba variants (fixed)
- IT-3: `loss_bridge.py` warns on negative Hessian but continues — should at minimum log
- **New (post-IT-4)**: `loop_ctm.py` NaN detection warns but never fixes with `nan_to_num` — incomplete error handling
- **New (post-IT-4)**: `loss_bridge.py:71` warns for `lambda_sharpe > 0` — should this be a `raise` instead?

**Checklist:**
- [ ] For each `warnings.warn()` call: is there a corresponding recovery action, or does execution continue in a degraded state?
- [ ] Is the warning actionable? Can the user fix the issue based on the warning message?
- [ ] Should this be `raise` instead of `warn`?

**Files at risk:**
- `src/model/mamba_*.py`, `loop_ctm.py`, `ctm_model.py` (NaN handling)
- `src/train/loss_bridge.py` (warning-only error handling)
- `src/model/ensemble.py` (IC computation edge cases)

---

## Pattern: numerical-stability

**Occurrences:** 6+
- IT-1: NaN propagation in Mamba SSM (values underflowing beyond machine epsilon)
- IT-1: NaN in Bollinger Bands / volatility (early timesteps with no history)
- IT-1: Zero-variance Sharpe ratio (constant predictions → NaN)
- IT-2: `realized_volatility` single-element window `correction=1` → NaN
- IT-3: `rolling_normalize` cumsum division edge cases
- **New (IT-6)**: `loop_ctm.encode()` had 0 NaN checks (4 NaN propagation risk points: after _encode_blocks, after cross-attn, after residual, after input_proj+cond) — now added full warn+nan_to_num+counter
- **New (IT-6)**: `multiasset_ctm.forward()` had 0 NaN checks for cross-attn, cross-ff, and output heads — now added warn+nan_to_num+counter
- **New (IT-6)**: `gbdt_features.py` `GBDTFeatureConfig.include_ctm_features` default `True` vs `build_gbdt_feature_matrix(include_ctm_features=False)` — aligned both to `False`
- **New (IT-6)**: `gbdt_features.py` `normalize_features()` used `np.nanstd(ddof=0)` — changed to `ddof=1` to match all other code
- **New (IT-7)**: `configs/default.yaml` used `model_dim: 128` with `input_dim: 9` and only ~1000 training sequences — 222:1 param/seq ratio → guaranteed overfitting. Reduced to model_dim=64, state_dim=16, added param count logging to train.py.
- **New (IT-5)**: `src/data/gbdt_features.py` used `np.nanstd(ddof=0)` (population) while all other loss/metric code uses `ddof=1` — fixed to `ddof=1`
- **New (IT-5)**: `src/train/loss_bridge.py:91` — `hessians.abs()` silently converts negative Hessian diagonals, masking non-convex loss composition bugs

**Checklist:**
- [ ] All divisions have epsilon guards (`value + 1e-8` or `torch.where(condition, ...)`)
- [ ] Variance of small samples (<2) uses `correction=0` not `correction=1`
- [ ] NaN detection has a recovery action (not just warn)

**Files at risk:**
- All files in `src/model/` (tensor computations)
- `src/data/features.py` (statistical indicators)
- `src/data/dataset.py` (normalization)
- `src/model/ensemble.py` (IC computation)

---

## Pattern: test-contract-drift

**Occurrences:** 7 (all in IT-1 regression round)
- NaN detection tests expected `RuntimeError`, got warning → 2 tests failed
- `composite_loss` test didn't pass `num_regression` → 1 test failed
- Sharpe tests failed due to dim alignment + zero-variance → 4 tests failed

**Checklist:**
- [ ] When changing function signature, update ALL callers (source + test)
- [ ] When changing error handling (raise → warn or vice versa), update test expectations
- [ ] Run full test suite after every fix batch, not just the changed file's tests

**Files at risk:**
- Any `tests/test_*.py` that tests functions modified in the source

---

## Pattern: dead-code

**Occurrences:** 7+
- **New (IT-6)**: `multiasset_ctm.py` dead `is None` guards for `RecurrentCTM` and `FusedMultiHeadCrossAttention` — both are direct imports at module top (REMOVED)
- **New (IT-6)**: `p3_trainer.py` unreachable duplicate `return results` at line 254 (REMOVED)
- **New (IT-6)**: `ensemble_trainer.py` `self.log_gradients = log_gradients` stored in __init__ but never read (REMOVED)
- **New (IT-6)**: `loss_bridge.py` `model_parameters` parameter accepted but never used — `_model_parameters` renamed to suppress lint (function callers always pass None)
- IT-1: `GBDTFeatureConfig` dataclass defined but never instantiated (CORRECTED: actually instantiated in IT-3, see below)
- IT-1: Output heads created but never used (MultiAssetCTM + fused_attention)
- IT-2: `import math` in 4 files with no `math.` usage
- **New (IT-5)**: `scripts/train.py` — `date_col = cfg["data"].get("date_column", None)` assigned but NEVER used
- **New (IT-5)**: `src/train/p3_trainer.py` — `_build_loss_wrapper()` defined but NEVER called (P3 uses `composite_loss` directly)
- **New (IT-5)**: `src/utils/serialization.py` — `load_ensemble()` defined but never imported/called anywhere (argument misalignment with `load_gbdt_model`)
- **New (IT-5)**: `src/utils/metrics.py` — `sharpe_ratio_torch` has no direct tests (only indirect via `sharpe_loss`)

**Checklist:**
- [ ] Search for unused imports (`grep` for import vs usage)
- [ ] Check dataclass/config definitions to verify they're consumed
- [ ] Check all `if` branches are reachable

**Files at risk:**
- `src/data/gbdt_features.py` (GBDTFeatureConfig)
- `src/train/*trainer*.py` (unused imports)

---

## Pattern: doc-clarity

**Occurrences:** 4
- IT-1: `lr_warmup_steps` named like per-batch but applied per-epoch
- IT-3: Missing comment on `y_pred.detach()` in ensemble.py
- IT-3: Missing comment on negative index prohibition in dataset.py
- IT-3: `min_ic_history` misleading — checks sample count, not IC history

**Checklist:**
- [ ] Parameter names should match their unit/scale (steps vs epochs, lookback vs threshold)
- [ ] Non-obvious tensor operations need comments (detach, clone, in-place ops)
- [ ] Validation/prohibition checks need rationale comments

**Files at risk:**
- `src/model/ensemble.py` (naming)
- `src/data/dataset.py` (validation rationale)

---

## Pattern: hasattr-heuristic

Using `hasattr(obj, attr)` for runtime dispatch is fragile — attribute names change, and the semantics can differ across call sites.

**Occurrences:** 3
- `trainer.py:100`: `hasattr(model, 'n_assets') and getattr(model, 'n_assets', 1) > 1` — checks > 1
- `advanced_trainer.py:41-43`: `hasattr(model, 'n_assets')` + `hasattr(model, 'output_dim')` — just checks existence
- `advanced_trainer.py:268`: `bool(hasattr(model, 'n_assets'))` — just checks existence
- `trainer.py:94-109`: `_is_ma` conditionally defined across 3 branches — fragile control flow

**Impact:** `hasattr(model, 'n_assets')` is True even when `n_assets=1`, causing the multi-asset path to activate incorrectly. The `> 1` check in trainer.py differs from `bool(hasattr(...))` in advanced_trainer.py, producing different validation metrics.

**Checklist:**
- [ ] Replace hasattr heuristics with explicit parameters forwarded through delegation chains
- [ ] `validate()` / `validate_advanced()` / `_train_single_window()` — pass `num_regression` and `is_multi_asset` explicitly
- [ ] `LossWrapper` — add optional `num_regression: int | None = None` parameter instead of requiring model instance

## Repository-level patterns learned

### Mamba NaN Handling — Mapping of all detection sites

| File | Warnings? | nan_to_num? | _nan_counter? | Status |
|------|-----------|-------------|---------------|--------|
| `mamba_block.py` (4 sites) | ✅ | ✅ | ✅ | OK |
| `mamba_parallel.py` (2 sites) | ✅ | ✅ | ✅ | OK |
| `ctm_model.py` (3 sites) | ✅ | ✅ | ✅ | OK |
| `loop_ctm.py` (forward: 4 sites, encode: 3 sites) | ✅ | ✅ | ✅ | FIXED IT-6 |
| `multiasset_ctm.py` (3 sites) | ✅ | ✅ | ✅ | NEW IT-6 |

### Trainer-Model Interface

```
scripts/train.py
  ├── _build_model_params() → dict → CTMStockModel / RecurrentCTM / MultiAssetCTM
  ├── WalkForwardTrainer / WalkForwardTrainerAdvanced / EnsembleWalkForwardTrainer
  │     └── all use walk_forward_windows() from _walk_forward_utils.py
  └── LossConfig, EnsembleConfig → trainers → loss functions
        └── CAUTION: lambda_sharpe ignored in GBDT loss bridge!
```

### Known Caller-Callee Delegations

- `advanced_trainer.validate_advanced()` → `trainer.validate()` (is_multi_asset not forwarded — BUG)
- `advanced_trainer._train_single_window()` → `validate_advanced()` (num_regression NOT forwarded — BUG)
- `scripts/train.py` → all 3 trainers (model params now unified via `_build_model_params`)
- `ensemble_trainer._prepare_gbdt_data()` → `gbdt_features.build_gbdt_feature_matrix()`
- `p3_trainer._prepare_per_asset_gbdt_preds()` — duplicates ensemble_trainer's GBDT data prep for per-asset

---

## Additional Patterns from Interface Contract Analysis

### Pattern: dangerous-default

A function signature default that silently causes incorrect behavior when used as-is.

**Occurrences:**
- `gbdt_features.py:166`: `build_gbdt_feature_matrix(ctm_hidden=None, include_ctm_features=True)` — default implies `ctm_hidden` is optional, but `include_ctm_features=True` with `ctm_hidden=None` raises `ValueError`. The expectation is `ctm_hidden` is always needed but typed as optional.

**Checklist:**
- [ ] Does the default value create a valid execution path?
- [ ] Does the type annotation match the actual requirements?
- [ ] Can a user call this with only required args and get correct behavior?

### Pattern: shared-reference-leak

Two output fields reference the same mutable object when they should be independent copies.

**Occurrences:**
- `ensemble_trainer.py:309,316`: `metrics=window_metrics` and `ctm_metrics=window_metrics` — same list identity. Mutating one affects the other.

**Checklist:**
- [ ] When two result fields are logically distinct, they should not share a reference
- [ ] After writing to a dataclass field, verify it's a copy, not the original reference

### Pattern: dead-config

A dataclass/config field is defined but never consumed by production code.

**Occurrences:**
- `gbdt_features.py:146-163`: `GBDTFeatureConfig` — defined but never instantiated in production (test-only)
- `gbdt_features.py:117-142`: `normalize_features()` — defined but never called in production
- `losses.py:54`: `LossConfig.skip_l2_reg` — never set from config in `scripts/train.py` (defaults to False always) — **WIRED in IT-5**
- `ensemble.py:113-114`: `EnsembleConfig.ctm_weight` and `gbdt_weight` — only used as fallback when `use_ic_weighting=False`. Primary path uses IC weighting exclusively.
- `configs/*.yaml` — `scaling.hierarchical` and `scaling.pool_factors` defined in all 5 configs but never read by any code
- `scripts/train.py` — `date_col` assignment (`date_column` config key) was dead code — **REMOVED in IT-5**

**Checklist:**
- [ ] Every dataclass field should have at least one production code path that reads it
- [ ] Search for all instantiations of a config class — are all fields represented?
- [ ] Unused code should be removed or wired up
- [ ] Config files: every key should have at least one read site in source code

---

## Additional Patterns from Test Coverage Analysis

### Pattern: config-default-mismatch

Config template default differs from actual code output. Auto-detect or override masks the issue for the primary path, but secondary consumers (auxiliary scripts, manual model construction) get the wrong value silently.

**Occurrences:**
- **FIXED (IT-7)**: `configs/*.yaml` all set `input_dim: 20`, but `features.py:compute_all_features()` produces exactly 9 features. `scripts/train.py:260` auto-detects (`input_dim = train_seq.shape[-1]`), so main training path is correct. But any code path reading `cfg["model"]["input_dim"]` directly (e.g., if someone builds a model without data) gets 20, causing shape mismatch.
- Root cause: The 20 was a conservative over-estimate, never updated when features changed.

**Checklist:**
- [x] After adding/removing features in `features.py`, update the `input_dim` default in ALL config files (FIXED IT-7: all 5 configs use input_dim=9)
- [x] Config defaults should be the actual expected value, not a conservative over-estimate (FIXED IT-7)
- [x] Any fallback value should have a corresponding warning when triggered (FIXED IT-7: fallbacks in verify_scales.py and run_daily.py updated to 9)

## Pattern: ambiguous-metric

A quantitative claim that lacks a clearly defined denominator or unit, making it misleading or non-reproducible.

**Occurrences:**
- **RESOLVED (IT-7)**: "5.7:1 param/data ratio" claimed without specifying denominator. With ~1000 training sequences (222K params / 1000 = 222:1), overfitting risk confirmed. Fix: reduced model_dim 128→64, state_dim 32→16, added param count logging to train.py.

**Checklist:**
- [x] Any ratio or proportion must specify exact numerator and denominator (FIXED IT-7: train.py logs params=, n_train=, param/seq ratio=X.X:1)
- [x] For time series: prefer "sequences" over "time steps" as denominator
- [x] When multiple denominators are possible, provide all and explain why each matters

Source interface changes without corresponding test updates.

**Known gaps (source → test mapping):**
- `src/data/features.py` (7 functions) → `tests/test_data.py`: `compute_volume_ratio`, `compute_realized_volatility`, `compute_all_features` have **zero test coverage** (43% gap)
- `src/model/fused_attention.py` (2 classes) → **no dedicated test file**: 0% coverage — entire module untested
- `src/train/_walk_forward_utils.py` → **no standalone tests**: boundaries like `step_size > window`, `purge_period=0` untested
- `src/utils/metrics.py` → **no standalone tests**: `sharpe_ratio_torch` used everywhere but edge cases untested
- `src/model/multiasset_ctm.py` → `use_fused_attention=True` path has **zero test coverage**

**NaN/edge-case tests that must be updated when fixing NaN/error handling:**
See docs/improvement_loop_report.md "NaN/Edge/Error Handling Test Reference" for the 19 specific tests.

**Checklist:**
- [ ] Fix touches `fused_attention.py` → MUST add test coverage (module is 0%)
- [ ] Fix touches `features.py` → check if `compute_volume_ratio`, `compute_realized_volatility`, or `compute_all_features` — they have no tests
- [ ] Fix touches `_walk_forward_utils.py` → add unit tests for boundary conditions
- [ ] Fix touches signature → grep for test callers and update
- [ ] Run full test suite after every fix batch (not just changed file's tests)

### 🔴 CORRECTION: GBDTFeatureConfig and normalize_features are NOT dead code

The defect_patterns KB (this file) previously flagged `GBDTFeatureConfig` and `normalize_features()` as dead code.
**This was incorrect.** Both are actively used:

- `GBDTFeatureConfig` is instantiated in `ensemble_trainer.py:354` and its fields are read at lines 355-361
- `normalize_features()` is called inside `build_gbdt_feature_matrix()` at line 210, which is called by `ensemble_trainer.py:367`
- Neither is dead code — they are integral to the GBDT feature pipeline

Remove the "dead-config" and "dead-code" entries for these items. The remaining dead-code items are still valid (unused output_heads, `import math` in 4 files which was already removed).

---

## Pattern: missing-test-coverage

Modules with zero test coverage — when modifying these, MUST add tests first.

| Module | Coverage | Priority | Lines | Action if modified |
|--------|----------|----------|-------|--------------------|
| `src/model/fused_attention.py` | **0%** | 🔴 CRITICAL | 226 | Must write tests before or alongside any changes |
| `src/train/p3_trainer.py` | **0%** | 🔴 CRITICAL | 502 | Must write smoke test before modifying |
| `src/utils/shap_explainer.py` | **0%** | 🔴 CRITICAL | 406 | Must add unit tests for tree parsing |
| `src/model/multiasset_ctm.py` | 40% | 🔴 HIGH | 303 | `use_fused_attention=True` path 0% coverage |
| `src/data/features.py` | 57% | 🟡 HIGH | 166 | 3/7 functions untested (volume_ratio, realized_vol, all_features) |
| `src/train/_walk_forward_utils.py` | 0% direct | 🟡 MODERATE | 38 | Boundary conditions untested |
| `src/utils/metrics.py` | 0% direct | 🟡 MODERATE | 41 | Edge cases for sharpe_ratio_torch |
| `src/utils/serialization.py` | ~5% | 🟡 MODERATE | 135 | Only smoke test on nn.Linear |
| `scripts/infer.py` | 0% | 🟡 MODERATE | 477 | Integration test needed |
| `src/data/gbdt_features.py` | 100% | ✅ | 161 | Best in project |
| `src/train/loss_bridge.py` | 100% | ✅ | 97 | Good direct coverage |
| `src/model/ensemble.py` | 71% | ✅ | OK | ic_weighted_fusion exercised indirectly
