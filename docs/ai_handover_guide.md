# AI Handover Guide — Glaubenskrieg CTM

> **Purpose**: This document is the entry point for any AI agent starting work on this project.
> **Read this first** before making any changes. Then read the supporting docs listed below.
> **Last updated**: 2026-06-06 (v5: new hardware, 5 root cause fixes, experiments running)

---

## Project Overview

**Glaubenskrieg** is a quantitative investment ML project combining:
- **CTM (Conv-Temporal-Mamba)**: Pure PyTorch Mamba S6 state space model for stock prediction
- **GBDT Ensemble**: C++ tree boosting (Hoffnung backend) fused via IC-weighted signals
- **Walk-Forward Validation**: Temporal cross-validation with purge period

**Architecture**: `data/` → `model/` → `train/` → `scripts/`

```
src/
├── data/        Dataset, technical indicators, GBDT feature engineering
├── model/       CTMStockModel, MambaBlock (S6 SSM), MultiAssetCTM, losses, ensemble
├── train/       Trainers (basic, advanced, ensemble), loss bridge
└── utils/       Gradient checker, metrics (Sharpe ratio)
```

---

## Improvement Loop History

This project has been through **multiple improvement loop iterations** across separate sessions:

| Session | Iterations | Issues Fixed | Scope |
|---------|-----------|-------------|-------|
| Pre-session 1 | 5 loops | Initial NaN handling, type annotations, .gitignore | Scattered |
| **Session 1** | **4 loops** | **26 issues** | **Full project review** |
| **Session 2 (IT-7)** | **1 loop** | **config defaults + overfitting + 8 pre-existing test fixes** | **Data quantity & overfitting** |
| **Session 3** | **v3-v4-v5** | **TGPE design, 5 root cause fixes, hardware migration** | **TimeGate, curriculum, Ascend→V100/3090** |

### This Session's Results

| Iteration | Fixes | Key Changes |
|-----------|-------|-------------|
| IT-1 | 10 (2C + 8M) + 7 regression fixes | NaN handling standardized, Sharpe unified, model params factory, loss_bridge warning, ensemble trainer dedup, lr_warmup renamed |
| IT-2 | 7 (1M + 6m) | Fused attention batch alignment, NaN in realized_volatility, removed 4 unused imports, vectorized SMA |
| IT-3 | 9 (9m) | TrainingResult dataclass, direct imports, config validation, vectorized rolling_normalize, torch.from_numpy, Hessian warning |
| IT-4 | Review passed | **REVIEW_PASSED: true** — codebase in excellent shape |

**243/243 tests pass** — verified at every iteration.

---

## How the Next AI Should Work

### 1. First: Read These Docs (in order)

| Order | Doc | Purpose |
|-------|-----|---------|
| 1 | `docs/ai_handover_guide.md` | **This file** — project context and methodology |
| 2 | `docs/defect_patterns.md` | **Critical** — known defect patterns with checklists. Load this into every review prompt. |
| 3 | `docs/improvement_loop_v2.md` | **The methodology** — how to run future improvement loops correctly. Supersedes `improvement-loop.md`. |
| 4 | `docs/improvement_loop_report.md` | Full inventory of all 26 fixes from this session |
| 5 | `docs/improvement_loop_config.json` | Agent routing configuration for differentiated review/fix agents |

### 2. Then: Check Git Log for Context

```bash
git log --oneline -30
git log --oneline --grep='pattern:' -20   # Find defect patterns in commit messages
```

### 3. Then: Load the Defect Patterns into Your Review Prompt

When starting a review pass, prepend the contents of `docs/defect_patterns.md` to your prompt.
This compensates for "fresh context" — you won't have memory of previous sessions, but
the patterns document gives you that memory.

---

## The v2 Improvement Loop (Quick Reference)

```
┌─ Review Phase (4 parallel passes) ───────────────────┐
│  Pass A: Oracle (architecture)                       │
│  Pass B: explore × 3 (consistency — sibling files)   │
│  Pass C: explore × 2 (interface contracts — callers) │
│  Pass D: explore × 1 (test coverage — fix→test map)  │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌─ Fix Phase ──────────────────────────────────────────┐
│  deep or unspecified-high agent                      │
│  Batch: 1-5→all, 6-15→C+M first, 16+→max 10/round   │
│  Tag commits: "iter-improve-loop: N fix [pattern:X]" │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌─ Regression Scan (Direction D) ──────────────────────┐
│  1. Test contract drift: full test suite             │
│  2. Sibling pattern sync: explore agent              │
│  3. Side effect check: imports, dead code            │
└──────────────────────┬───────────────────────────────┘
                       ▼
              Back to Review or Done
```

### Agent Routing Table

| Role | Agent Type | Cost | Notes |
|------|-----------|------|-------|
| Architecture Review | `oracle` | High | Read-only, highest IQ. Never modifies code. |
| Consistency Scan | `explore` (×3, parallel) | Low | Grep-focused cross-file matching |
| Interface Check | `explore` (×2, parallel) | Low | Caller-callee signature verification |
| Coverage Check | `explore` (×1) | Low | Module→test mapping |
| Fix Execution | `deep` or `unspecified-high` | Medium | Goal-oriented, reads first then edits |
| Regression Scan | `unspecified-low` | Low | Quick side-effect detection |

### DO NOT:
- Use the same agent/session for review AND fix (confirmation bias)
- Skip regression scan after fixes
- Use vague "review the code" prompts — use structured multi-pass prompts
- Fix more than 10 issues per iteration batch

---

## Known Remaining Issues (Unfixed)

These are the issues found in IT-4 review that were deemed non-blocking:

### Low Priority (Optional Polish)

| # | File | Issue | Impact |
|---|------|-------|--------|
| 1 | `multiasset_ctm.py:158,190` | Dead `is None` guards after direct-import fix. `RecurrentCTM`/`FusedMultiHeadCrossAttention` can never be `None` now. | Cosmetic — misleading but harmless |
| 2 | `loop_ctm.py:206,213,228,236` | NaN detection only warns — no `nan_to_num`. Inconsistent with other Mamba variants (mamba_block, mamba_parallel, ctm_model all do both warn + fix). | Low — underlying `_encode_blocks` handles NaN, so these are just defensive checks on unhandled paths |
| 3 | `loss_bridge.py:71` | `warnings.warn` fires every GBDT call. Log flood risk. Should use module-level `_warned_sharpe_loss = False` flag. | Low — cosmetic |
| 4 | `loop_ctm.py:160-172 vs 210-230` | Loop logic duplicated between `encode()` and `forward()`. DRY violation. | Info — no bugs |

### From Explore Agents (Not Yet Fixed)

| # | File | Issue | Severity |
|---|------|-------|----------|
| 5 | `loop_ctm.py` | NaN detection warns but doesn't `nan_to_num` — inconsistent with all other Mamba variants | **Major** — cross-file inconsistency |
| 6 | `ensemble.py:37-38` | `np.std()` uses population ddof=0, while `sharpe_ratio_torch` uses sample ddof=1 | **Major** — numerical inconsistency |
| 7 | `advanced_trainer.py → trainer.py` | `validate_advanced()` doesn't forward `is_multi_asset` to `validate()` | **Major** — delegation broken |
| 8 | `ensemble_trainer.py vs advanced_trainer.py` | `log_gradients` at different levels (`__init__` vs `train_walk_forward` param) | **Minor** — signature inconsistency |
| 9 | `gbdt_features.py` | `build_gbdt_feature_matrix` default `include_ctm_features=True` requires `ctm_hidden` — dangerous default | **Minor** — design smell |
| 10 | `ensemble_trainer.py:309,316` | `ctm_metrics` and `metrics` share same list reference | **Minor** — mutation leak |
| 11 | `losses.py` | `num_regression` typed `Optional[int]` but raises ValueError if None — signature/code mismatch | **Minor** — documentation |
| 12 | `loss_bridge.py` | `skip_l2_reg` from LossConfig silently ignored | **Minor** — config unused |
| 13 | `gbdt_features.py` | `GBDTFeatureConfig` never used in production, `normalize_features()` never called | **Minor** — dead code |

### Coverage Gaps (Highest Priority)

| Module | Coverage | Action Needed |
|--------|----------|--------------|
| `src/model/fused_attention.py` | **0%** | Entire module untested. Add tests before modifying. |
| `src/data/features.py` | 57% | `compute_volume_ratio`, `compute_realized_volatility`, `compute_all_features` untested |
| `src/model/multiasset_ctm.py` | 40% | `use_fused_attention=True` path untested |
| `src/train/_walk_forward_utils.py` | 0% | No standalone boundary tests |
| `src/utils/metrics.py` | 0% | `sharpe_ratio_torch` no standalone edge case tests |

---

## Defect Patterns Summary (Top 3 Risk Areas)

The most common defect categories across all iterations:

### 1. Cross-File Inconsistency (🔴 Most Common — 9+ occurrences)
When a bug is fixed in one file, sibling files with the same pattern are missed.
- **Check-list**: When modifying any Mamba variant, check ALL 4 (`mamba_block.py`, `mamba_parallel.py`, `loop_ctm.py`, `ctm_model.py`). When modifying a trainer, check ALL 3.

### 2. Interface Contract Mismatch (🟠 4 occurrences)
Config fields defined in one place, silently ignored in consumers.
- **Check-list**: After adding a config field, grep for all `.field_name` usages across all consumers. After changing a function signature, grep for all callers.

### 3. Silent Fallback (🟠 4 occurrences)
`.get(key, default)`, `.abs().clamp()`, integer division — values silently replaced with no warning.
- **Check-list**: Every `warnings.warn` should have a recovery action. Every `.get()` fallback should be justified.

---

## Docs/ File Index

| File | What It Contains | When to Read |
|------|-----------------|--------------|
| `docs/ai_handover_guide.md` | **This file** — master context for new AI | First |
| `docs/defect_patterns.md` | Known defect patterns with severity, checklists, risk files | Every review prompt |
| `docs/improvement_loop_v2.md` | Full v2 methodology spec — supersedes v1 | Before running any loop |
| `docs/improvement_loop_report.md` | Complete inventory of all 26 fixes from this session | Reference for what was done |
| `docs/improvement_loop_config.json` | Machine-readable agent routing config | CI/automation |
| `docs/pretrained_ensemble_guide.md` | **NEW** — Pre-trained weight loading for ensemble training | Before using `--ensemble-inference` |
| `docs/ensemble_deployment_guide.md` | Full training/inference deployment guide | Before deploying |
| `docs/fusion_roadmap.md` | Original ensemble fusion design doc | Background only |
| `docs/loop_mamba_design.md` | Recurrent CTM design doc | Background only |
| `docs/performance.md` | Performance benchmarks | Background only |

---

## P3 Three-Stage Fusion (New)

A `P3EnsembleTrainer` (`src/train/p3_trainer.py`) extends the ensemble workflow with a Stage 3: after GBDT training, builds `MultiAssetCTM(use_fused_attention=True)`, transfers backbone weights from Stage 1, freezes the Mamba backbone, and fine-tunes `FusedMultiHeadCrossAttention` (including `GBDTModulator`) + output heads. GBDT predictions are pre-computed per-asset and injected via `model(x, gbdt_preds=...)`.

Usage:
```bash
python scripts/train.py --config configs/default.yaml \
  --multi-asset --n-assets 10 \
  --ensemble --p3 \
  --modulator-epochs 10 --modulator-lr 1e-4
```

P3 results appear in metrics as `p3_sharpe`, `p3_ic`, `p3_ensemble_sharpe` alongside the standard ensemble metrics.

## Quick Start for Next AI Session

### Server Access (2026-06-06)

```bash
# S1 (2× V100 32GB): ssh root@223.109.239.36 -p 24212  # pwd: feingo7h
# S2 (2× RTX 3090 24GB): ssh root@223.109.239.36 -p 33748  # pwd: <REDACTED>
# Both: PYTHONPATH=/root/Glaubenskrieg:/root/Hoffnung/build:/root/Hoffnung/build/python
```

### Session Checklist

```markdown
1. Read: docs/ai_handover_guide.md (this)
2. Read: docs/experiments.md (server info, experiment design, root causes)
3. Read: docs/defect_patterns.md (load into prompt)
4. Read: .omo/session_context.md (full project state, decision log)
5. Check: git log --oneline -10
6. Run: python -m pytest tests/ -q (should be 256 passed)
7. Verify: v5 experiments status on servers
   - Results: /root/results/v5/
   - Logs: /root/logs/v5_*.log
8. Decide: What to do next?
   - Analyze v5 results (3 variants × 5 seeds)
   - Compare v5 vs v3 baseline
   - Fix remaining issues (see Known Remaining Issues)
   - Extend TGPE fusion variants
```

---

## Contact / Origin

- Project: Glaubenskrieg CTM — Conv-Temporal-Mamba for Quantitative Investment
- architecture docs: `docs/ctm_architecture_guide.pdf`
- entry point: `scripts/train.py`
- config: `configs/default.yaml`
