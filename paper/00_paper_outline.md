# Glaubenskrieg: A Multi-Paradigm Exhaustive Test of ML Stock Return Prediction

## A Methodology Paper with Negative Empirical Results

**Authors**: [TBD]  
**Target Venue**: ICAIF 2026 / Journal of Financial Data Science  
**Status**: Draft materials compiled, paper outline below

---

## Abstract (Draft)

> We conduct a multi-paradigm exhaustive test of machine learning stock return prediction from OHLCV-derived features across two markets (A-Share, US; 677 total stocks). Six paradigms—deep learning (Mamba SSM), gradient boosting (GBDT), hybrid ensemble, feature-engineered LightGBM, cross-sectional ranking, and meta-labeling—all converge to IC ≈ 0.00 on held-out test data. An initial walk-forward validation IC of 0.14 was traced through five bug fixes to a true out-of-sample IC of 0.006, revealing a systematic overfitting pattern that we formalize as a five-step IC validation diagnostic framework. Volatility prediction proved robust across markets (QLIKE = −6.42 to −7.10), but GARCH(1,1) with three parameters outperformed gradient-boosted trees on all 47 tested stocks. A regime-adaptive volatility-targeting strategy improved Sharpe by +0.216 through tactical risk reduction, not stock selection alpha. We conclude that OHLCV-derived features at daily frequency contain no detectable return-predictive signal across markets, and that ML complexity adds noise rather than signal to volatility prediction. We release the full codebase, all experimental results, and the five-step diagnostic protocol as open-source tools for the financial ML community.

---

## Paper Structure (Option A + B Hybrid: Methodology Case Study)

### 1. Introduction (~1500 words)
- The financial ML reproducibility crisis (Nature 2025: "most LSTM/DNN stock prediction studies irreproducible")
- The gap between claimed and realized performance
- Motivation: rigorous negative results as a public good
- Our approach: exhaustive testing + diagnostic framework + cross-market validation
- Contribution statement: (1) multi-paradigm negative result, (2) 5-step diagnostic framework, (3) GARCH beats ML for vol prediction

### 2. Related Work (~2000 words)
> **Foundation**: A 10-paper head-to-head comparison (`paper/CLASSIC_LITERATURE_COMPARISON.md`) maps Glaubenskrieg's results against each major published benchmark. **Table 1** (in §2) summarizes agreement/divergence across all 10 studies.
- ML in finance: Gu-Kelly-Xiu (2020), Kelly-Malamud LLG (2025), McElfresh (2023), Grinsztajn (2022)
- DL architectures for stock prediction: Mamba, Transformer, LSTM, GNN-based (SAMBA, HIGSTM)
  - CNN-LightGBM with wavelet denoising (Bai et al. 2026) — contrasted with Glaubenskrieg's EMA denoising (ΔIC=−0.009)
  - VLSTM with VSN adaptive gating (Saly-Kaufmann et al. 2026, Oxford) — different data regime (5 asset classes, 50-seed top-10)
- Volatility modeling: GARCH (Engle 1982, Bollerslev 1986), LightGBM comparisons
  - Glaubenskrieg confirms GARCH(1,1) beats ML on 300/300 stocks (100%)
- Known pitfalls: walk-forward overfitting, look-ahead bias, alpha hallucination (Ye 2025)
  - Reproducibility crisis in stock prediction: Nature (2025) survey — Glaubenskrieg's protocol (purged WF + held-out test + 5-step diagnosis + power analysis) is a counter-example
- Meta-labeling and ensemble methods (López de Prado 2018)


### 3. Methodology (~2500 words)

#### 3.1 Data & Features
- Two markets: A-Share (multi-asset 50 + cross-sectional 200), US (477 stocks, S&P500+NASDAQ100)
- 9 OHLCV-derived features (SMA, RSI, Bollinger, volume ratio, etc.)
- All computations causal (no look-ahead bias)

#### 3.2 Six Prediction Paradigms
1. **Deep Learning**: MultiAssetCTM (Mamba SSM, 97K params) with cross-asset attention
2. **Gradient Boosting**: Hoffnung C++ GBDT + LightGBM
3. **Hybrid Ensemble**: CTM+GBDT with TimeDecayGate (TGPE)
4. **Feature Engineering**: LightGBM + EMA smoothing + DM test
5. **Cross-Sectional Ranking**: lambdarank with CS z-scores
6. **Meta-Labeling**: XGBoost binary filter (López de Prado §3)

#### 3.3 Walk-Forward Protocol
- Purged walk-forward: train=1000d, purge=126d, val=200d, step=126d
- Held-out test set (never used in training or early stopping)
- Bug evolution: v1→v5 (shuffle fix, step_size 21→63, lambda_sharpe 0.5→0.1)

#### 3.4 Statistical Tests
- Spearman IC with per-window decomposition
- Diebold-Mariano test for prediction comparison
- Ridge regression (451 params) as linear baseline
- QLIKE (Patton 2011) for volatility forecast evaluation
- Paired t-tests for strategy comparison

### 4. The Five-Step IC Validation Diagnostic Framework (~1500 words)
*This is the paper's strongest novel contribution*

| Step | Name | Method | Pass Threshold |
|:----:|------|--------|:--------------:|
| 1 | Held-out Test IC | Checkpoint → unseen test set | |Δ|(WF IC − Test IC) < 0.02 |
| 2 | Linear Baseline | Ridge regression on same data | DL IC > Ridge IC + 0.01 |
| 3 | Window Stability | Per-window IC decomposition | CV(IC) < 0.3 across windows |
| 4 | Per-Stock Significance | Binomial test on per-stock IC sign | p < 0.01 (two-sided) |
| 5 | Loss Gap Analysis | Train vs validation loss divergence | No U-shaped divergence before epoch 10 |

**Application to Glaubenskrieg**: v3 WF IC=0.14 → Step 1 reveals test IC=0.006 → Steps 2-5 confirm systematic overfitting. Framework would have flagged the false positive at v3.

### 5. Results (~3000 words)

#### 5.1 Cross-Market Return Prediction
```
Market   Stocks  Windows   LGB IC        Ridge IC      Verdict
A-Share (multi-asset)  50   3      0.053±0.054   0.006         ≈0
A-Share  200     6         -0.003±0.042  0.030±0.054   ≈0
US       477     10        0.007±0.043   0.031±0.030   ≈0
```
**Finding**: IC ≈ 0 across all three markets, all window counts, all models.

#### 5.2 Architecture Comparison (v5 Final)
```
Model              Params   WF IC    Test IC   Test Sharpe
CTM (Mamba SSM)    97,000   0.049    0.006     -0.093
CTM + TimeGate     —        0.037    —         -0.024
GBDT (Hoffnung)    —        0.008    —         -0.088
Linear Ridge       451      —        0.006     +0.55
```
**Finding**: 97K DL parameters = 451 linear parameters. Zero DL uplift.

#### 5.3 Volatility Prediction: GARCH(1,1) vs LightGBM
```
Predictor         Mean QLIKE   Wins (of 47 stocks)
GARCH(1,1)        2.31         47/47 (100%)
Historical Mean    2.43         0/47
LightGBM           12.75        0/47 (0%)
```
**Finding**: GARCH's 3 parameters systematically outperform LightGBM's 200 trees. DM test: p ≈ 1.6×10⁻¹¹.

#### 5.4 Portfolio-Aware Optimization
```
Approach              Sharpe   IC      Δ vs MSE
Random                0.04     0.001   —
LGB-MSE (baseline)    2.36     0.011   —
LGB-SharpeTuned       2.98     0.015   +0.62 (p=0.011)
```
**Finding**: Portfolio-aware HP selection improves Sharpe by 26%, but IC remains at 0.015 — far below the 0.02 threshold for reliable trading.

#### 5.5 Regime-Adaptive Volatility Strategy
```
Strategy          Sharpe   ΔSharpe   MaxDD
Equal-Weight      1.080    —         37.1%
Inverse-Vol       1.061    -0.019    35.1%
Vol-Target 15%    1.110    +0.030    22.0%
Regime-Adaptive   1.296    +0.216    19.2%
```
**Finding**: All improvement comes from tactical leverage reduction during high-vol regimes, not from stock selection.

### 6. Discussion (~1500 words)
- LLG framework interpretation: IC≈0 may reflect learning gap, not signal absence
- Why volatility prediction ≠ return prediction (first vs second moment)
- The Sharpe paradox: positive Sharpe + zero IC = strategy construction artifact
- Comparison with published claims (MambaStock, FinCon, PULSE-KAN)
- Why this negative result is a public good for the field

### 7. Limitations (~800 words)
- Daily frequency only (no intraday microstructure)
- Pure OHLCV features (no text, fundamentals, alternative data)
- No transaction cost modeling for strategies
- Cross-market protocols not perfectly aligned
- No permutation test executed (designed but not run)
- Universe survivorship bias in US stock selection

### 8. Conclusion (~500 words)
- OHLCV at daily frequency is exhausted as a signal source
- The 5-step diagnostic framework should be adopted as standard practice
- GARCH remains the gold standard for daily volatility prediction
- Future work MUST incorporate non-price data sources and/or intraday frequency

---

## Key Tables (from paper/results/01_master_table.md)

See `paper/results/01_master_table.md` for the complete unified results table covering all 12 experiment groups.

## Key Figures (from paper/results/02_cross_market_figure_data.md)

See `paper/results/02_cross_market_figure_data.md` for ready-to-plot data for 8 figures.

---

## Literature Coverage

Detailed literature analysis in:
- `paper/literature/01_proven_systems.md` — Tier 1-3 systems with evidence grades
- `paper/literature/02_gbdt_vs_dl.md` — Structural reasons GBDT beats DL on tabular data
- `paper/literature/03_finance_ml_limits.md` — LLG framework, alpha hallucination, key references

## Methodology Details

- `paper/methodology/01_walk_forward_protocol.md` — Purged walk-forward design
- `paper/methodology/02_diagnostic_framework.md` — 5-step IC validation framework
- `paper/methodology/03_statistical_tests.md` — DM test, paired t-tests, permutation design

## Experimental Findings

- `paper/findings/01_cross_market_ic.md` — Definitive cross-market IC table
- `paper/findings/02_volatility_prediction.md` — GARCH vs LGB, regime-switching
- `paper/findings/03_architecture_comparison.md` — CTM vs GBDT vs Ensemble vs Ridge
- `paper/findings/04_portfolio_optimization.md` — Portfolio-aware training results
- `paper/findings/05_negative_results.md` — All failed paradigms in detail

---

## Venue Strategy

1. **Primary target**: ICAIF 2026 (ACM Int'l Conference on AI in Finance) — methodology papers welcome
2. **Secondary target**: Journal of Financial Data Science — explicitly publishes negative results
3. **Preprint**: arXiv (q-fin.ST or cs.LG) simultaneous with submission
4. **Backup**: NeurIPS Workshop on ML in Finance, ICAIF Workshop

---

## Pre-Submission Checklist

- [ ] Write full paper draft following this outline
- [ ] Create .bib file with all citations in consistent format
- [ ] Generate all figures from figure_data.md
- [ ] Run permutation test (Step 4 of diagnostic framework)
- [ ] Add transaction cost sensitivity analysis
- [ ] Engage with LLG framework interpretation
- [ ] Fix README to remove "live trade execution" claims
- [ ] Create replication package (requirements.txt, Docker)
- [ ] External review by non-project member
- [ ] Format for target venue
