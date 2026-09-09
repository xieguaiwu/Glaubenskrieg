# Oracle Assessment: Glaubenskrieg Paper Viability

> **Date**: 2026-06-06  
> **Assessor**: Oracle (verification & validation specialist)  
> **Assessment Type**: Pre-writing feasibility evaluation  
> **Status of paper/ directory**: Empty scaffolding (5 subdirectories, zero content files)

---

## Executive Summary

**The Glaubenskrieg project has a publishable negative result, but not as currently framed.** The raw materials are strong: 6-paradigm exhaustive testing, cross-market (3-market) validation, careful bug diagnosis, and a literature survey that properly contextualizes the findings. However, the paper has not been written — the `paper/` directory is empty scaffolding. There is no draft manuscript, no structured narrative, no compiled figures, no bibliography. What exists is excellent raw material, but raw material is not a paper.

**Brutal honesty**: If submitted today in its current form (i.e., a zip file of progress.md + architecture_assessment.md), it would be rejected from any peer-reviewed venue. With proper writing, it could be a solid workshop or lower-tier journal paper. The core problem is that the paper's contribution is a *negative result*, which is scientifically valuable but publication-venue-hostile.

---

## 1. Completeness Assessment

### What IS captured in the source materials:

| Component | Status | Quality |
|-----------|:------:|:-------:|
| Experimental methodology | ✅ Documented | Excellent — all 6 paradigms, walk-forward protocol, bug fixes tracked |
| All results (positive & negative) | ✅ Fully recorded | Excellent — per-seed, per-window, per-market |
| Cross-market validation | ✅ HK + A-Share + US (477 stocks) | Strong — rare in literature |
| Architecture description & audit | ✅ Full architecture trace | Excellent — 157K→97K params, dead code identified |
| Bug diagnosis & IC paradox | ✅ 5-step diagnostic framework | Outstanding — the strongest single contribution |
| Literature comparison | ✅ Tiered survey with citations | Good — covers 20+ systems with evidence grading |
| Theoretical grounding | ✅ LLG framework, SNR analysis | Adequate — Kelly & Malamud, Grinsztajn, Bai |
| Statistical protocol | ✅ DM test, QLIKE, permutation design | Strong — DM test used, QLIKE for vol, Ridge baseline |
| Code & reproducibility | ✅ Full codebase, fixed seeds | Good — but no container/Docker recipe |

### What is MISSING:

| Gap | Severity | Notes |
|-----|:--------:|-------|
| **No paper draft** | 🔴 CRITICAL | Paper directory is empty — nothing to evaluate as a "paper" |
| **No abstract/narrative** | 🔴 CRITICAL | Raw data dump, not a story with a thesis |
| **No figures/tables** | 🔴 CRITICAL | figures/ directory is empty |
| **No bibliography file** | 🔴 | No .bib, no formal citation format |
| **No structured contribution statement** | 🔴 | What is THE one thing this paper contributes? Unclear |
| **No related work section** | 🟡 | Literature survey exists but is a survey, not a focused related-work |
| **No formal statistical tests reported** | 🟡 | DM test exists but permutation test only *designed*, not executed |
| **No replication package** | 🟡 | Code exists but no requirements.txt pinned to paper version |
| **No author list / contribution statement** | 🟡 | Who wrote what? |
| **Inconsistent baseline reporting** | 🟡 | US data has 477 stocks, HK has 200, A-Share has 200 — not truly comparable |
| **Volatility backtest economic interpretation** | 🟡 | QLIKE is technically correct but strategy-level economic value is weak (+0.001 Sharpe in US bull) |
| **No ablation studies on data quantity** | 🟡 | 200→477 stocks, 3yr→10yr — is the result stable with even more data? |

---

## 2. Honesty Assessment

**Score: 9/10** — This is the project's strongest virtue.

### What's done well:

1. **Negative results are THE headline**, not buried in appendix. The title "final verdict" in progress.md is exactly what a paper abstract should say.
2. **Walk-forward IC deliberately debunked.** The IC paradox diagnosis (v3 IC=0.14 → v5 IC=0.049 → test IC=0.006) is a rare example of a project catching its own false positive.
3. **GBDT used as a "lie detector".** The project explicitly notes that GBDT's IC=0.008 is the most honest indicator because trees don't hallucinate false patterns the way over-parameterized NNs do.
4. **TGPE and Meta-Labeling failures reported in full detail**, not hidden.
5. **"Sharpe ≠ prediction ability" is explicitly stated** with sign-based strategy construction identified as the confound.
6. **Limitations section is thorough** — data constraints, frequency limits, no external data sources.

### What could be more honest:

1. **"Cross-market validation" overstates the case.** HK (200 stocks, 3yr, 3 windows), A-Share (200 stocks, 2.4yr, 6 windows), US (477 stocks, 10yr, 10 windows) — these are not comparable experimental designs. The US experiment is substantially larger. Calling this "cross-market confirmation" when the protocols differ is slightly misleading.
2. **"Exhaustive" testing claim.** 6 paradigms is thorough, but "exhaustive" implies every possible approach. No GNN, no attention-only Transformer, no RL, no ensemble-of-ensembles. "Comprehensive" would be more accurate than "exhaustive."
3. **The volatility "success" is overclaimed.** QLIKE = -6.58 ± 0.001 shows statistical robustness, but the economic significance is negligible: inverse-vol weighting produces ΔSharpe of +0.001 (US) to +0.056 (A-Share). A paper reviewer will ask: "If the best you can do is 0.056 Sharpe improvement, is this economically meaningful?"
4. **Linear Ridge = DL claim needs a caveat.** Ridge test Sharpe = +0.55 vs CTM = -0.028. But both have IC ≈ 0. The Sharpe difference is from strategy construction variance, not predictive skill. The paper must clearly separate "Ridge is better" (false — both have IC=0) from "Ridge is not worse" (true).
5. **Missing: explicit statement that this project cannot be used for trading.** The README still says "from raw OHLCV data to live trade execution in one pipeline" — this is contradicted by the experimental results. If this paper is submitted, the README must be corrected to avoid misleading readers.

---

## 3. Novelty Assessment

**Score: 4/10** — There is a genuine contribution, but it is narrow.

### Genuinely novel contributions:

| Contribution | Novelty | Supporting evidence |
|-------------|:-------:|---------------------|
| **Multi-paradigm negative result** with cross-market validation | Medium | No paper we surveyed does 6-paradigm exhaustive testing on the same data with a 3-market comparison. Most papers test one model on one market and report positive results. |
| **IC paradox diagnostic framework** (5-step) | Medium-High | The 5-step diagnosis (checkpoint→test IC, linear baseline, window stability, per-stock decomposition, loss gap analysis) is a reusable methodology. This could stand as a methods paper. |
| **Mamba SSM in financial prediction** → negative result | Low-Medium | Several papers have tested Mamba on finance (MambaStock, FinMamba) with positive claims. A rigorous negative result is novel and valuable. |
| **TGPE design + failure analysis** | Low | The TimeGate design is original, but the fact that it failed limits its contribution value. |

### What is NOT novel:

1. **"OHLCV features don't predict daily returns"** — This is well-established. Gu-Kelly-Xiu (2020), the LLG framework (2025), and dozens of papers confirm this. The project's finding is a *replication*, not a discovery.
2. **"Deep learning doesn't beat linear models on tabular data"** — McElfresh et al. (NeurIPS 2023) demonstrated this on 176 datasets. Grinsztajn et al. (NeurIPS 2022) provided the theoretical explanation.
3. **"Volatility clustering is predictable"** — Engle (1982) got a Nobel for this. LightGBM beating persistence by QLIKE=0.6 is a minor increment.
4. **"Meta-Labeling doesn't help with zero-IC primary model"** — This is mathematically obvious from López de Prado's own formulation.
5. **"Walk-forward IC can be misleading"** — Known issue; Bailey & López de Prado's deflated Sharpe ratio and PBO framework address this.

### Novelty bottleneck:

The paper's core finding is: *"We tried very hard and found nothing."* This is scientifically honest, but it's a **confirmation of existing knowledge**, not a discovery. The incremental contribution is the *rigor of the confirmation* (multi-paradigm, cross-market, bug diagnosis), not the finding itself.

---

## 4. Literature Comparison Quality

**Score: 7/10** — Thorough but not paper-ready.

### Strengths:

1. **Evidence-grading system** (★★★★★ to ★☆☆☆☆) is excellent and should be retained in the paper.
2. **Tier 1 systems** (CatBoost/LightGBM, Linear Ridge, VLSTM, CNN-LightGBM, intraday SSRN) are properly identified with specific metrics.
3. **Tier 3 (debunked claims)** section correctly identifies FinCon, MambaStock, PULSE-KAN as non-credible.
4. **Quant fund comparison** provides industry context that pure academic papers often lack.
5. **10-dimension gap analysis** (Section 4 of survey) is a useful self-assessment.

### Weaknesses:

1. **No formal bibliography format.** Citations are scattered across multiple .md files in inconsistent formats (some have DOIs, some only URLs, some just names).
2. **Some critical papers are cited but not deeply engaged.** For example, Kelly & Malamud (2025) LLG is cited, but how does Glaubenskrieg's IC=0 relate to the LLG prediction? This connection is not made.
3. **Missing comparison to Gu-Kelly-Xiu (2020) results.** Their monthly R² ≈ 0.4% → what does Glaubenskrieg's daily IC=0 imply for monthly? No conversion/comparison provided.
4. **McElfresh et al. (2023) 176-dataset benchmark** is the most relevant comparison but only mentioned in passing. The paper should explicitly say: "Our finding that DL ≤ GBDT on financial tabular data is consistent with McElfresh's finding on 176 general tabular datasets."
5. **No engagement with the "dark matter" implication.** If Kelly & Malamud are right that true predictability is ~20% but realized OOS R² is ~1-2%, then IC=0 doesn't mean "no signal exists" — it means "we can't learn it from finite data." The paper should engage with this interpretation.
6. **The survey over-claims Glaubenskrieg's protocol quality.** In Section 7 it says "Glaubenskrieg's rigorous validation protocol... is itself top-tier." This is accurate for walk-forward + held-out test, but the project does NOT implement purged k-fold CV, deflated Sharpe ratio, or the probability of backtest overfitting (PBO). These are standard in top-tier quantitative finance papers (Bailey & López de Prado).

---

## 5. Recommended Paper Structure

Based on the materials available, there are **three viable paper types**:

### Option A: Negative-Result / Replication Paper (★★★★★ RECOMMENDED)

**Title example**: *"The Signal Ceiling: A Multi-Paradigm Exhaustive Test of ML Return Prediction from OHLCV Features"*

**Structure**:
1. **Introduction**: The reproducibility crisis in financial ML; motivation for rigorous negative results
2. **Related Work**: ML in finance, GBDT vs DL, known predictability ceilings (Gu-Kelly-Xiu, LLG, Grinsztajn)
3. **Methodology**:
   - Data: 200 HK + 200 A-Share + 477 US stocks
   - Features: 9 OHLCV-derived indicators
   - 6 paradigms: DL (Mamba SSM), GBDT, Hybrid Ensemble, Feature Engineering, CS Ranking, Meta-Labeling
   - Protocol: walk-forward, held-out test, DM test, Ridge baseline
4. **Results**:
   - All 6 paradigms → IC ≈ 0.00 (centerpiece table)
   - IC paradox diagnosis (v3=0.14 → v5=0.049 → test=0.006)
   - Cross-market consistency (3-market table)
   - Volatility prediction as the only positive signal (QLIKE)
5. **Diagnostic Framework** (standalone contribution):
   - 5-step IC validation protocol
   - Application to Glaubenskrieg data
   - Recommendation for the field
6. **Discussion**: Why this result is informative even though it's negative; implications for the field
7. **Limitations**: Data constraints, frequency, no external data sources
8. **Conclusion**: Signal ceiling is real; ML practitioners should test signal existence before architecture

**Venue fit**: 
- **Best**: Workshop on reproducibility in ML (NeurIPS workshop, ICLR workshop) — these explicitly welcome negative results
- **Good**: *Journal of Financial Data Science* — publishes methodology papers, including negative findings
- **Possible**: *Quantitative Finance* — if framed as a methodology contribution
- **Unlikely**: Top ML conferences (NeurIPS, ICML, ICLR) — these rarely accept negative results
- **Unlikely**: Top finance journals (JFE, RFS, JF) — require positive economic contributions

### Option B: Methodology / Diagnostic Framework Paper (★★★★☆)

**Title example**: *"Five Steps to Trust Your Validation: A Diagnostic Protocol for Financial ML Signal Detection"*

**Structure**:
1. Introduction: The IC-overfitting problem
2. The 5-step diagnostic framework
3. Case study: Glaubenskrieg (applying the framework, showing how v3=0.14 was debunked)
4. Secondary case studies (if available)
5. Recommendations for practitioners
6. Open-source release of diagnostic toolkit

**Venue fit**:
- *Journal of Financial Data Science* (good fit)
- *Journal of Portfolio Management* (if focused on practitioner tools)
- ACM International Conference on AI in Finance (ICAIF) — methodology papers welcome

### Option C: Benchmark / Survey Paper (★★★☆☆)

**Title example**: *"What Works in ML for Stock Return Prediction? A Systematic Comparison of 6 Paradigms Across 3 Markets"*

**Structure**: Similar to Option A but framed as a systematic comparison rather than a negative result.

**Venue fit**: Lower-tier conferences, arXiv only. Top venues expect benchmarks to find *something* that works better.

### My recommendation: **Option A with Option B's diagnostic framework as the secondary contribution.**

The paper's strongest claim is: *We exhausted 6 approaches and found zero signal. Here's how we verified this finding isn't a methodological artifact. Here's how you can verify your own findings.* This is a coherent, defensible narrative.

---

## 6. Fatal Flaws Audit

### Potential fatal flaws that would cause desk rejection or R&R:

#### Flaw 1: "No signal" vs "Signal exists but we can't detect it"

**Severity**: 🟡 MEDIUM (addressable)

The paper claims "no detectable signal exists." But Kelly & Malamud (2025) argue that true population R² for market returns is ~20%, and the Limits-to-Learning Gap means OOS R² is systematically lower. Under this framework, IC=0 doesn't prove signal absence — it proves the LLG is too large for 200-stock, 3-year, daily-frequency data.

**Fix**: Frame the finding as "we cannot *detect* a signal with this data/model combination" rather than "no signal exists." Engage with the LLG framework explicitly.

#### Flaw 2: The volatility "success" contradicts the returns "failure" narrative

**Severity**: 🟡 MEDIUM (addressable)

The paper argues OHLCV features contain no predictive information, but volatility prediction works (QLIKE=-6.58, highly significant). A reviewer will ask: "If OHLCV features predict volatility, why can't they predict returns? The features are the same." The paper needs to explain why volatility is structurally different (variance clustering, long memory), and why this doesn't extend to the first moment.

**Fix**: Add a dedicated subsection on the first-moment vs second-moment predictability distinction, citing the Engle (1982) and Bollerslev (1986) literature.

#### Flaw 3: Cross-market comparison is not controlled

**Severity**: 🟡 MEDIUM (mitigated by transparency)

The three markets use different protocols: HK (200 stocks, 3yr, 3 windows), A-Share (200 stocks, 2.4yr, 6 windows), US (477 stocks, 10yr, 10 windows). The "cross-market consistency" claim is weakened by protocol inconsistency.

**Fix**: Either (a) re-run all markets with identical protocols (10-year, 10-window), or (b) explicitly acknowledge the protocol differences and argue that the consistency despite different protocols strengthens the finding.

#### Flaw 4: No transaction cost analysis for volatility strategy

**Severity**: 🟡 MEDIUM

The volatility backtest reports Sharpe ratios without transaction costs. Weekly rebalancing of top-50 by volume is not frictionless. If costs eliminate the ΔSharpe of +0.001 to +0.056, the economic significance evaporates.

**Fix**: Add transaction cost sensitivity analysis for the volatility backtest, or explicitly state that costs are not modeled and the reported Sharps are gross.

#### Flaw 5: Diebold-Mariano test applied to Sharpe, not predictions

**Severity**: 🟡 LOW-MEDIUM

The DM test in the FE LightGBM experiment (Phase 1) compares strategy Sharpe values, not prediction errors. DM is designed to compare forecast errors (MSE), not strategy Sharpe ratios. A reviewer may flag this as a methodological error.

**Fix**: Either (a) apply DM to squared prediction errors, not Sharpe, or (b) use a different test for Sharpe comparison (e.g., bootstrap Sharpe difference test by Ledoit & Wolf), or (c) justify the use of DM on economic loss functions (which Diebold himself has endorsed in later work).

#### Flaw 6: README claims contradict experimental results

**Severity**: 🔴 HIGH (must fix before submission)

The project README says: *"Glaubenskrieg predicts stock returns... from raw OHLCV data to live trade execution in one pipeline."* The experimental results show IC≈0, test Sharpe negative. If a reviewer reads the README and then the paper, this is a credibility-destroying contradiction.

**Fix**: Update README to reflect the project's actual findings. The README should describe the project as a research investigation, not a production-ready trading system.

#### Flaw 7: No statistical power analysis

**Severity**: 🟡 LOW-MEDIUM

The paper claims IC≈0. But with 750 effective samples (HK), what is the minimum detectable IC? If the minimum detectable IC is 0.05 at 80% power, then IC=0.006 falling below that threshold is expected, not informative. A power analysis would strengthen (or weaken) the "no signal" claim.

**Fix**: Add a power analysis: given N=750, α=0.05, what IC can be detected with 80% power? Report that the observed IC falls below the detectable threshold.

---

## 7. Venue Recommendation

### Tier 1: Best Fit (realistic acceptance chance 40-60%)

| Venue | Type | Why |
|-------|------|-----|
| **ICAIF** (ACM Int'l Conf on AI in Finance) | Conference | Welcomes methodology papers, negative results, practitioner focus. Perfect fit for diagnostic framework. |
| **Journal of Financial Data Science** | Journal | Explicitly publishes "negative results and replication studies." Methodology contributions valued. |
| **NeurIPS Workshop on ML in Finance** | Workshop | Lower bar than main conference. Welcomes "lessons learned" and negative results. |

### Tier 2: Possible Fit (realistic acceptance chance 15-35%)

| Venue | Type | Why |
|-------|------|-----|
| **ICAIF Workshop** | Workshop | Even lower bar than main ICAIF. |
| **Quantitative Finance** (journal) | Journal | Publishes empirical studies. But negative results need strong methodology framing. |
| **Journal of Portfolio Management** | Journal | Practitioner-focused. The diagnostic framework is practically useful. Negative-result framing won't fly. |
| **SSRN / arXiv only** | Preprint | Lowest bar. Gets the work out but no peer-review stamp. |

### Tier 3: Unlikely (acceptance chance <10%)

| Venue | Why not |
|-------|---------|
| **NeurIPS / ICML / ICLR** | Negative results rarely accepted. Need breakthrough positive finding. |
| **JFE / RFS / JF** | Require positive economic contributions. Negative results are not publishable. |
| **KDD / AAAI** | Require novelty in methods, not in negative empirical findings. |
| **Nature Machine Intelligence** | Requires paradigm-shifting finding. Negative result on one dataset is insufficient. |

### My recommendation: **Target ICAIF 2026 or Journal of Financial Data Science.** Prepare arXiv preprint simultaneously.

---

## 8. What Would Make This Paper Rejected (Real Reviewer Comments)

Here's what an actual reviewer would say, based on my reading:

> **Reviewer 1 (Reject):**
> "The paper reports that 6 ML paradigms fail to predict daily stock returns from OHLCV features. This finding is well-established in the literature (Gu-Kelly-Xiu 2020, McElfresh 2023, Grinsztajn 2022). The paper does not contribute new knowledge — it confirms what we already know. A confirmation paper requires either a substantially larger scale (e.g., 10,000+ stocks, 50+ markets) or a novel methodological insight that changes how we interpret existing results. The 5-step diagnostic framework is interesting but insufficient as a standalone contribution."

> **Reviewer 2 (Major Revision):**
> "The claim that 'no signal exists' is too strong. The Kelly & Malamud (2025) LLG framework shows that OOS R² systematically understates true predictability. The authors' IC≈0 result may reflect the LLG rather than signal absence. The paper must engage with this alternative interpretation. Additionally, the cross-market comparison uses inconsistent protocols (3 windows for HK, 10 for US), which undermines the cross-market consistency argument."

> **Reviewer 3 (Accept with minor revision):**
> "This is a well-executed negative result paper with strong methodological rigor. The IC paradox diagnosis is valuable and should be highlighted more prominently. I recommend the authors reframe the contribution as a diagnostic methodology with Glaubenskrieg as a case study, rather than as an empirical finding. The 5-step framework has standalone value for practitioners."

---

## 9. Pre-Submission Checklist

Before writing the paper, these must be addressed:

### Critical (would cause rejection if not fixed):

- [ ] Update README to remove "live trade execution" claims that contradict results
- [ ] Decide on paper type (negative-result vs methodology vs benchmark) → determines all writing choices
- [ ] Create consistent cross-market experimental protocol (or explicitly justify the differences)
- [ ] Add transaction cost analysis (or state it as a limitation)
- [ ] Fix DM test application (apply to prediction errors, not Sharpe)

### Important (would cause major revision requests):

- [ ] Engage with LLG framework (Kelly & Malamud 2025) as alternative interpretation
- [ ] Add statistical power analysis
- [ ] Create consistent citation format (.bib file)
- [ ] Compile all tables into publication-ready format (not .md tables)
- [ ] Generate all figures (empty figures/ directory)
- [ ] Write abstract that clearly states the contribution
- [ ] Resolve "exhaustive" vs "comprehensive" language

### Nice-to-have:

- [ ] Run permutation test (designed but not executed)
- [ ] Create Docker/replication package
- [ ] Add SHAP analysis for interpretability
- [ ] Run the protocol on a truly independent dataset (not just different market)
- [ ] Sensitivity analysis on window size, feature count, asset count

---

## 10. Final Verdict

### Can this be a paper? 
**Yes, but with caveats.**

The Glaubenskrieg project has produced one of the most rigorously documented negative results in financial ML that I've seen. The methodology is stronger than 90% of published financial ML papers (which typically report unverified positive results). The IC paradox diagnosis is genuinely useful. The cross-market validation is unusual and valuable.

However, the project's finding is fundamentally: *"We tested 6 approaches across 3 markets and confirmed the existing consensus that daily stock returns cannot be predicted from OHLCV-derived features."* This is a **confirmation**, not a **discovery**. Confirmation papers have a harder path to publication.

### What's the strongest possible paper?

**A methodology paper centered on the 5-step IC validation diagnostic framework, with Glaubenskrieg as the detailed case study.** This reframes the contribution from "we found nothing" (negative) to "here's how to verify whether you found something" (positive, methodological). The negative empirical result becomes evidence that the diagnostic framework works, rather than the paper's primary contribution.

### Bottom line:

| Dimension | Score | Notes |
|-----------|:-----:|-------|
| **Data quality** | 8/10 | Multi-market, consistent features, clean pipeline |
| **Methodology rigor** | 7/10 | Walk-forward + held-out test + Ridge baseline + DM test. Missing: permutation test, PBO, purged CV |
| **Honesty** | 9/10 | Negative results are front and center. Limitations acknowledged |
| **Novelty** | 4/10 | Negative result confirms known ceiling. Diagnostic framework is the novel piece |
| **Literature engagement** | 7/10 | Good coverage but citations not formalized, missing LLG engagement |
| **Paper readiness** | 0/10 | Paper directory is empty. No draft exists |
| **Publishability** | 5/10 | Publishable at workshops / specialist venues. Unlikely at top venues |

**Overall: The materials justify a workshop or specialist journal paper, focused on methodology rather than empirical findings. The paper has NOT been written — the paper/ directory is empty scaffolding. If written well (Option A + Option B hybrid), it could be accepted at ICAIF or Journal of Financial Data Science. If submitted as-is (raw progress.md), it would be desk-rejected from any venue.**

---

## Appendix A: Concrete Next Steps

### Week 1: Foundation
1. Write a one-paragraph abstract and get it reviewed by 2+ people
2. Choose the paper type definitively (recommendation: methodology case-study)
3. Create .bib file with all citations in consistent format
4. Fix README contradiction

### Week 2: Draft
1. Write Introduction + Related Work (2000 words)
2. Write Methodology (2000 words) — focus on the 5-step diagnostic framework
3. Create all figures (at minimum: main results table, IC paradox progression, cross-market comparison, volatility QLIKE)
4. Draft Results section (3000 words)

### Week 3: Polish
1. Write Discussion + Limitations (1500 words)
2. Run permutation test and add to results
3. Add transaction cost sensitivity analysis for vol backtest
4. Get external review from someone not involved in the project

### Week 4: Submit
1. Final formatting for target venue
2. Create replication package (pinned requirements.txt, Docker recipe)
3. Submit to arXiv + target venue simultaneously
