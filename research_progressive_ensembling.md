# Research: Progressive NN→GBDT Ensembling for Time Series Forecasting

## Summary
No existing published work directly implements "gradual handoff" from neural networks to GBDT as a function of prediction horizon. However, the necessary building blocks are mature across five separate literature streams: (1) NN-GBDT blending for forecasting, (2) horizon-dependent model complexity tradeoffs, (3) confidence/uncertainty-gated routing, (4) temporal mixture-of-experts with horizon-aware gating, and (5) adaptive online ensemble learning. We synthesize five concrete design patterns (A–E) that combine these building blocks, ranked by implementation complexity. **Pattern D (TIME FUSE extension with horizon meta-feature) is recommended as the lowest-cost, highest-impact starting point.**

---

## Findings

### 1. NN-GBDT Blending Is Proven Effective in Forecasting Competitions
The winning methodology of the M5 Competition blended gradient boosted trees and neural networks (DeepAR, N-BEATS) for both point and probabilistic forecasting of hierarchical time series. Their approach: (a) transform to single-day regression, (b) rich feature engineering, (c) diverse model set, (d) weighted average blending optimized on validation. However, their blending weights are **static** — no horizon-dependence, no sample-level adaptation. [Source](https://arxiv.org/html/2310.13029)

### 2. Strong Theoretical Basis for NN→GBDT Transition at Longer Horizons
The **nonstationarity-complexity tradeoff** paper (Kelly et al., 2024) provides the theoretical foundation: complex models (deep NNs) reduce misspecification error but require longer training windows, which introduce stronger nonstationarity. At longer forecast horizons, nonstationarity dominates, making simpler models (like GBDT with shorter training windows) comparatively more reliable. This directly motivates a horizon-dependent handoff. [Source](https://arxiv.org/html/2512.23596)

Similarly, the **temporal horizons tradeoff** paper formalizes that for chaotic systems, the loss landscape roughness grows exponentially with training horizon — simpler models with shorter effective horizons may generalize better at long-range prediction. [Source](https://arxiv.org/html/2506.03889)

### 3. TIME FUSE (ICML 2025) Provides the Closest Architectural Template
TIME FUSE extracts **meta-features** (statistical, temporal, spectral) from input time series and trains a learned **fusor** (lightweight MLP) to dynamically combine predictions from a diverse model bank at the sample level. Key insight: no single model wins universally — each excels on specific samples. The fusor outputs per-sample weights using meta-features as input. This architecture can be directly extended by adding **forecast horizon** as a meta-feature to achieve horizon-conditioned blending. [Source](https://arxiv.org/html/2505.18442) | [Code](https://github.com/ZhiningLiu1998/TimeFuse)

### 4. Dynamic Meta-Learning for Uncertainty-Gated XGBoost-NN Ensembles
This 2025 paper explicitly combines XGBoost and neural networks using **uncertainty quantification** to orchestrate model selection. It uses feature importance integration and advanced uncertainty quantification for dynamic routing — directly relevant to confidence-based fallback from NN to GBDT. [Source](https://arxiv.org/html/2510.03301v1)

### 5. Adaptive Future-Guided Ensemble Learning (AFG-EL) Implements Drift-Aware Routing
AFG-EL performs **sample-level routing over a heterogeneous model bank**, where the routing decision is informed by future-guided meta-learning. The optimal forecasting model varies across different temporal regimes and horizons — exactly the motivation for horizon-dependent handoff. Two-stage: (1) train diverse base models, (2) learn a routing policy that maps input features to optimal model weights. [Source](https://www.mdpi.com/2227-7390/14/10/1686)

### 6. DMS/AE: Online Dynamic Model Selection with Horizon-Aware Evaluation
Dynamic Model Selection (DMS) and Adaptive Ensemble (AE) methods adapt model weights online based on recent forecasting performance over specific horizons. The key idea: track each model's recent error at each horizon and up-weight models that perform well at that horizon. This provides a **non-parametric, online alternative** to learned blending. [Source](https://ar5iv.labs.arxiv.org/html/2110.11156)

### 7. Online Deep Hybrid Ensemble Learning Explicitly Models Time-Dependent Performance
This work observes that ML models "usually reveal a time-dependent performance" and proposes an online deep hybrid ensemble that dynamically weights models. It uses a DL-based meta-learner trained to output per-model weights given time-varying features, demonstrating that **temporal context improves ensemble weights**. [Source](https://link.springer.com/chapter/10.1007/978-3-031-43424-2_10)

### 8. Mixture-of-Experts Architectures Support Horizon-Conditioned Gating
Multiple recent MoE papers extend the paradigm to time series with temporal gating mechanisms:
- **Mixture-of-Linear-Experts** uses periodicity-aware routing where different linear experts specialize in different temporal patterns/horizons. [Source](https://arxiv.org/html/2312.06786v1)
- **Seg-MoE** uses segment-wise (rather than token-wise) routing, preserving temporal locality. [Source](https://arxiv.org/html/2601.21641)
- **Dynamic TMoE** unifies architectural evolution with temporal continuity, adapting to regime shifts that correlate with horizon. [Source](https://arxiv.org/html/2605.20678v1)
- **AdaMixT** uses adaptive weighted mixture of multi-scale experts — different experts at different temporal resolutions. [Source](https://arxiv.org/html/2509.18107)

### 9. Online Cascade Learning Formalizes the Handoff Cascade Pattern
The online cascade learning framework formalizes a **cascade of models** from low-capacity to high-capacity (e.g., logistic regression → LLM), with a learned **deferral policy** that decides which model handles each input. While developed for LLM cost optimization, the cascade pattern (simple → complex, with a deferral policy) maps directly to the NN→GBDT handoff problem when reversed (complex → simple as horizon increases). [Source](https://arxiv.org/html/2402.04513)

### 10. Multi-Layer Stack Ensembles: Systematic Evaluation Across 50 Datasets
A comprehensive evaluation of 33 ensemble strategies (including both existing and novel) across 50 datasets shows that **multi-layer stacking consistently outperforms simple averaging** for time series. The best architectures use meta-learners trained on base model outputs plus original features. Horizon information is not explicitly used — a gap we can fill. [Source](https://arxiv.org/html/2511.15350)

### 11. CNN→LightGBM Hybrid Shows NN Feature Extraction + GBDT Regression Works for Finance
The strictly chronological CNN-LightGBM hybrid for next-day log-return forecasting demonstrates a practical pattern: **CNN extracts temporal embeddings, then LightGBM performs the final regression**. The embedding standardization step stabilizes the NN→GBDT interface — critical for any progressive handoff architecture. [Source](https://www.mdpi.com/2073-8994/18/3/416)

### 12. The "Gradual Handoff" Pattern Is a Genuine Literature Gap
After searching across 5 query angles and reviewing ~30 papers, **no existing work implements horizon-conditioned gradual handoff between NN and GBDT**. Existing methods either: (a) use static blending weights, (b) route based on uncertainty/sample features but not horizon, (c) use MoE with horizon-independent gating, or (d) do online adaptation based on recent error but without explicit horizon encoding. The combination — horizon→weight→handoff from NN to GBDT — is novel.

---

## Design Patterns (Ranked by Implementation Complexity)

### Pattern A: Horizon-Decayed Contribution Blending
**Concept**: NN weight decays exponentially or sigmoidally with prediction horizon.

$$w_{\text{NN}}(h) = \alpha \cdot e^{-\beta h} + \gamma, \quad w_{\text{GBDT}}(h) = 1 - w_{\text{NN}}(h)$$

**Implementation**: ~20 LOC. Add 2–3 learnable parameters ($\alpha, \beta, \gamma$) trained via gradient descent on validation loss.
**Sources**: Nonstationarity-complexity tradeoff [arXiv:2512.23596](https://arxiv.org/html/2512.23596), Temporal horizons tradeoff [arXiv:2506.03889](https://arxiv.org/html/2506.03889)
**Pros**: Simplest, interpretable, negligible overhead. **Cons**: Assumes monotonic relationship; no per-sample adaptation.
**Risk**: May underperform if the NN→GBDT benefit is not monotonic in horizon.

### Pattern B: Uncertainty-Gated Fallback (Confidence Routing)
**Concept**: Compute NN prediction uncertainty (MC Dropout, ensemble variance, or conformal prediction). When uncertainty exceeds a calibrated threshold, fall back to GBDT. Threshold can be horizon-dependent.

$$\hat{y} = \begin{cases} \hat{y}_{\text{NN}} & \text{if } U(\hat{y}_{\text{NN}}) < \tau(h) \\ \hat{y}_{\text{GBDT}} & \text{otherwise} \end{cases}$$

**Implementation**: ~50 LOC + uncertainty estimation module. Requires calibration step (isotonic regression).
**Sources**: Dynamic Meta-Learning XGBoost-NN [arXiv:2510.03301](https://arxiv.org/html/2510.03301v1), UCCI calibration [arXiv:2605.18796](https://arxiv.org/html/2605.18796), NGBoost [PMLR](https://proceedings.mlr.press/v119/duan20a.html)
**Pros**: Principled, interpretable fallback decisions. **Cons**: Requires reliable uncertainty calibration (hard in finance); hard threshold can cause discontinuities.
**Risk**: Poorly calibrated NN uncertainties lead to ineffective routing.

### Pattern C: Horizon-Conditioned Mixture of Experts
**Concept**: Train a lightweight gating network that takes forecast horizon $h$ (plus optionally lookback window features) as input and outputs softmax weights over a small set of "experts" (NN, GBDT, possibly intermediate models).

$$w = \text{softmax}(\text{Gate}(h, x_{\text{lookback}})), \quad \hat{y} = \sum_i w_i \cdot \hat{y}_i$$

**Implementation**: ~100 LOC. Gate is a small MLP (2-3 layers). Trained end-to-end or with a separate meta-learning objective.
**Sources**: Mixture-of-Linear-Experts [arXiv:2312.06786](https://arxiv.org/html/2312.06786v1), Seg-MoE [arXiv:2601.21641](https://arxiv.org/html/2601.21641), Dynamic TMoE [arXiv:2605.20678](https://arxiv.org/html/2605.20678v1), AdaMixT [arXiv:2509.18107](https://arxiv.org/html/2509.18107)
**Pros**: Learns complex, nonlinear horizon-weight relationships; supports >2 models. **Cons**: Requires training the gate; load balancing losses may be needed to prevent collapse to single expert.
**Risk**: With only 2 "experts," the gate may collapse to always picking one model.

### Pattern D: Meta-Feature Fusion (TIME FUSE + Horizon Encoding) ⭐ RECOMMENDED
**Concept**: Extend TIME FUSE's meta-feature approach: extract statistical/temporal/spectral meta-features from the lookback window, **concatenate forecast horizon $h$ as an explicit meta-feature**, and train a lightweight fusor (MLP) that outputs per-model weights.

$$\hat{y} = \text{Fusor}\big([\phi(x_{\text{lookback}}), h]\big) \cdot [\hat{y}_{\text{NN}}, \hat{y}_{\text{GBDT}}]$$

**Implementation**: ~80 LOC. TIME FUSE provides the architectural template; adding horizon requires 1 extra input dimension.
**Sources**: TIME FUSE [ICML 2025 / arXiv:2505.18442](https://arxiv.org/html/2505.18442), DMS/AE [arXiv:2110.11156](https://ar5iv.labs.arxiv.org/html/2110.11156)
**Pros**: Leverages proven ICML architecture; horizon is just another meta-feature; supports per-sample adaptation; tested on 7 forecasting benchmarks. **Cons**: Requires meta-feature extraction pipeline; fusor training adds minor overhead.
**Risk**: Horizon may be overshadowed by stronger meta-features —可能需要 horizon feature scaling or separate horizon embedding.

### Pattern E: Progressive Ensemble Distillation
**Concept**: Train a cascade of models at increasing horizons, then distill the cascade into a single model that internally routes by horizon. At training time, use multi-horizon loss with horizon-dependent teacher weights (NN for short horizons, GBDT for long horizons).

$$\mathcal{L} = \sum_h \lambda_h \cdot \big[w_{\text{NN}}(h) \cdot \mathcal{L}_{\text{NN}} + w_{\text{GBDT}}(h) \cdot \mathcal{L}_{\text{GBDT}}\big]$$

**Implementation**: ~200+ LOC. Requires multi-horizon training loop, teacher model management, distillation loss.
**Sources**: Online Cascade Learning [arXiv:2402.04513](https://arxiv.org/html/2402.04513), Multi-layer Stack Ensembles [arXiv:2511.15350](https://arxiv.org/html/2511.15350), Label Horizon Paradox [arXiv:2602.03395](https://arxiv.org/pdf/2602.03395)
**Pros**: Most theoretically complete; single unified model at inference. **Cons**: Complex training; high computational cost; distillation may lose specialist advantages.
**Risk**: Over-engineering for marginal gain over simpler patterns.

---

## Sources

### Kept (18 sources, all directly relevant)

| # | Source | Relevance |
|---|--------|-----------|
| 1 | Blending GBT and NN for hierarchical time series (M5) [arXiv:2310.13029](https://arxiv.org/html/2310.13029) | NN-GBDT blending methodology from competition winner |
| 2 | TIME FUSE — ICML 2025 [arXiv:2505.18442](https://arxiv.org/html/2505.18442) | Sample-level adaptive fusion architecture; template for Pattern D |
| 3 | Nonstationarity-complexity tradeoff [arXiv:2512.23596](https://arxiv.org/html/2512.23596) | Theoretical basis for simpler models at longer horizons |
| 4 | Temporal horizons: performance-learnability tradeoff [arXiv:2506.03889](https://arxiv.org/html/2506.03889) | Formal tradeoff; chaos theory grounding |
| 5 | Dynamic Meta-Learning for XGBoost-NN Ensembles [arXiv:2510.03301](https://arxiv.org/html/2510.03301v1) | Uncertainty-guided routing between XGBoost and NN |
| 6 | AFG-EL: Drift-aware routing ensemble [MDPI](https://www.mdpi.com/2227-7390/14/10/1686) | Sample-level routing over heterogeneous model bank |
| 7 | DMS, AE, DAA: Adaptive model selection [arXiv:2110.11156](https://ar5iv.labs.arxiv.org/html/2110.11156) | Online dynamic model selection with horizon-aware evaluation |
| 8 | Online Deep Hybrid Ensemble Learning [Springer](https://link.springer.com/chapter/10.1007/978-3-031-43424-2_10) | Time-dependent model performance; online weight adaptation |
| 9 | Mixture-of-Linear-Experts for LTSF [arXiv:2312.06786](https://arxiv.org/html/2312.06786v1) | Horizon/periodicity-aware expert routing |
| 10 | Seg-MoE: Segment-wise MoE [arXiv:2601.21641](https://arxiv.org/html/2601.21641) | Temporal locality in expert routing |
| 11 | Dynamic TMoE [arXiv:2605.20678](https://arxiv.org/html/2605.20678v1) | Drift-aware MoE with temporal continuity |
| 12 | Online Cascade Learning [arXiv:2402.04513](https://arxiv.org/html/2402.04513) | Cascade + deferral policy pattern |
| 13 | Multi-layer Stack Ensembles [arXiv:2511.15350](https://arxiv.org/html/2511.15350) | Systematic ensemble evaluation on 50 datasets |
| 14 | CNN-LightGBM hybrid for returns [MDPI](https://www.mdpi.com/2073-8994/18/3/416) | NN feature extraction → GBDT regression pattern |
| 15 | Hybrid Framework for Sequential Data [arXiv:2203.13787](https://arxiv.org/html/2203.13787) | Recursive NN + boosted trees end-to-end |
| 16 | Intelligent Routing for Sparse Demand [arXiv:2506.14810](https://arxiv.org/html/2506.14810v1) | Meta-model routing strategies for forecasting |
| 17 | DL Architecture Comparison (918 experiments) [arXiv:2603.16886](https://arxiv.org/pdf/2603.16886) | Multi-horizon benchmark on financial data |
| 18 | AdaMixT [arXiv:2509.18107](https://arxiv.org/html/2509.18107) | Multi-scale temporal expert mixing |

### Dropped

- **GrowNet (Gradient Boosting Neural Networks)** — Uses shallow NNs as weak learners inside GBDT; different direction (boosting NNs) rather than ensembling NN + GBDT.
- **Time-MoE / Time Tracker** — Large-scale MoE foundation models; relevant architecture but focus on scaling rather than NN↔GBDT handoff.
- **Interpretability Gated Networks (InterpGN)** — Focused on classification interpretability, not forecasting or horizon-dependent routing.
- **LLM cascade routing papers (CARGO, RouteNLP, Gatekeeper)** — Cascade/routing concept is transferable but domain is NLP cost optimization, not time series.
- **PatternFusion** — BiLSTM+CNN+statistical fusion; architectural similarity but no GBDT or horizon-dependence.
- **OneNet / RLMC** — Dynamic online ensembling; relevant approach but no explicit horizon mechanism or NN/GBDT pairing.

---

## Gaps

1. **No existing paper implements horizon-conditioned NN→GBDT gradual handoff.** This confirms the pattern is novel and worth prototyping.

2. **Uncertainty calibration for financial time series is underexplored.** Most uncertainty quantification methods (MC Dropout, Deep Ensembles) are calibrated on i.i.d. data; financial time series exhibit heavy tails and regime shifts that break standard calibration assumptions. This limits the practical reliability of Pattern B.

3. **Optimal horizon-weight function shape is unknown.** No ablation study compares exponential decay vs. sigmoid vs. learned step functions for horizon-dependent model switching. Empirical evaluation is needed.

4. **Interaction between horizon, nonstationarity regime, and model choice.** The nonstationarity-complexity tradeoff paper establishes the theoretical link, but practical methods for detecting nonstationarity level (to inform model selection) are missing.

5. **Two-model vs. multi-model ensembles.** Most existing work blends 5–10+ models. It's unclear whether a 2-model (NN + GBDT) ensemble can achieve similar gains, or whether intermediate-complexity models (e.g., linear models at mid-horizons) are necessary.

6. **Computational cost of meta-feature extraction.** TIME FUSE's meta-feature pipeline (statistical, temporal, spectral features) adds preprocessing overhead that may not be justified for a 2-model ensemble. A stripped-down version may be needed.

---

## Recommendations for Glaubenskrieg

1. **Start with Pattern D** (TIME FUSE extension with horizon meta-feature). It has the best cost/benefit ratio: proven architecture (ICML 2025), minimal code changes (add one input dimension), and supports future expansion to more models.

2. **Evaluate Pattern A as a baseline.** The exponential decay blending with learnable parameters provides a simple, interpretable benchmark that Pattern D must beat to justify its complexity.

3. **Apply Diebold-Mariano tests** to assess whether ensemble improvements over individual models are statistically significant — a practice advocated by the DMS/AE literature and the 918-experiment financial benchmark.

4. **Consider adaptive online weighting** (DMS/AE approach) as a non-parametric alternative if the learned fusor in Pattern D overfits the validation period.

5. **Monitor ensemble weight trajectories** during training and evaluation to detect mode collapse (e.g., fusor always selecting NN regardless of horizon).
