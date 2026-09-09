# Research: Novel AI Architectures 2024–2026

## Summary
The 2024–2026 period witnessed a Cambrian explosion of post-Transformer architectures driven by the need for sub-quadratic sequence modeling. Three major families dominate: **State Space Models** (Mamba-2/3, S4 lineage), **revived RNNs with expressive hidden states** (xLSTM, TTT, RWKV-6/7, Titans), and **hybrid SSM-Attention architectures** (Jamba, Samba, Zamba, Griffin). Additionally, **Kolmogorov-Arnold Networks** introduced a fundamentally different MLP alternative, and dedicated **time series architectures** (TimeMixer, PatchTST, iTransformer, xLSTM-Mixer, Mamba-MAT) have matured into a distinct sub-field. The convergence toward hybrids—mixing SSM efficiency with attention's recall—is the dominant design trend.

---

## Findings

### 1. State Space Model Family

**1.1 Mamba** (Dec 2023) — Selective SSM with input-dependent parameters (S6), linear O(N) complexity, hardware-aware CUDA kernel (scan instead of convolution). [arXiv:2312.00752]

**1.2 Mamba-2 / Structured State Space Duality (SSD)** (May 2024) — Showed Transformers and SSMs are dual formulations connected via structured matrices. Introduced SSD layer unifying SSM and linear attention with a single matrix multiplication primitive, enabling tensor-core-friendly computation. 2–8× faster than Mamba-1. [arXiv:2405.21060]

**1.3 Mamba-3** (Mar 2026) — Three improvements: (i) **Spectral discretization** using ZOH with learnable step sizes better suited for long sequences; (ii) **Complex-valued dynamics** enabling richer state transitions (previously only real diagonal); (iii) **MIMO (Multiple-Input Multiple-Output)** state expansion for higher state capacity without proportional memory increase. Inference-first design. [arXiv:2603.15569]

**1.4 H3 (Hungry Hungry Hippos)** (Dec 2022, ICLR 2023) — Pre-Mamba SSM that first showed SSMs can approach Transformer LM quality. Key insight: use two SSMs (one with diagonal state, one shift-SSM) interleaved with gating to capture both token interactions and long-range context. O(N log N) with FFT convolution. [arXiv:2212.14052]

| Architecture | Complexity | Key Innovation | Best Domain |
|---|---|---|---|
| Mamba-1 | O(N) | Selective SSM with input-dependent parameters | Language modeling, DNA, audio |
| Mamba-2/SSD | O(N) | SSM-Attention duality, tensor-core optimized | Language, long sequences |
| Mamba-3 | O(N) | Complex dynamics, MIMO, spectral discretization | Inference-first LLM |

### 2. RNN Revival

**2.1 xLSTM (Extended LSTM)** (May 2024, NeurIPS 2024) — Modernized LSTM with two blocks: (i) **sLSTM** (scalar LSTM) with exponential gating and memory mixing (new memory cell + normalization); (ii) **mLSTM** (matrix LSTM) with matrix-valued memory and parallelizable covariance update rule. mLSTM achieves O(N) complexity through parallel scan. Competitive with Transformers and Mamba at 1.3B-scale on SlimPajama. [NeurIPS 2024, arXiv:2405.04517]

**2.2 TTT (Test-Time Training)** (Jul 2024) — Makes the RNN hidden state a mini MLP that updates via gradient descent at test time. The hidden state "learns" to compress context; the update rule is self-supervised (reconstructing input from compressed representation). Linear O(N) complexity with expressive hidden states. Two variants: TTT-Linear (simpler) and TTT-MLP (more expressive). Matches Mamba at 1.3B scale. [arXiv:2407.04620, ICML 2025]

**2.3 Titans** (Jan 2025) — Three memory systems inspired by human memory: (i) **Core memory** (persistent, trained); (ii) **Context memory** (short-term, updated at test time via surprise-based mechanism); (iii) **Persistent memory** (learned task-specific parameters). Uses "learning to memorize at test time" with a neural memory module that updates via a meta-learning-like surprise metric. Outperforms Transformer + Mamba on long-context needle-in-haystack and BABILong. [arXiv:2501.00663]

**2.4 RWKV** (2023–2025) — Linear attention architecture reinventing RNNs. RWKV-5 (Eagle, 2024) added multi-headed matrix-valued states; RWKV-6 (Finch, 2024) introduced data-dependent time-mixing with dynamic decay; RWKV-7 (2025) further refined gating. O(N) training and O(1) inference per token. [arXiv:2305.13048]

### 3. Convolution & Signal-Processing Architectures

**3.1 Hyena** (2023, ICML 2023) — Replaces attention with implicit long convolutions parametrized by small MLPs + element-wise gating. Hyena operator: H(u) = Conv(Order-2 gated implicit filter). Sub-quadratic O(N log N). Successor architectures: HyenaDNA (genomics, 1M context). [arXiv:2302.10866]

**3.2 StripedHyena** (2023–2024) — Hybrid architecture alternating **Hyena blocks** (gated long convolutions) with **rotary attention blocks** in a striped pattern (16 Hyena + 16 attention layers for 7B model). First non-Transformer architecture competitive with open-source Transformers at 7B scale on both short and long context. [Together AI, arXiv:2403.17844]

### 4. Hybrid SSM-Attention Architectures ★ Most Active Area

**4.1 Jamba** (Mar 2024) — AI21 Labs. Interleaves Transformer layers + Mamba layers + Mixture-of-Experts (MoE). Architecture: 8× (Mamba → Transformer → MoE) blocks. 256K context window. Strong throughput gains over pure Transformer. Production-grade. [arXiv:2403.19887]

**4.2 Samba** (Jun 2024, ICLR 2025) — Microsoft. Simple stacked hybrid: Mamba → MLP → Sliding Window Attention → MLP. Trained on 3.2T tokens (Phi-3 data). 3.8B model beats Phi-3-mini on MMLU, GSM8K, HumanEval. Unlimited extrapolation capability (trained on 4K, tested on 1M). [arXiv:2406.07522]

**4.3 Zamba** (May 2024) — Zyphra. Mamba backbone + single shared attention module (not per-layer). Unique: attention module is shared across all layers, applied periodically. Compact 7B. Best non-Transformer at 7B scale. [arXiv:2405.16712]

**4.4 Griffin** (Feb 2024) — Google DeepMind. Mixes **RG-LRU (Real-Gated Linear Recurrent Unit)** with **local sliding window attention**. RG-LRU is a gated linear recurrence derived from the LRU framework. Key finding: global attention is unnecessary → local attention + recurrence suffices. Griffin-14B matches Llama-2 at 7× less training compute for equivalent perplexity. [arXiv:2402.19427]

**4.5 Mechanistic Design of Hybrids** (Mar 2024) — Systematic study: optimal ratio is ~1 attention layer per 3–4 SSM/convolution layers. Full attention every layer wastes compute; full SSM degrades recall. [arXiv:2403.17844]

**4.6 TransMamba** (2025) — Sequence-level hybrid: applies full Transformer to a compressed sequence representation, Mamba for detailed token-level processing. [arXiv:2503.24067]

### 5. Novel Mathematical Frameworks

**5.1 Kolmogorov-Arnold Networks (KAN)** (Apr 2024) — Replaces MLPs' fixed activation functions with **learnable activation functions on edges** (parametrized as B-splines). Inspired by Kolmogorov-Arnold representation theorem. Advantages: (i) much smaller networks for equivalent accuracy; (ii) interpretable—can discover symbolic formulas (e.g., rediscovering physical laws); (iii) outperforms MLPs on PDE solving and scientific tasks. Limitation: training is 10× slower than MLP. Extensions: UKAN (unbounded domains), FastKAN (speed optimizations). [arXiv:2404.19756]

**5.2 Liquid Neural Networks (LNNs)** (2020–2025) — Brain-inspired continuous-time neural ODEs where ODE parameters are input-dependent (liquid time-constants). Key properties: (i) adapt after training without fine-tuning; (ii) causally grounded; (iii) compact (19 neurons can drive autonomous car). 2024 extensions: LRC (Liquid-Resistance Liquid-Capacitance networks) with diagonal state-transition for parallel scan → O(TD) time. Liquid AI's LFM (Liquid Foundation Models, 2024) scaled LNNs to 1.3B–40B parameters, competitive with Transformers. [Liquid AI, arXiv:2403.08791, arXiv:2505.21717]

**5.3 Monarch Mixer (M2)** (Oct 2023, NeurIPS 2023) — Sub-quadratic architecture using **Monarch matrices** (structured sparse matrices generalizing FFT) along both sequence and model dimensions. Complexity: O(N^(3/2)) for N sequence, O(D^(3/2)) for D model dimension. Entirely GEMM-based—no attention, no recurrence, no convolution. Competitive with Transformers on BERT-style tasks. [arXiv:2310.12109]

### 6. Time Series Forecasting — Architectures Designed for or Successfully Applied

This is a rapidly maturing sub-field with its own architectural innovations:

**6.1 General-purpose architectures adapted for time series:**

| Architecture | Time Series Variant | Key Adaptation | Reference |
|---|---|---|---|
| Mamba | S-Mamba, Bi-Mamba+, MambaMixer | Bidirectional SSM scan for temporal modeling, channel mixing | arXiv:2403.19888 |
| xLSTM | xLSTM-Mixer | sLSTM + mLSTM combined with variate mixing | arXiv:2410.16928 |
| Mamba+Transformer | MAT (Mamba-Attention-Transformer) | Mamba for long-range + attention for short-range | arXiv:2409.08530 |
| Mamba+Transformer | SST (Multi-Scale Hybrid Mamba-Transformer Experts) | Multi-scale decomposition + MoE routing | arXiv:2404.14757 |
| LLM | LLM-Mixer, Time-LLM | Frozen LLM backbone + time series tokenization/patching | arXiv:2410.11674 |

**6.2 Purpose-built time series architectures:**

**6.2.1 PatchTST** (ICLR 2023) — Channel-independent patching of time series + Transformer. De facto standard. Key insight: segment time series into patches (like ViT for images) → O(L²/P²) attention. [arXiv:2211.14730]

**6.2.2 iTransformer** (ICLR 2024) — Inverts Transformer: applies attention across **variate (channel) tokens** rather than temporal tokens. Each variate is one token; temporal dependencies captured by FFN. Strong on multivariate forecasting. [ICLR 2024]

**6.2.3 TimeMixer** (ICLR 2024) — Fully MLP-based architecture. Core innovation: **Past-Decomposable-Mixing (PDM)** decomposes historical series into trend + seasonal at multiple scales, then mixes across scales. **Future-Multipredictor-Mixing (FMM)** ensembles predictions from different scales. State-of-the-art on long-term forecasting. [ICLR 2024]

**6.2.4 TimesNet** (ICLR 2023) — Transforms 1D time series into 2D tensors by discovering multiple periods (FFT → top-k frequencies). Then applies 2D convolution (Inception blocks) to capture both intra-period and inter-period variations. [ICLR 2023]

**6.2.5 Is Mamba Effective for TSF?** (Mar 2024) — Systematic evaluation: Mamba competitive with SOTA Transformers on time series, especially with bidirectional scan + channel independence. Simple S-Mamba equals or beats complex Transformers. [arXiv:2403.11144]

**6.2.6 xLSTM-Mixer** (Oct 2024) — Combines xLSTM's sequential modeling with a linear forecast shared across variates, refined by scalar sLSTM/mLSTM memories for joint time-variate interaction. [arXiv:2410.16928]

### 7. Other Notable Architectures

**7.1 RetNet (Retentive Network)** (Jul 2023) — Three-in-one computation: parallel (training), recurrent (low-cost inference), chunkwise recurrent (hybrid). Retention mechanism = attention without softmax + causal decay mask. O(N) inference. Strong LM performance at scale. [arXiv:2307.08621]

**7.2 Mega / MegaByte** (2023) — Gated attention with exponential moving average (EMA). Chunked model architecture with separate patch and byte-level models. [Meta FAIR]

**7.3 Based (Linear Attention)** (2023–2024) — Combines linear attention (Taylor expansion of softmax) with sliding window attention. Simple, fast, competitive at 1.3B.

### 8. Architecture Landscape Map

```
                    Linear O(N) ◄─────────────────────────► Quadratic O(N²)
                          │                                        │
    ┌─────────────────────┼──────────────────────┐                 │
    │                     │                      │                 │
  Pure SSM           RNN Revival           SSM+Attn Hybrid     Transformer
  ─────────          ───────────           ───────────────     ───────────
  Mamba-1/2/3        xLSTM (2024)          Jamba (2024)        GPT-4
  S4/H3 (2022)       TTT (2024)            Samba (2024)        Llama
  RG-LRU             RWKV-6/7 (2024-25)    Zamba (2024)        Mistral
                     Titans (2025)         Griffin (2024)
                                           TransMamba (2025)

    ┌─────────────────────┼──────────────────────┐
    │                     │                      │
  Convolution         Mathematical           Memory-Based
  ────────────        ─────────────          ────────────
  Hyena (2023)        KAN (2024)             Titans (2025)
  StripedHyena        Monarch Mixer          TTT (2024)
                      Liquid NN/LFM
```

### 9. Compute Complexity Summary

| Architecture | Train Complexity | Inference Complexity | Memory (KV-cache) |
|---|---|---|---|
| Transformer (vanilla) | O(N²·D) | O(N²) per step | O(N·D) |
| Mamba-1 | O(N·D) | O(D) per step | O(D·state_dim) |
| Mamba-2 (SSD) | O(N·D) | O(D) per step | O(D·state_dim) |
| Mamba-3 | O(N·D) | O(D) per step | O(D·state_dim×MIMO) |
| xLSTM (mLSTM) | O(N·D) | O(D) per step | O(D²) matrix state |
| TTT | O(N·D) | O(D) per step | O(D²) MLP state |
| Titans | O(N·D) | O(D) per step | O(D²) neural memory |
| RWKV-6/7 | O(N·D) | O(D) per step | O(D) |
| Griffin (RG-LRU + local attn) | O(N·D) | O(D) per step | O(W·D) for window W |
| Hyena | O(N log N) | O(N log N) or O(N) | O(N) logN |
| Jamba (SSM+Attn+MoE) | O(N·D) (dominated by attn layers) | O(D) per step | Mixed |
| KAN | O(N·D·G) G=grid | O(D·G) | N/A |
| Monarch Mixer | O(N^(3/2)·D^(3/2)) | Same | N/A |

### 10. Key Trends (2024–2026)

1. **Hybrid is winning**: Pure architectures (SSM-only, RNN-only) consistently lose to hybrids that combine SSM efficiency with strategic attention. The optimal ratio appears to be 1 attention layer per 3–4 non-attention layers.

2. **Test-time learning/compute**: TTT, Titans, and Liquid NNs all invest computation at inference time to adapt. This mirrors the "inference-time compute scaling" trend in LLMs.

3. **State expansion**: Mamba-3's MIMO, xLSTM's matrix memory, TTT's MLP hidden state, Titans' neural memory — all expand state capacity beyond simple vectors without proportional compute increase.

4. **Time series as proving ground**: New architectures are increasingly validated on time series first (cheaper training) before scaling to language. xLSTM-Mixer, Mamba-TSF, SST, MAT all target forecasting.

5. **Hardware-algorithm co-design**: Mamba-2/3, FlashAttention lineage, Monarch Mixer all optimize for specific hardware primitives (tensor cores, GEMM, scan operations).

6. **Convergence**: SSD duality shows SSM ≈ linear attention; Hyena ≈ implicit convolution ≈ linear attention. The field is converging on a unified framework of sub-quadratic sequence mixing operators.

---

## Sources

### Kept (Key Papers with Arxiv IDs)

- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752, Dec 2023. *Foundation of selective SSM family.*
- **Mamba-2**: Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality," arXiv:2405.21060, May 2024. *Unified SSM-Attention framework.*
- **Mamba-3**: Lahoti, Li, Chen et al., "Mamba-3: Improved Sequence Modeling using State Space Principles," arXiv:2603.15569, Mar 2026. *Latest SSM with complex dynamics and MIMO.*
- **xLSTM**: Beck et al., "xLSTM: Extended Long Short-Term Memory," NeurIPS 2024, arXiv:2405.04517, May 2024. *Modernized LSTM competitive with Transformers.*
- **TTT**: Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States," ICML 2025, arXiv:2407.04620, Jul 2024. *RNNs that learn at inference time.*
- **Titans**: Behrouz et al., "Titans: Learning to Memorize at Test Time," arXiv:2501.00663, Jan 2025. *Multi-memory architecture with surprise-based updates.*
- **H3**: Fu, Dao et al., "Hungry Hungry Hippos: Towards Language Modeling with State Space Models," ICLR 2023, arXiv:2212.14052, Dec 2022. *First SSM competitive on language.*
- **Hyena**: Poli et al., "Hyena Hierarchy: Towards Larger Convolutional Language Models," ICML 2023, arXiv:2302.10866. *Implicit long convolutions replacing attention.*
- **StripedHyena**: Together AI, "Mechanistic Design and Scaling of Hybrid Architectures," arXiv:2403.17844, Mar 2024. *Alternating Hyena + Attention blocks at 7B scale.*
- **Griffin**: De, Smith et al. (DeepMind), "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models," arXiv:2402.19427, Feb 2024. *RG-LRU + local attention.*
- **Jamba**: AI21 Labs, "Jamba: A Hybrid Transformer-Mamba Language Model," arXiv:2403.19887, Mar 2024. *Transformer-Mamba-MoE hybrid, production grade.*
- **Samba**: Ren et al. (Microsoft), "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling," ICLR 2025, arXiv:2406.07522, Jun 2024. *Mamba + Sliding Window Attention.*
- **Zamba**: Zyphra, "Zamba: A Compact 7B SSM Hybrid Model," arXiv:2405.16712, May 2024. *Mamba backbone + single shared attention.*
- **RWKV**: Peng et al., "RWKV: Reinventing RNNs for the Transformer Era," EMNLP 2023, arXiv:2305.13048. *Linear attention RNN architecture.*
- **RetNet**: Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models," arXiv:2307.08621, Jul 2023. *Three-paradigm retention mechanism.*
- **KAN**: Liu et al., "KAN: Kolmogorov-Arnold Networks," arXiv:2404.19756, Apr 2024. *Learnable activation functions on edges.*
- **Monarch Mixer**: Fu et al., "Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture," NeurIPS 2023, arXiv:2310.12109, Oct 2023. *GEMM-only sub-quadratic architecture.*
- **TimeMixer**: Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting," ICLR 2024. *MLP-based multiscale time series architecture.*
- **PatchTST**: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers," ICLR 2023, arXiv:2211.14730. *Channel-independent patching.*
- **iTransformer**: Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting," ICLR 2024. *Attention across variates.*
- **xLSTM-Mixer**: Anonymous, "xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories," arXiv:2410.16928, Oct 2024. *xLSTM for time series.*
- **MAT**: "Integration of Mamba and Transformer for Long-Short Range Time Series Forecasting," arXiv:2409.08530, Sep 2024. *Mamba-Attention hybrid for weather.*
- **SST**: "SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series Forecasting," arXiv:2404.14757, Apr 2024. *Multi-scale SSM-Attention-MoE.*
- **Is Mamba Effective for TSF?**: "Is Mamba Effective for Time Series Forecasting?", arXiv:2403.11144, Mar 2024. *Systematic Mamba TSF evaluation.*
- **Liquid NNs**: Hasani et al., multiple papers (Nature MI 2020–2025), Liquid AI blog. *Continuous-time adaptive neural ODEs.*
- **TransMamba**: "TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model," arXiv:2503.24067, Mar 2025. *Sequence-level hybridization.*

### Dropped
- General Mamba surveys (too broad, secondary sources) — kept only the systematic "Is Mamba Effective for TSF" paper.
- Medium/Blog posts (e.g., Nebius, Latent Space podcast) — used for context but not as primary sources.
- Quantamagazine KAN article — popular science, not technical source; used original arxiv paper instead.
- HuggingFace paper pages — redundant with arxiv.
- GitHub READMEs — only used for release notes (StripedHyena, Samba).

---

## Gaps

1. **Empirical benchmarks across architectures at equivalent compute**: No single paper benchmarks xLSTM vs Mamba-3 vs TTT vs Titans at equivalent training FLOPs. Each paper picks different baselines and scales. A controlled "Chinchilla-optimal" comparison across all architectures remains missing.

2. **Time series performance of Mamba-3 / TTT / Titans**: These are very new (2025–2026); their time series forecasting performance has not been benchmarked. xLSTM-Mixer and Mamba-S-Mamba exist but only for older versions.

3. **KAN for time series**: Several preprints exist (KAN for forecasting) but no systematic SOTA comparison. Performance vs TimeMixer/PatchTST remains unclear.

4. **Liquid Foundation Models (LFM) detailed architecture**: Liquid AI's 2024 LFM models are proprietary. Architecture details remain partially disclosed. The arXiv papers cover the theoretical framework but not the scaled model specifics.

5. **RWKV-7 details**: As of the knowledge cutoff, RWKV-7 exists but detailed technical reports may have been released after the cutoff. Reliable details come from RWKV-6 (Finch).

## Supervisor Coordination
No decisions required. Research is self-contained and complete for the stated scope.
