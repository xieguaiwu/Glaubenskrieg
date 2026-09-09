# Research: SOTA AI Architectures for Financial Time Series / Quantitative Trading

## Summary
No single deep learning architecture has **convincingly and consistently** beaten well-tuned GBDT (XGBoost/LightGBM/CatBoost) on tabular financial return data in a leakage-free, strictly chronological protocol. The best empirical results come from **hybrid approaches** where a neural encoder (CNN/LSTM/Mamba) learns representations fed into a GBDT regressor, or from **multi-modal fusion** where structured price data is combined with text/graph signals. Pure DL on price-only daily return prediction yields at most ~2–3% OOS-R² improvement vs. a zero-return null — statistically significant but economically marginal.

---

## Findings

### 1. Hybrid Neural + GBDT Architectures (Most Promising Paradigm)

1.1 **CNN → LightGBM hybrid with explicit embedding standardization** (Bai et al., *Symmetry* 2026). One-sided CNN encodes multivariate OHLCV → last-step embedding → channel-wise standardization → LightGBM regressor. On NIFTY50 daily log-return prediction: OOS-R² = 0.0285 vs. Naive0 (p<0.01), OOS-R² = 0.403 vs. Persist. DM-tests show statistically significant superiority over standalone CNN, LightGBM, LSTM-LightGBM, and TimesNet. Key design elements: (a) embedding standardization critical for stable GBDT interface, (b) wavelet denoising on close price for return feature construction, (c) recency-aware fit-latest-W standardization. [Source](https://www.mdpi.com/2073-8994/18/3/416)

1.2 **DNN + TabNet hybrid** (PMC 2024). Hybrid DNN and TabNet model for stock return prediction. Claims neural network + attention-based tabular model can outperform single models on specific financial prediction tasks, though evaluation protocol rigor varies. [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11639136/)

1.3 **Generic CNN-LightGBM for fault diagnosis** pattern is transferable to finance. The general paradigm of CNN feature extraction → LightGBM classification/regression is well-established across domains and transfers naturally to financial prediction. [Source](https://www.mdpi.com/2073-8994/18/3/416) (references bearing fault diagnosis work)

### 2. GNN + Mamba / Transformer Fusion for Stock Graphs

2.1 **SAMBA: Bidirectional Mamba + Adaptive Graph Convolution** (Mehrabian et al., arxiv 2410.03707). BI-Mamba captures temporal dependencies; AGC models daily feature interactions as adaptive graph. On NASDAQ/NYSE/DJIA (2010–2023): IC improvements of 85% (NASDAQ), 36% (NYSE), 33% (DJIA) over second-best. Near-linear complexity — much faster than Transformers. 167K params, 0.11M MACs. [Source](https://arxiv.org/pdf/2410.03707v2)

2.2 **HIGSTM: Hierarchical Information-Guided Spatio-Temporal Mamba** (arxiv 2503.11387). Multiscale Mamba framework that captures both global market dynamics and individual stock interdependencies. Addresses limitations of vanilla Mamba for stock prediction. [Source](https://arxiv.org/html/2503.11387v1)

2.3 **MaGNet: Mamba Dual-Hypergraph Network** (arxiv 2511.00085). Combines Mamba with dual hypergraphs for temporal-causal and global relational learning. Designed to capture temporal dependencies and dynamic inter-stock interactions via hypergraph structures rather than simple pairwise graphs. [Source](https://arxiv.org/html/2511.00085)

2.4 **MDGNN: Multi-Relational Dynamic Graph Neural Network** (AAAI 2024, arxiv 2402.06633). Models comprehensive stock relationships through multi-relational dynamic graphs. Captures multifaceted and temporal influences by maintaining dynamic graph structures with heterogeneous edge types. [Source](https://arxiv.org/pdf/2402.06633)

2.5 **OmniGNN: Attention-based multi-relational dynamic GNN** (arxiv 2510.10775). Integrates macroeconomic context via sector nodes as global information hubs. Uses attention-based message passing with heterogeneous node/edge types for robust propagation during macroeconomic shocks. [Source](https://arxiv.org/pdf/2510.10775)

### 3. Multi-Modal Financial Models (Price + Text + Alternative Data)

3.1 **DASF-Net: Diffusion-Aware Sentiment Fusion Network** (Nguyen et al., *JRFM* 2025). Three-way fusion: (a) diffusion on dual graphs (industry + fundamental), (b) FinBERT sentiment with 3-day optimized aggregation window, (c) multi-head attention fusion → LSTM prediction. On 12 S&P 500 stocks (2020–2023): 91.6% relative MSE reduction vs MGAR baseline. Ablation confirms all three components matter independently. [Source](https://www.mdpi.com/1911-8074/18/8/417)

3.2 **Cross-Modal Temporal Fusion (CMTF)** (arxiv 2504.13522). Transformer-based framework that integrates price trends, macroeconomic indicators, and financial news via cross-modal attention. Models inter-modal interactions explicitly rather than concatenating modalities. [Source](https://arxiv.org/html/2504.13522v1)

3.3 **M2VN: Multi-Modal Volatility Network** (arxiv 2510.20699). Deep learning framework unifying time series features with unstructured news data for volatility forecasting. Addresses alignment/fusion of heterogeneous modalities and look-ahead bias mitigation. [Source](https://arxiv.org/html/2510.20699)

3.4 **LLM-based fusion for factors + newsflow** (arxiv 2510.15691). Compares three fusion methods: representation combination, summation, and attention-based fusion for merging LLM-generated newsflow embeddings with traditional factor representations. [Source](https://arxiv.org/pdf/2510.15691)

3.5 **TradExpert: Mixture of Expert LLMs** (arxiv 2411.00782). Uses four specialized LLMs as experts analyzing distinct data modalities (technical, fundamental, news, sentiment), with a gating mechanism to synthesize insights. MoE paradigm applied to financial multi-modal fusion. [Source](http://arxiv.org/pdf/2411.00782)

3.6 **KAN + Multi-Modal Integration** (Springer *Computational Economics* 2026). Integrates historical prices, financial news text, and social media sentiment via attention-based feature fusion with Kolmogorov-Arnold Networks. [Source](https://link.springer.com/article/10.1007/s10614-026-11314-x)

### 4. Architectures That Claim to Beat GBDT on Financial Data

4.1 **CNN-LightGBM > standalone LightGBM** — The hybrid beats standalone LightGBM with statistical significance (DM t=−2.51, p<0.01 on NIFTY50) by using CNN-learned embeddings as features for LightGBM rather than raw features. This is a feature engineering gain, not a pure architecture win. [Source](https://www.mdpi.com/2073-8994/18/3/416)

4.2 **No pure DL model convincingly beats well-tuned CatBoost/LightGBM on tabular financial data.** The authoritative benchmark by McElfresh et al. (*NeurIPS 2023*) on 176 tabular datasets shows: (a) CatBoost is the single best algorithm overall, (b) GBDTs outperform NNs on large datasets and datasets with high n_samples/n_features ratio, (c) for ~1/3 of datasets, light hyperparameter tuning on CatBoost gives more gain than choosing between GBDTs and NNs. The financial domain's characteristics (large n_samples, moderate n_features, heavy tails) favor GBDTs. [Source](https://arxiv.org/html/2305.02997v4)

4.3 **PULSE-KAN achieves 57.56% ACC on StockNet** — Beats Adv-ALSTM by +1.32% ACC with three innovations: (a) P-EMA Trend Bridge (dual raw+smoothed input), (b) Pola Pulse Router (polarized linear attention replacing softmax), (c) KAN Signal Refiner (Chebyshev polynomial activations). But this is ~57% accuracy — far from a reliable trading signal. The baseline LSTM is only ~51.5%. [Source](https://www.mdpi.com/2227-7390/14/9/1494)

4.4 **Deep tabular models (TabPFN, FT-Transformer, NODE) show promise on smaller datasets** but underperform GBDTs as dataset size grows. TabPFN is notable for achieving strong results on datasets ≤3000 samples with sub-second inference, but its O(n²) complexity limits financial applications. [Source](https://arxiv.org/html/2305.02997v4)

### 5. What Actually Works for Daily Stock Return Prediction

5.1 **The signal is extremely weak.** CNN-LightGBM achieves OOS-R² of only 0.0285 vs. Naive0 on NIFTY50 — this means the model explains <3% of the variance not explained by predicting zero. The absolute R² is 0.028 (i.e., 2.8% of total variance). [Source](https://www.mdpi.com/2073-8994/18/3/416)

5.2 **Horizon matters enormously.** The measurable predictive edge exists only at 1-day horizon; by H=5, OOS-R² turns negative (−0.004) and becomes non-significant. Multi-day returns aggregate too much noise. [Source](https://www.mdpi.com/2073-8994/18/3/416)

5.3 **Strictly chronological, no-look-ahead protocols are essential.** Many papers reporting stronger results inadvertently leak future information through feature normalization, wavelet transforms applied globally, or overlapping windows. The CNN-LightGBM paper's ablations show that removing denoising nearly eliminates the edge (OOS-R² drops from 0.029 to 0.002). [Source](https://www.mdpi.com/2073-8994/18/3/416)

5.4 **Feature engineering still dominates architecture choice.** Wavelet denoising + recency-aware standardization + embedding standardization contribute more to performance than the choice between CNN, LSTM, or TimesNet as the encoder. The ablation in CNN-LightGBM: NoDenoise drops OOS-R² by 93%, NoEmbStd drops it by 26%, NoRecency drops it by 17%. [Source](https://www.mdpi.com/2073-8994/18/3/416)

5.5 **Deep learning adds limited value beyond buy-and-hold for most stocks.** Sciencedirect 2025 study: "Deep learning adds limited value for most stocks beyond buy-and-hold. LSTM shows superior performance only in Technology and Consumer sectors." [Source](https://www.sciencedirect.com/science/article/pii/S2666827025001276)

5.6 **Foundation models for finance are emerging but unproven.** Kronos (arxiv 2508.02739) proposes a pre-trained transformer for candlestick data with novel pre-training tasks (stock code classification, sector classification, moving average prediction), but out-of-sample trading performance is not yet demonstrated. [Source](https://arxiv.org/html/2508.02739)

5.7 **Stock chart analysis via DNN is largely a myth.** Nature *Humanities and Social Sciences Communications* 2025 review: "Our analysis reveals that the most prominent studies regarding LSTMs and DNNs predictors for stock market prediction are not reproducible in real-world applications." [Source](https://www.nature.com/articles/s41599-025-04761-8)

### 6. Novel Fusion Paradigms in Quantitative Finance

6.1 **Adaptive Model Fusion (TimeFuse, ICML 2025).** Sample-level inspection reveals no single model consistently outperforms others — each excels in specific cases. TimeFuse adaptively selects the best model per sample, providing a meta-fusion framework. General time series, not finance-specific but principles apply. [Source](https://q-rz.github.io/static/icml25/icml25-timefuse-paper.pdf)

6.2 **LLM as Monte-Carlo alternative.** Aldridge & Kim (SSRN 2024) propose Temporal Fusion Transformers fed with LLM-generated scenarios as an alternative to Monte-Carlo simulation for quantitative financial models. Novel paradigm: LLM → scenario generation → TFT → risk/reward ratios. [Source](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4999492)

6.3 **Ensemble of heterogeneous models with transfer learning.** MDPI *Entropy* 2025: Hybrid framework integrating ensemble models, fusion models, and transfer learning for flexible target forecasting. Systematically combines multiple model types with target flexibility. [Source](https://www.mdpi.com/1099-4300/28/1/84)

6.4 **Monte Carlo + Ensemble ML integration.** *Quantitative Finance and Economics* 2024: Integrates Monte Carlo simulations with ensemble ML models for risk-reward analysis (Sharpe, Sortino, Treynor) on SPY ETF and major stocks. [Source](http://www.aimspress.com/aimspress-data/qfe/2024/2/PDF/QFE-08-02-011.pdf)

---

## Sources

### Kept (high-quality, directly relevant)

- **CNN-LightGBM hybrid for next-day log-return** (Bai et al., *Symmetry* 2026) — Most rigorous paper found: strict no-look-ahead protocol, ablation studies, Diebold-Mariano significance, economic evaluation. [https://www.mdpi.com/2073-8994/18/3/416](https://www.mdpi.com/2073-8994/18/3/416)
- **When Do Neural Nets Outperform Boosted Trees?** (McElfresh et al., NeurIPS 2023) — Definitive 176-dataset benchmark; CatBoost is top overall, GBDTs dominate large datasets. [https://arxiv.org/html/2305.02997v4](https://arxiv.org/html/2305.02997v4)
- **SAMBA: Graph-Mamba for Stock Prediction** (Mehrabian et al., arxiv 2410.03707) — Novel GNN+Mamba fusion; strong results on US indices. [https://arxiv.org/pdf/2410.03707v2](https://arxiv.org/pdf/2410.03707v2)
- **DASF-Net: Diffusion GNN + Sentiment Fusion** (Nguyen et al., *JRFM* 2025) — State-of-the-art multimodal fusion with diffusion graph learning + FinBERT. [https://www.mdpi.com/1911-8074/18/8/417](https://www.mdpi.com/1911-8074/18/8/417)
- **PULSE-KAN** (Zhang & Li, *Mathematics* 2026) — Novel KAN head + polarized attention + EMA trend bridge for stock movement classification. [https://www.mdpi.com/2227-7390/14/9/1494](https://www.mdpi.com/2227-7390/14/9/1494)
- **Cross-Modal Temporal Fusion** (arxiv 2504.13522) — Transformer-based fusion of price, macro, and news for financial forecasting. [https://arxiv.org/html/2504.13522v1](https://arxiv.org/html/2504.13522v1)
- **M2VN: Multi-Modal Volatility Network** (arxiv 2510.20699) — Deep learning framework for volatility forecasting fusing numerical + textual data. [https://arxiv.org/html/2510.20699](https://arxiv.org/html/2510.20699)
- **HIGSTM: Hierarchical Spatio-Temporal Mamba** (arxiv 2503.11387) — Multiscale Mamba capturing global dynamics and stock interdependencies. [https://arxiv.org/html/2503.11387v1](https://arxiv.org/html/2503.11387v1)
- **MaGNet: Mamba Dual-Hypergraph** (arxiv 2511.00085) — Mamba + hypergraph for temporal-causal and relational learning. [https://arxiv.org/html/2511.00085](https://arxiv.org/html/2511.00085)
- **MDGNN: Multi-Relational Dynamic GNN** (AAAI 2024, arxiv 2402.06633) — Comprehensive dynamic stock graph modeling. [https://arxiv.org/pdf/2402.06633](https://arxiv.org/pdf/2402.06633)
- **OmniGNN: Globalized Multi-relational GNN** (arxiv 2510.10775) — Macroeconomic context integration via GNN. [https://arxiv.org/pdf/2510.10775](https://arxiv.org/pdf/2510.10775)
- **TradExpert: LLM MoE for Trading** (arxiv 2411.00782) — Four LLM experts + gating for multi-modal trading signals. [http://arxiv.org/pdf/2411.00782](http://arxiv.org/pdf/2411.00782)
- **Kronos: Foundation Model for Financial Markets** (arxiv 2508.02739) — Pre-trained transformer for candlestick data. [https://arxiv.org/html/2508.02739](https://arxiv.org/html/2508.02739)
- **TimeFuse: Adaptive Model Fusion** (ICML 2025) — Sample-level model selection for time series. [https://q-rz.github.io/static/icml25/icml25-timefuse-paper.pdf](https://q-rz.github.io/static/icml25/icml25-timefuse-paper.pdf)
- **LLM + TFT as Monte-Carlo Alternative** (Aldridge & Kim, SSRN 2024) — Novel fusion paradigm: LLM scenarios → TFT → risk metrics. [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4999492](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4999492)
- **Stock market DNN via chart analysis: myth?** (Nature *Humanities & Social Sciences Communications* 2025) — Critical review finding most DL stock prediction studies are not reproducible. [https://www.nature.com/articles/s41599-025-04761-8](https://www.nature.com/articles/s41599-025-04761-8)

### Dropped

- **LOBCAST benchmark** (Springer 2024) — Focuses on limit order book data, not daily returns; domain mismatch.
- **Comparing Four Regression Models** (Atlantis Press) — Conference paper, too basic; compares XGBoost/LightGBM/CatBoost without novel architecture.
- **Various MDPI forecasting surveys** — Survey/review papers without primary architectural contributions.
- **SST: Multi-Scale Hybrid Mamba-Transformer** (arxiv 2404.14757) — General time series, not finance-specific; dropped for relevance.
- **HAELT** (arxiv 2506.13981) — High-frequency focused; too new and unvalidated.
- **SSPT: Stock Specialized Pre-trained Transformer** (arxiv 2506.16746) — Pre-training tasks are basic (sector classification, MA prediction); limited novelty.
- **MEANT: Multimodal Encoder** (EMNLP 2024) — NLP-focused; financial application is secondary.

---

## Gaps

1. **No large-scale, reproducible benchmark comparing GBDT vs. DL specifically on daily stock return prediction** with strict chronological protocols. The McElfresh et al. benchmark covers general tabular data, not financial returns specifically.

2. **Most multi-modal papers (DASF-Net, CMTF) evaluate on price level MSE**, not on trading returns or strategy performance with transaction costs. The gap between "accurate price prediction" and "profitable trading" remains largely unbridged.

3. **All reported daily return prediction improvements are economically small** — OOS-R² in the 0.01–0.03 range. Whether these translate to profitable strategies after transaction costs is uncertain. Even the CNN-LightGBM paper's economic evaluation (Sharpe 1.69 at 5bp costs, falling to 0.39 at 30bp) suggests the edge is thin.

4. **Foundation models for finance (Kronos) are pre-trained but their out-of-sample trading performance is not yet evaluated.** This is a rapidly moving area but currently unvalidated.

5. **No paper convincingly demonstrates pure DL beating well-tuned LightGBM/CatBoost on financial tabular data** — all "wins" come from either (a) hybrid DL+GBDT designs, or (b) multi-modal augmentation where the gain is from additional data, not architecture.

6. **Sample sizes in most studies are small** — NIFTY50 has ~2500 trading days, StockNet has ~20K samples. Financial ML's fundamental challenge is low n relative to noise, making overfitting the dominant risk.

---

## Recommendations for Glaubenskrieg

Based on this landscape survey, the most promising directions are:

1. **Hybrid CTM/LSTM → GBDT** — The CNN-LightGBM paper's embedding standardization trick is directly transferable. A Mamba encoder → LightGBM/CatBoost regressor with explicit embedding normalization is the natural extension of v5's CTM architecture.

2. **Multi-modal text augmentation** — Adding FinBERT sentiment or newsflow embeddings via cross-attention fusion (DASF-Net style) could add signal. But this requires reliable news data aligned with trading timestamps.

3. **Stacking/ensemble of heterogeneous models** — TimeFuse (ICML 2025) shows no single model wins everywhere. A meta-learner that selects among CTM, GBDT, and ensemble predictions per-window could outperform any single model.

4. **Do not expect DL to beat well-tuned LightGBM on tabular-only features.** The evidence strongly suggests GBDT remains the best single-model class for pure tabular financial prediction. DL adds value as a feature extractor feeding into GBDT, not as a standalone predictor.

5. **Rigorous no-look-ahead protocols are non-negotiable.** The CNN-LightGBM paper's meticulous approach (split-wise standardization, one-sided wavelet denoising, training-only statistics) should be the gold standard.
