The research is complete. The output file is at `/home/xieguiawu/Desktop/ML/Glaubeskrieg/research_limits_ml_finance.md`.

**Key findings synthesized across 6 research angles:**

1. **R² Ceiling:** Best monthly OOS R² for stock returns is 1–3% with sophisticated ML. Daily R² is sub-1%. The Kelly-Malamud LLG framework reveals true population predictability may be ~20%, but finite samples make it fundamentally unreachable.

2. **IC Benchmarks:** Realistic cross-sectional Rank IC for stock selection rarely exceeds 0.02–0.05, with high volatility making it often statistically indistinguishable from zero.

3. **EMH & ML:** Markets are not perfectly efficient, but the predictable component is tiny, noisy, and non-stationary. OOS directional accuracy converges to ~50% (random) across all hyperparameter choices.

4. **Structural Limits:** Three hard ceilings — low SNR (signal <0.05% vs noise ~2% daily), non-stationarity (shifting DGPs), and the curse of dimensionality (LLG = O(P/T)).

5. **Practical Ceilings:** After transaction costs (~57% reduction), post-publication decay, and researcher degrees of freedom (59% non-standard error), net deployable Sharpe is realistically 0.3–0.8. Any gross Sharpe above ~3 or daily R² above ~2% is likely contaminated.