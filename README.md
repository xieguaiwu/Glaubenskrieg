# ⚔️ Glaubenskrieg — Conv-Temporal-Mamba for Quantitative Investment

<p align="center">
  <a href="README.zh-CN.md">🇨🇳 中文版</a>
</p>

**Mamba State Space Models + GBDT Ensembles for Stock Return & Volatility Prediction.**

A rigorous PyTorch system combining selective state-space models (Mamba), causal temporal convolutions, seasonal-trend decomposition, and gradient-boosted trees for multi-asset financial time series forecasting.

## Quick Start

```bash
pip install -e .

# Train a CTM model
python scripts/train.py --config configs/default.yaml --device cuda

# Inference
python scripts/infer.py --ckpt checkpoints/best.pt --data data/features.csv

# Backtest
python scripts/backtest.py --predictions results/predictions.csv

# Benchmark against baselines
python scripts/baseline_compare.py --ctm results/ctm.json --gbdt results/gbdt.json
```

## Architecture

```
OHLCV → CausalConv → SeasonalTrendDecomp → MambaBlock×N
                                                ↓
                                   [Bi-Mamba backward pass]
                                                ↓
                                   Multi-Task Output Heads
                                   ├── Return Prediction (IC loss)
                                   ├── Volatility Prediction (QLIKE)
                                   └── Direction Classification
                                                ↓
                                   GBDT Ensemble (optional)
                                                ↓
                                   Portfolio Optimizer → Signals
```

| Component | Description |
|-----------|-------------|
| `CausalConv1d` | Depthwise causal convolution for local temporal patterns |
| `SeasonalTrendDecomp` | Learnable trend + seasonal + residual decomposition |
| `MambaBlock` | Selective SSM (Gu & Dao 2023), input-dependent state transitions |
| `Bi-Mamba` | Forward + backward Mamba passes for bidirectional context |
| `Ensemble` | CTM + LightGBM stacked ensemble with time-gated fusion |
| `Curriculum Trainer` | Progressive: easy→hard samples, MSE→Sharpe→IC loss |

## Features

- **Walk-forward validation** with purged cross-validation (no lookahead bias)
- **Triple-barrier labeling** for supervised return prediction
- **Fractional differentiation** for stationarity without full differencing
- **Wavelet denoising** for signal extraction
- **Multi-asset** cross-sectional + temporal attention
- **Volatility prediction** with GARCH(1,1) baseline and QLIKE loss
- **Portfolio optimization**: mean-variance, risk-parity, regime-adaptive
- **SHAP explainer** for feature importance

## Training

```bash
# CTM-only
python scripts/train.py --config configs/default.yaml --device cuda

# Multi-seed reproducibility
python scripts/train.py --config configs/default.yaml --seeds 42,123,456

# Curriculum training
python scripts/train.py --config configs/scale_loop.yaml --curriculum

# GBDT baseline
python scripts/train_gbdt_only.py --data data/features.csv
```

## Evaluation

```bash
# Full benchmark: CTM vs LightGBM vs GARCH
python scripts/baseline_compare.py --seeds 42,123,456,789,1024

# Backtest with portfolio construction
python scripts/backtest.py --predictions results/predictions.csv --capital 1000000

# GARCH baseline
python scripts/garch_baseline.py --data data/returns.csv
```

## Project Structure

```
src/
├── data/           # Dataset, features, labeling, walk-forward splits
├── model/          # CTM, Mamba blocks, attention, ensemble, losses
├── train/          # Trainers (standard, advanced, curriculum, ensemble)
├── utils/          # Metrics (IC, Sharpe, QLIKE), serialization, SHAP
└── execution/      # Live trading broker adapters (Alpaca)

scripts/
├── train.py        # Main training entry point
├── infer.py        # Inference / prediction
├── backtest.py     # Walk-forward backtesting
├── baseline_compare.py    # Multi-model benchmark
├── portfolio_optimizer.py # Portfolio construction
├── garch_baseline.py      # GARCH(1,1) volatility model
├── volatility_backtest.py # Volatility backtesting
├── download_sp500.py      # Data download utilities
├── enhanced_features.py   # Feature engineering
└── synthetic_data.py      # Synthetic data for testing

configs/            # YAML configuration files
tests/              # Pytest-based test suite (20+ test modules)
```

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.0, numpy, pandas, scipy, pyyaml, LightGBM
- Optional: `alpaca-py` for live trading, `shap` for explainability

## License

MIT License.

## References

- Gu & Dao (2023): *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Shi (2024): *MambaStock: Selective SSM for Stock Prediction*
- Gu, Kelly & Xiu (2020): *Empirical Asset Pricing via Machine Learning*
- López de Prado (2018): *Advances in Financial Machine Learning*

<p align="center">
  <a href="README.zh-CN.md">🇨🇳 中文版</a>
</p>
