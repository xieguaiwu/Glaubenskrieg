# ⚔️ Glaubenskrieg — Conv-Temporal-Mamba + C++ GBDT for Quantitative Investment

<p align="center">
  <a href="README.zh-CN.md">🇨🇳 中文版</a>
</p>

**A unified quantitative research system combining Mamba State Space Models, causal temporal convolutions, and a custom high-performance C++ GBDT engine for stock return & volatility prediction.**

Glaubenskrieg brings together two engines:
- **CTM (Conv-Temporal-Mamba)** — a rigorous PyTorch system with selective SSMs, seasonal-trend decomposition, and walk-forward validation
- **Hoffnung** — a custom C++17 Gradient Boosted Decision Tree library with quant-specific loss functions and Python bindings

## Quick Start

```bash
# Install Python dependencies
pip install -e .

# Train a CTM model
python scripts/train.py --config configs/default.yaml --device cuda

# Benchmark CTM vs LightGBM vs GARCH
python scripts/baseline_compare.py --seeds 42,123,456,789,1024

# Backtest
python scripts/backtest.py --predictions results/predictions.csv

# --- Hoffnung C++ GBDT ---
cd Hoffnung
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
make -j$(nproc)
cd ../python && pip install -e .
```

```python
# Hoffnung GBDT usage
from gbdt import GBDTRegressor, RankICLoss
model = GBDTRegressor(n_trees=200, max_depth=6, loss=RankICLoss())
model.fit(X_train, y_train)
scores = model.predict(X_test)
```

## System Architecture

```
                    ╔═══════════════════════════════════╗
                    ║       Glaubenskrieg System        ║
                    ╚═══════════════════════╤═══════════╝
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            ▼                               ▼                               ▼
   ┌─────────────────┐           ┌─────────────────────┐       ┌─────────────────────┐
   │  CTM (PyTorch)  │           │  Hoffnung (C++17)   │       │  Baselines          │
   │  Mamba SSM      │           │  Custom GBDT Engine │       │  LightGBM / GARCH   │
   │  CausalConv     │           │  Histogram-based    │       │  Ridge / XGBoost    │
   │  SeasonalTrend  │ ──────►   │  Leaf-wise growth   │ ────► │                     │
   │  Bi-Mamba       │  ensemble │  Quant-specific     │ bench  │                     │
   │  Multi-task     │  fusion   │  losses (RankIC,    │        │                     │
   │  heads          │           │   DirectionalSharpe)│        │                     │
   └────────┬────────┘           └──────────┬──────────┘       └─────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
            ┌──────────────────────────────┐
            │   Portfolio Optimizer        │
            │   Mean-Variance / Risk-Parity│
            │   / Regime-Adaptive          │
            └──────────────────────────────┘
```

### CTM Engine (`src/`, `scripts/`)

| Component | Description |
|-----------|-------------|
| `CausalConv1d` | Depthwise causal convolution for local temporal patterns |
| `SeasonalTrendDecomp` | Learnable trend + seasonal + residual decomposition |
| `MambaBlock` | Selective SSM (Gu & Dao 2023), input-dependent state transitions |
| `Bi-Mamba` | Forward + backward Mamba passes for bidirectional context |
| `Ensemble` | CTM + LightGBM stacked ensemble with time-gated fusion |
| `Curriculum Trainer` | Progressive: easy→hard samples, MSE→Sharpe→IC loss |

### Hoffnung GBDT Engine (`Hoffnung/`)

| Component | Description |
|-----------|-------------|
| `C++17 core` | LibTorch backend, histogram-based split finding, leaf-wise growth |
| `Python bindings` | pybind11, seamless numpy/torch interop |
| `Quant-specific losses` | RankIC, Directional Sharpe, Huber, MAE, MSE |
| `Multi-output` | Joint prediction of return + volatility |
| `OpenMP parallel` | Histogram construction |

## Features

- **Walk-forward validation** with purged cross-validation (no lookahead bias)
- **Triple-barrier labeling** for supervised return prediction
- **Fractional differentiation** for stationarity without full differencing
- **Wavelet denoising** for signal extraction
- **Multi-asset** cross-sectional + temporal attention
- **Volatility prediction** with GARCH(1,1) baseline and QLIKE loss
- **Portfolio optimization**: mean-variance, risk-parity, regime-adaptive
- **SHAP explainer** for feature importance
- **C++ GBDT** with custom quant loss functions for high-performance ensemble

## Project Structure

```
├── src/                    CTM PyTorch source
│   ├── data/               Datasets, features, labeling, walk-forward splits
│   ├── model/              CTM, Mamba blocks, attention, ensemble, losses
│   ├── train/              Trainers (standard, advanced, curriculum, ensemble)
│   ├── utils/              Metrics (IC, Sharpe, QLIKE), serialization, SHAP
│   └── execution/          Live trading broker adapters (Alpaca)
├── scripts/                Training, inference, backtesting, benchmark scripts
├── Hoffnung/               C++ GBDT engine (standalone sub-project)
│   ├── src/                C++ source (gbdt.cpp, builder.cpp, tree.cpp, bindings)
│   ├── include/gbdt/       Public C++ headers
│   ├── python/gbdt/        Python bindings (GBDTRegressor, losses, pipeline)
│   ├── tests/              Python & C++ test suites
│   └── CMakeLists.txt      CMake build configuration
├── configs/                YAML configuration files
├── tests/                  Pytest-based test suite (20+ test modules)
├── data/                   Training data
├── results/                Experimental results, benchmarks, backtests
└── checkpoints/            Trained model checkpoints
```

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

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.0, numpy, pandas, scipy, pyyaml, LightGBM
- Optional: `alpaca-py` for live trading, `shap` for explainability
- For Hoffnung C++ GBDT: CMake ≥ 3.18, LibTorch, pybind11, OpenMP

## License

MIT License.

## References

- Gu & Dao (2023): *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Shi (2024): *MambaStock: Selective SSM for Stock Prediction*
- Gu, Kelly & Xiu (2020): *Empirical Asset Pricing via Machine Learning*
- López de Prado (2018): *Advances in Financial Machine Learning*
- Friedman (2001): *Greedy Function Approximation: A Gradient Boosting Machine*
- Chen & Guestrin (2016): *XGBoost: A Scalable Tree Boosting System*

<p align="center">
  <a href="README.zh-CN.md">🇨🇳 中文版</a>
</p>
