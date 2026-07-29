# Graph Report - Glaubenskrieg  (2026-07-29)

## Corpus Check
- 361 files · ~299,328 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2000 nodes · 4050 edges · 95 communities (93 shown, 2 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 270 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ae81fc1f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- EnsembleWalkForwardTrainer
- ensemble_trainer.py
- LossConfig
- CTMStockModel
- AlpacaBroker
- FusedMultiHeadCrossAttention
- portfolio_optimizer.py
- synthetic_data.py
- MockBroker
- RecurrentCTM
- TradingAccount
- compare_gradients
- Position
- 量化金融深度补强分析报告
- Order
- TestPipeline
- TimeDecayGate
- volatility_backtest.py
- test_data.py
- tests/conftest.py
- P3EnsembleTrainer
- garch_baseline.py
- pipeline_v2.py
- gbdt.cpp
- gbdt/__init__.py
- TestTreeSHAPExplainer
- GBDT
- JsonParser
- glauben_v2_noise_free.py
- test_tree.cpp
- builder.cpp
- compute_all_features
- GBDTConfig
- main
- TestLossFunctions
- test_gbdt_features.py
- train.py
- ctm_composite_loss_for_gbdt
- TreeNode
- pipeline_v3.py
- sharpe_ic_mismatch.py
- vector
- validate
- TestGBDTIntegration
- purged_walkforward.py
- Tensor
- BaseBroker
- GBDTTrainer
- fractional_diff.py
- process_stock
- process_stock
- FlatNode
- GlaubenskriegPredictor
- train_gbdt_only.py
- RecencyAwareScaler
- config.py
- shap_explainer.py
- ⚔️ Glaubenskrieg — Conv-Temporal-Mamba + C++ GBDT for Quantitative Investment
- ⚔️ Glaubenskrieg — Mamba + C++ GBDT 量化投资系统
- backtest.py
- download_nasdaq100.py
- ._tree_shap_single
- TreeBuilder
- FeaturePipeline
- train_volatility.py
- get_bins
- GMADLLoss
- test_shap_explainer.py
- benchmark_ridge_vs_glauben.py
- download_sp500.py
- data/__init__.py
- TreeSHAPExplainer
- enhanced_features.py
- quantitative_deep_dive.py
- build_gbdt_feature_matrix
- extract_ctm_hidden_features
- normalize_features
- PYBIND11_MODULE
- TGPE v4 实验状态报告
- .forward
- InferenceEngine
- diagnose_v2.py
- ._sync_order_states
- _resolve_child_ids
- sharpe_ratio_torch
- GradHessOutput
- GBDTFeatureConfig
- apply_curriculum_dropout
- associative_scan
- run_tests.sh
- ctm-stock-prediction

## God Nodes (most connected - your core abstractions)
1. `CTMStockModel` - 71 edges
2. `AlpacaBroker` - 65 edges
3. `TradingAccount` - 64 edges
4. `MockBroker` - 52 edges
5. `LossConfig` - 50 edges
6. `OrderManager` - 48 edges
7. `compute_all_features()` - 41 edges
8. `MultiAssetCTM` - 41 edges
9. `Order` - 39 edges
10. `EnsembleWalkForwardTrainer` - 38 edges

## Surprising Connections (you probably didn't know these)
- `FrozenSSMPredictor` --uses--> `CTMStockModel`  [INFERRED]
  scripts/glauben_v2_noise_free.py → src/model/ctm_model.py
- `ResidualSSMPredictor` --uses--> `CTMStockModel`  [INFERRED]
  scripts/glauben_v2_noise_free.py → src/model/ctm_model.py
- `DecoupledPredictor` --uses--> `CTMStockModel`  [INFERRED]
  scripts/glauben_v2_noise_free.py → src/model/ctm_model.py
- `SyntheticConfig` --uses--> `MultiAssetCTM`  [INFERRED]
  scripts/synthetic_data.py → src/model/multiasset_ctm.py
- `test_compute_sma_length()` --calls--> `compute_sma()`  [EXTRACTED]
  tests/test_data.py → src/data/features.py

## Import Cycles
- None detected.

## Communities (95 total, 2 thin omitted)

### Community 0 - "EnsembleWalkForwardTrainer"
Cohesion: 0.05
Nodes (38): _requires_ensemble, _apply_config(), _compute_sharpe(), EnsembleWalkForwardTrainer, Any, DataLoader, device, Module (+30 more)

### Community 1 - "ensemble_trainer.py"
Cohesion: 0.05
Nodes (61): Parameter, Standalone inference pipeline for CTM + GBDT ensemble. Loads trained CTM and…, compute_ic(), _differentiable_ranking_helper(), EnsembleConfig, EnsembleSignal, evaluate_ensemble(), ic_weighted_fusion() (+53 more)

### Community 2 - "LossConfig"
Cohesion: 0.07
Nodes (42): main(), Quick training pipeline test with synthetic data (large scale)., SequentialLR, LossConfig, LossWarmupScheduler, LossWrapper, lr_warmup_cosine(), Any (+34 more)

### Community 3 - "CTMStockModel"
Cohesion: 0.06
Nodes (42): CausalConv1d, CTMStockModel, CTM: Conv-Temporal-Mamba model for stock prediction. Full architecture:…, Conv-Temporal-Mamba stock prediction model. Architecture (simplified):…, Depthwise causal 1D convolution for local temporal feature extraction.…, Return total number of trainable parameters., Seasonal-trend decomposition via moving average, per DMamba (2026). x_t = s_t +…, Root Mean Square Layer Normalization (Zhang & Sennrich, 2019). (+34 more)

### Community 4 - "AlpacaBroker"
Cohesion: 0.07
Nodes (55): datetime, AlpacaBroker, _parse_alpaca_dt(), Broker implementation using the Alpaca Markets API (alpaca-py). Supports: -…, Alpaca supports fractional-share trading. Always returns True., Parse Alpaca datetime (ISO str / datetime / None) → datetime., broker(), connected_broker() (+47 more)

### Community 5 - "FusedMultiHeadCrossAttention"
Cohesion: 0.05
Nodes (31): FusedMultiHeadCrossAttention, GBDTModulator, Tensor, Fused multi-head cross-asset attention with GBDT modulation. Extends…, Cache GBDT predictions for the next forward pass. Parameters ---------- preds :…, Cross-asset multi-head mixing. Parameters ---------- x : (B, N, T, D) or (B*T,…, Converts per-asset GBDT predictions to pairwise attention biases. Architecture:…, Return the number of trainable parameters. (+23 more)

### Community 6 - "portfolio_optimizer.py"
Cohesion: 0.06
Nodes (53): LGBMRanker, build_panel(), differentiable_sharpe_loss(), _discretize_labels(), evaluate_window_ls(), generate_windows(), ic_spearman(), load_stock_data() (+45 more)

### Community 7 - "synthetic_data.py"
Cohesion: 0.07
Nodes (50): count_params(), GRUModel, LSTMModel, main(), prepare_data(), Any, callable, DataLoader (+42 more)

### Community 8 - "MockBroker"
Cohesion: 0.07
Nodes (16): AccountInfo, OrderManager, Manages the full order lifecycle: signal → order → fill → risk checks.…, Return the most recent *limit* orders from history., mock_broker(), mock_broker_no_fractional(), MockBroker, DataFrame (+8 more)

### Community 9 - "RecurrentCTM"
Cohesion: 0.06
Nodes (42): build_model_params(), count_params(), load_config(), main(), Verify forward + backward pass for all model scales., verify_scale(), Module, Tensor (+34 more)

### Community 10 - "TradingAccount"
Cohesion: 0.07
Nodes (15): Reduce or close a position, realizing P&L on the sold portion., Tracks capital allocation, positions, portfolio value, and PnL. All quantities…, Portfolio total value: cash + sum of position market values., Sum of unrealized P&L across all positions. Computed live from current data:…, Cumulative realized P&L from closed trades., Available cash for new positions., Gross exposure as a ratio of total portfolio value. Gross exposure =…, Single position market value as a fraction of total portfolio. (+7 more)

### Community 11 - "compare_gradients"
Cohesion: 0.10
Nodes (31): compare_gradients(), Any, Module, Tensor, Gradient checking utility for CTM model verification. Compares analytical…, Compute relative error between analytical and numerical gradients. Parameters…, load_ctm_model(), load_ensemble_trainer_state() (+23 more)

### Community 12 - "Position"
Cohesion: 0.07
Nodes (20): Return the current position for *symbol*, or None., DataFrame, Convert a human-friendly timeframe string to an alpaca-py TimeFrame string., Lazy-init the alpaca-py TradingClient., Lazy-init the alpaca-py StockHistoricalDataClient., Verify connectivity by fetching the account. Returns ------- bool True on…, Retrieve account summary. Raises ------ RuntimeError If not connected or the…, Return all open positions. (+12 more)

### Community 13 - "量化金融深度补强分析报告"
Cohesion: 0.06
Nodes (35): 1.1 基础公式, 1.2 基准参数下的期望收益, 1.3 盈亏平衡 IC, 1.4 IC → 收益敏感性分析, 1.5 组合波动率与 Sharpe Ratio, 1.6 Grinold-Kahn 信息率分解, 2.1 窗口间 IC 序列, 2.2 IC 标准差 0.044 的交易含义 (+27 more)

### Community 14 - "Order"
Cohesion: 0.16
Nodes (19): Enum, _order_type_from_alpaca(), Place an order via Alpaca. Fractional shares ------------------ When ``qty <…, Map an alpaca Order object → internal Order dataclass., Map an alpaca order-type string → OrderType enum., _to_alpaca_order_type(), _to_alpaca_side(), _to_order_status() (+11 more)

### Community 15 - "TestPipeline"
Cohesion: 0.08
Nodes (21): compute_ic(), make_synthetic_data(), purged_walk_forward_cv(), purged_walk_forward_split(), ndarray, Data pipeline utilities for GBDT quant research., Generate purged walk-forward train/test index pairs. Financial time series…, Generate synthetic regression data for testing. The target is constructed as a… (+13 more)

### Community 16 - "TimeDecayGate"
Cohesion: 0.09
Nodes (18): Horizon-dependent exponential decay gate for CTM ↔ GBDT blending. At forecast…, Horizon-dependent exponential decay gate for CTM ↔ GBDT blending. At forecast…, Args: ctm_pred: (B, N, T, C) CTM regression predictions gbdt_pred: (B, N, T)…, TimeDecayGate, Tests for TimeDecayGate — horizon-dependent CTM↔GBDT blending module., Tests for TimeDecayGate — exponential horizon-based NN→GBDT blending., String representation includes alpha, beta, gamma values., Single time step (T=1) works without division by zero. (+10 more)

### Community 17 - "volatility_backtest.py"
Cohesion: 0.11
Nodes (31): annualized_sharpe(), build_lgb_model(), calmar_ratio(), compute_all_metrics(), compute_forward_realized_vol(), load_and_prepare_data(), main(), max_drawdown() (+23 more)

### Community 18 - "test_data.py"
Cohesion: 0.09
Nodes (25): Dataset, create_sequences(), DataFrame, ndarray, Tensor, Stock dataset pipeline with causal normalization and temporal splits., Create sliding window sequences of length seq_len from (N, D) data. Returns…, Temporal (chronological) split respecting time order. Returns (train, val,… (+17 more)

### Community 19 - "tests/conftest.py"
Cohesion: 0.10
Nodes (30): Linear, grad_check_data(), linear_grad_model(), loss_config(), mse_only_loss_config(), fixture, ndarray, Tensor (+22 more)

### Community 20 - "P3EnsembleTrainer"
Cohesion: 0.10
Nodes (19): _requires_p3, P3EnsembleTrainer, Any, device, Module, ndarray, OrderedDict, Tensor (+11 more)

### Community 21 - "garch_baseline.py"
Cohesion: 0.15
Nodes (28): PredictorResult, aggregate_results(), compute_metrics(), diebold_mariano(), fit_garch_and_forecast(), fit_lgb_and_predict(), get_walk_forward_folds(), historical_mean_forecast() (+20 more)

### Community 22 - "pipeline_v2.py"
Cohesion: 0.15
Nodes (25): add_cross_sectional_features(), add_illiquidity_feature(), add_momentum_rank_features(), compute_multi_horizon_targets(), compute_vol_adjusted_returns(), dir_accuracy(), generate_windows(), ic_score() (+17 more)

### Community 23 - "gbdt.cpp"
Cohesion: 0.13
Nodes (26): FeatureImportance, fit_one_tree, line_search, predict_tree, random_col_subset, random_subset, compute_scalar_loss(), string (+18 more)

### Community 24 - "gbdt/__init__.py"
Cohesion: 0.12
Nodes (19): GBDT for Quantitative Investment — Python Package. Provides: - C++ core…, composite_quant_loss(), CompositeQuantLoss, compute_gradients(), _differentiable_rank(), log_loss(), Tensor, quantile_loss() (+11 more)

### Community 25 - "TestTreeSHAPExplainer"
Cohesion: 0.10
Nodes (13): MockGBDTModel, _tree_shap returns (N, D) for single-tree explainer., explain returns (N, D) array for multi-tree ensemble., explain with empty trees returns zeros., explain with mismatched feature count emits warning and adapts., feature_importance_shap returns (D,) array with finite values., Mock GBDT model exposing _trees, _feature_names, to_json(), and __call__., Tests for TreeSHAPExplainer with mock GBDT model and synthetic trees. (+5 more)

### Community 26 - "GBDT"
Cohesion: 0.08
Nodes (24): GBDT, bridge_, config_, fit, from_json, get_feature_importance, get_feature_importance_coverage, get_feature_importance_full (+16 more)

### Community 27 - "JsonParser"
Cohesion: 0.17
Nodes (6): string, vector, JsonParser, pos_, JsonWriter, ostringstream

### Community 28 - "glauben_v2_noise_free.py"
Cohesion: 0.13
Nodes (17): build_features(), build_seq(), DecoupledPredictor, FrozenSSMPredictor, ic_score(), main(), SSM with HiPPO-initialized, FROZEN parameters. Only projection layers train.…, x: (B, N, T, D_in) — raw OHLCV (+9 more)

### Community 29 - "test_tree.cpp"
Cohesion: 0.21
Nodes (21): Config, HistogramBuilder, build_histogram, compute_bin_boundaries, approx_equal(), main(), test_assert(), test_assert_approx() (+13 more)

### Community 30 - "builder.cpp"
Cohesion: 0.18
Nodes (20): FeatureHistogram, MonotoneConstraint, dir, feature_idx, vector, Histograms, feature_histograms, num_bins (+12 more)

### Community 31 - "compute_all_features"
Cohesion: 0.17
Nodes (20): load_multi_asset_data(), ndarray, Load multiple stock CSV files and align them for MultiAssetCTM training. Each…, compute_all_features(), compute_bollinger_bands(), compute_forward_returns(), compute_realized_volatility(), compute_rsi() (+12 more)

### Community 32 - "GBDTConfig"
Cohesion: 0.09
Nodes (22): GBDTConfig, early_stopping_rounds, early_stopping_tol, enable_missing_values, gamma, huber_delta, lambda_l1, lambda_l2 (+14 more)

### Community 33 - "main"
Cohesion: 0.15
Nodes (21): Namespace, _build_model_params(), _extract_gbdt_importance(), main(), _nan_safe_output(), Any, DataLoader, device (+13 more)

### Community 34 - "TestLossFunctions"
Cohesion: 0.16
Nodes (8): huber_loss(), mae_loss(), mse_loss(), Mean Squared Error loss. Closed-form: L = mean((y_pred - y_true)²) g_i =…, Mean Absolute Error loss. Closed-form: L = mean(|y_pred - y_true|) g_i =…, Huber loss — blends MSE and MAE with a smooth transition at ``delta``. Loss per…, Verify loss functions return correct shapes and basic properties., TestLossFunctions

### Community 35 - "test_gbdt_features.py"
Cohesion: 0.16
Nodes (11): aggregate_sequence_features(), _compute_slope(), ndarray, GBDT feature aggregation from temporal sequences. Converts CTM's temporal…, Aggregate temporal sequences into tabular features. Applies 6 aggregation…, Compute linear trend (slope) for each sequence and feature. Uses OLS: slope =…, test_aggregate_sequence_features_last_equals_slice(), test_aggregate_sequence_features_shape() (+3 more)

### Community 36 - "train.py"
Cohesion: 0.16
Nodes (17): _build_model_params(), class_targets_fn(), compute_class_weights(), load_config(), main(), Any, Series, Tensor (+9 more)

### Community 37 - "ctm_composite_loss_for_gbdt"
Cohesion: 0.12
Nodes (14): pinball_loss(), Quantile (pinball) loss for VaR estimation., ctm_composite_loss_for_gbdt(), Any, Tensor, Bridge CTM's composite loss to GBDT's gradient/Hessian interface. Converts…, Compute (-sharpe_ratio, grad, hess) for GBDT loss bridge. When ``y_true`` is…, # NOTE: The analytical gradient uses N in the denominator, but `var` above (+6 more)

### Community 38 - "TreeNode"
Cohesion: 0.13
Nodes (16): TreeNode, default_left, depth, feature_idx, gain, leaf_value, left_child, num_samples (+8 more)

### Community 39 - "pipeline_v3.py"
Cohesion: 0.18
Nodes (16): build_enhanced_features(), ensemble_predict(), gen_windows(), ic_score(), load_macro_features(), main(), make_targets(), DataFrame (+8 more)

### Community 40 - "sharpe_ic_mismatch.py"
Cohesion: 0.27
Nodes (17): add_unit(), _extract_units_recursive(), main(), parse_archives(), parse_benchmark(), parse_fe_lgb(), parse_misc_remote(), parse_newdata_baselines() (+9 more)

### Community 41 - "vector"
Cohesion: 0.20
Nodes (11): string, vector, HistogramBin, count, sum_grad, sum_hess, SplitResult, feature_idx (+3 more)

### Community 42 - "validate"
Cohesion: 0.18
Nodes (13): _batch_ic(), Any, DataLoader, device, Module, no_grad, Optimizer, Tensor (+5 more)

### Community 43 - "TestGBDTIntegration"
Cohesion: 0.13
Nodes (7): gbdt_python, _import_cpp(), Basic tests for the GBDT Python package. These tests require the compiled C++…, End-to-end GBDT training and inference (requires ``gbdt_python`` .so)., Import C++ module or raise SkipTest with build instructions., TestGBDTIntegration, skipIf

### Community 44 - "purged_walkforward.py"
Cohesion: 0.16
Nodes (12): get_avg_uniqueness(), get_sample_weights(), purged_train_test_split(), PurgedKFold, ndarray, Purged Walk-Forward Cross-Validation with Sample Weights. Implements methods…, Compute sample weights combining average uniqueness, return attribution, and…, Split chronologically-sorted indices into purged training and test sets.… (+4 more)

### Community 45 - "Tensor"
Cohesion: 0.16
Nodes (8): Tensor, Shared encoder blocks: conv → [decomp] → Mamba stack → [bidirectional]. Assumes…, Shared encoding: input_proj → (cond add) → _encode_blocks. Parameters…, Forward pass. Parameters ---------- x : (B, T, input_dim) input features.…, Encode input to hidden features, optionally with conditioning. Delegates to…, Extract hidden features without output heads. Returns (B, T, model_dim) encoder…, Apply causal depthwise conv1d. Parameters ---------- x : (B, T, C) input.…, Decompose input into seasonal and trend components. Parameters ---------- x :…

### Community 46 - "BaseBroker"
Cohesion: 0.13
Nodes (5): ABC, Sync internal state with live broker account and positions., BaseBroker, DataFrame, Abstract broker interface. All broker implementations (Alpaca, IBKR, kabu) must…

### Community 47 - "GBDTTrainer"
Cohesion: 0.15
Nodes (7): GBDTTrainer, ndarray, Train the model using the configured loss function. At each boosting round: 1.…, Aggregate split-count feature importance across all trees. Returns: Importance…, Convenience split for panel (time-series cross-section) data. Two modes:…, High-level trainer bridging Python loss functions to the C++ GBDT core. The C++…, train_test_split()

### Community 48 - "fractional_diff.py"
Cohesion: 0.19
Nodes (14): _adf_critical_values(), _adf_test(), ffd_weights(), find_min_d(), fractional_diff(), _pvalue_from_tstat(), ndarray, Fractional Differentiation (FFD) per López de Prado (2018), Advances in… (+6 more)

### Community 49 - "process_stock"
Cohesion: 0.22
Nodes (13): clip_extreme_returns(), detect_corporate_actions(), forward_adjust_one(), main(), process_stock(), DataFrame, ndarray, Path (+5 more)

### Community 50 - "process_stock"
Cohesion: 0.22
Nodes (13): detect_splits(), forward_adjust(), main(), process_stock(), DataFrame, ndarray, Path, Forward-adjust (前复权) OHLCV data from Tencent Finance API. Tencent raw data… (+5 more)

### Community 51 - "FlatNode"
Cohesion: 0.23
Nodes (12): FlatNode, feature_idx, leaf_value, left_child, right_child, split_value, Tensor, vector (+4 more)

### Community 52 - "GlaubenskriegPredictor"
Cohesion: 0.23
Nodes (9): GlaubenskriegPredictor, main(), DataFrame, Generate signals for a batch of stocks passed as DataFrames., Predict forward return for a single stock. Steps: 1. Compute base OHLCV…, Convenience: get top-N long picks., Generate portfolio weights. weighting: "equal" → equal weight top N…, Production predictor: loads model + feature pipeline, generates signals. Usage:… (+1 more)

### Community 53 - "train_gbdt_only.py"
Cohesion: 0.23
Nodes (13): _compute_sharpe(), _directional_accuracy(), load_config(), load_multi_asset_data(), main(), Any, ndarray, Standalone GBDT-only walk-forward training script. Baseline for paper comparing… (+5 more)

### Community 54 - "RecencyAwareScaler"
Cohesion: 0.23
Nodes (9): ndarray, Recency-aware walk-forward normalization. Implements the method from Bai et al.…, Standardization with recency-aware statistics. Two modes: - "expanding": Use…, Compute statistics from training data. Parameters ---------- X : (N, D) array —…, Apply standardization using training statistics. Parameters ---------- X : (N,…, Fit on X and transform X., Walk-forward normalization with strict no-look-ahead. For each window: 1. Fit…, RecencyAwareScaler (+1 more)

### Community 55 - "config.py"
Cohesion: 0.20
Nodes (13): AccountConfig, BrokerConfig, load_trading_config(), Trading execution configuration loaded from YAML. Provides a TradingConfig…, Broker connection settings., Account constraints and capital settings., Risk-management thresholds., Rebalancing strategy parameters. (+5 more)

### Community 56 - "shap_explainer.py"
Cohesion: 0.16
Nodes (13): _get_bool(), _get_float(), _get_int(), _is_leaf_node(), TreeSHAP explainer for GBDT models. Implements the TreeSHAP algorithm (Lundberg…, Whether NaN / missing values go left., Return first matching int-valued key from a tree node dict., Return first matching float-valued key from a tree node dict. (+5 more)

### Community 57 - "⚔️ Glaubenskrieg — Conv-Temporal-Mamba + C++ GBDT for Quantitative Investment"
Cohesion: 0.15
Nodes (12): CTM Engine (`src/`, `scripts/`), Evaluation, Features, ⚔️ Glaubenskrieg — Conv-Temporal-Mamba + C++ GBDT for Quantitative Investment, Hoffnung GBDT Engine (`Hoffnung/`), License, Project Structure, Quick Start (+4 more)

### Community 58 - "⚔️ Glaubenskrieg — Mamba + C++ GBDT 量化投资系统"
Cohesion: 0.15
Nodes (12): CTM 引擎（`src/`、`scripts/`）, ⚔️ Glaubenskrieg — Mamba + C++ GBDT 量化投资系统, Hoffnung GBDT 引擎（`Hoffnung/`）, 参考文献, 快速开始, 特性, 环境要求, 系统架构 (+4 more)

### Community 59 - "backtest.py"
Cohesion: 0.23
Nodes (12): _build_comparison_table(), _compute_hit_rate_by_decile(), _compute_ic_series(), _compute_metrics(), main(), DataFrame, Series, Simple backtesting script for CTM + GBDT ensemble signals. Simulates a long-… (+4 more)

### Community 60 - "download_nasdaq100.py"
Cohesion: 0.24
Nodes (12): download_all(), download_batch(), download_single(), fetch_constituents(), main(), DataFrame, Download a batch of tickers using yfinance (proxy via env vars)., Download a single ticker individually (for retry). (+4 more)

### Community 61 - "._tree_shap_single"
Cohesion: 0.19
Nodes (9): ndarray, Compute SHAP values for each sample across all trees. Parameters ---------- X :…, Mean absolute SHAP value per feature (global importance). Parameters ----------…, Compute SHAP values for one tree across all samples. Parameters ----------…, SHAP contribution of a single sample for one tree. Follows the decision path…, Resolve feature (split) index from multiple naming conventions., Resolve split threshold from multiple naming conventions., _resolve_feature_idx() (+1 more)

### Community 62 - "TreeBuilder"
Cohesion: 0.17
Nodes (12): mt19937, unique_ptr, SplitFinder, compute_gain, find_best_split, TreeBuilder, build_tree, compute_leaf_value (+4 more)

### Community 63 - "FeaturePipeline"
Cohesion: 0.23
Nodes (6): FeaturePipeline, main(), ndarray, Encapsulates the full feature engineering pipeline with saved state. On…, Compute normalization parameters from training data. X_base: (n_days, n_stocks,…, Apply saved feature engineering to new data. X_base: (n_days, n_stocks,…

### Community 64 - "train_volatility.py"
Cohesion: 0.35
Nodes (10): load_data(), main(), qlike(), Load CSVs → compute_all_features → (B,N,T,D) data, (B,T,N) targets., (B,N,T,D)→(B*N,D) at last timestep, (B,T,N)→(B*N,)., QLIKE = log(h) + σ²/h where h=y_pred², σ²=y_true². Lower is better., to_tabular(), train_lgb() (+2 more)

### Community 65 - "get_bins"
Cohesion: 0.24
Nodes (10): get_bins(), get_daily_vol(), get_events(), DataFrame, ndarray, Series, Triple-Barrier Labeling per López de Prado (2018) Advances in Financial ML,…, Compute daily volatility as exponentially weighted moving standard deviation.… (+2 more)

### Community 66 - "GMADLLoss"
Cohesion: 0.22
Nodes (7): compute(), GMADLLoss, Tensor, Generalized Mean Absolute Directional Loss (GMADL). Reference: Glaubenskrieg v5…, Generalized Mean Absolute Directional Loss. Parameters ---------- alpha :…, Compute the GMADL scalar loss. Parameters ---------- y_pred : torch.Tensor…, Standalone GMADL computation. Can be used without instantiating ``GMADLLoss``,…

### Community 67 - "test_shap_explainer.py"
Cohesion: 0.18
Nodes (8): Resolve leaf (prediction) value from multiple naming conventions., _resolve_leaf_value(), fixture, Tests for TreeSHAPExplainer and helper functions (shap_explainer module)., _resolve_leaf_value extracts leaf prediction value., _resolve_leaf_value returns 0.0 for missing keys., A simple 3-node tree: root → left leaf, root → right leaf (flat list format)., synthetic_tree_json()

### Community 68 - "benchmark_ridge_vs_glauben.py"
Cohesion: 0.40
Nodes (9): build_features(), build_sequences(), ic_score(), load_data(), main(), Build (B, N, T, D) sequences for MultiAssetCTM. X: (n_days, n_stocks,…, run_glauben(), run_ridge() (+1 more)

### Community 69 - "download_sp500.py"
Cohesion: 0.31
Nodes (9): download_batch(), fetch_sp500_tickers(), main(), DataFrame, Path, Download OHLCV data for a batch of tickers using yfinance. Returns dict of…, Save a single ticker's data as CSV. Returns the file path., Scrape the current S&P 500 constituent list from Wikipedia. Returns cleaned… (+1 more)

### Community 70 - "data/__init__.py"
Cohesion: 0.36
Nodes (8): causal_wavelet_denoise(), compute_denoised_returns(), ndarray, Wavelet denoising for financial time series. Implements the method from Bai et…, Compute forward returns using wavelet-denoised prices. Parameters ----------…, Apply wavelet denoising to a 1D signal. Parameters ---------- signal : (T,)…, Causal (one-sided) wavelet denoising. Each point uses only past data for…, wavelet_denoise()

### Community 71 - "TreeSHAPExplainer"
Cohesion: 0.33
Nodes (6): Any, TreeSHAP explainer for Hoffnung GBDT models. Implements a single-path SHAP…, Extract flat tree node lists from a GBDT model. Supports: - C++ pybind GBDT…, Parse JSON from to_json() into a list of tree node lists., Convert a list of pybind TreeNode objects to dicts., TreeSHAPExplainer

### Community 72 - "enhanced_features.py"
Cohesion: 0.39
Nodes (8): compute_feature_matrix(), ic_score(), load_fred_macro(), load_stocks(), main(), Compute features for all stocks, optionally with wavelet denoising., sharpe(), walk_forward_ret()

### Community 73 - "quantitative_deep_dive.py"
Cohesion: 0.29
Nodes (7): compute_monthly_return(), Mean of truncated normal N(mu,sigma) in [a, b], Convert IC to monthly net return., Run Monte Carlo simulation of monthly portfolio returns. Parameters: - ic_mean:…, run_simulation(), StrategyParams, truncated_mean()

### Community 74 - "build_gbdt_feature_matrix"
Cohesion: 0.36
Nodes (4): build_gbdt_feature_matrix(), Build a flat GBDT-ready feature matrix from temporal sequences. Aggregates raw…, test_build_gbdt_feature_matrix_no_ctm(), TestBuildGBDTFeatureMatrix

### Community 75 - "extract_ctm_hidden_features"
Cohesion: 0.36
Nodes (4): extract_ctm_hidden_features(), Extract a fixed-size representation from CTM encoder hidden states. Parameters…, test_extract_ctm_hidden_features_methods(), TestExtractCTMHiddenFeatures

### Community 76 - "normalize_features"
Cohesion: 0.36
Nodes (4): normalize_features(), Z-score normalise features, optionally reusing precomputed statistics.…, test_normalize_features(), TestNormalizeFeatures

### Community 77 - "PYBIND11_MODULE"
Cohesion: 0.48
Nodes (6): array_t, Tensor, numpy_to_tensor(), PYBIND11_MODULE(), tensor_to_numpy(), m

### Community 78 - "TGPE v4 实验状态报告"
Cohesion: 0.29
Nodes (6): Server 1 (223.109.239.36:24520), Server 2 (223.109.239.32:20248), TGPE v4 实验状态报告, 下载文件清单, 快速分析命令, 远程进程状态

### Community 79 - ".forward"
Cohesion: 0.29
Nodes (4): Tensor, Forward pass. Parameters ---------- x : (B, N, T, input_dim) features.…, Extract hidden features without output heads. Processes in chunks along the…, Cross-asset mixing. Parameters ---------- x : (B, N, T, D) — batch, n_assets,…

### Community 80 - "InferenceEngine"
Cohesion: 0.40
Nodes (6): InferenceEngine, batch_predict, convert_forest, convert_tree, predict_single, GBDT::predict_batch()

### Community 81 - "diagnose_v2.py"
Cohesion: 0.53
Nodes (5): build_features(), dm_test(), ic_spearman(), main(), Diebold-Mariano test: is model 1 significantly better than model 2?

### Community 82 - "._sync_order_states"
Cohesion: 0.33
Nodes (3): Return all currently open (non-terminal) orders., Cancel every open order. Returns the count of successful cancels., Poll the broker for each open order's current status. Orders that reach a…

### Community 83 - "_resolve_child_ids"
Cohesion: 0.33
Nodes (4): Extract left / right child node ids with cross-format fallback., _resolve_child_ids(), _resolve_child_ids extracts left/right child IDs from node dict., _resolve_child_ids returns (-1, -1) for missing keys.

### Community 84 - "sharpe_ratio_torch"
Cohesion: 0.40
Nodes (4): Tensor, Shared quantitative metrics used across training and loss modules., Annualised Sharpe ratio from a PyTorch tensor of returns. Computes…, sharpe_ratio_torch()

### Community 85 - "GradHessOutput"
Cohesion: 0.40
Nodes (5): GradHessOutput, gradients, hessians, loss_value, Tensor

### Community 86 - "GBDTFeatureConfig"
Cohesion: 0.60
Nodes (3): GBDTFeatureConfig, Configuration for the GBDT feature engineering pipeline. Note: aggregation…, TestGBDTFeatureConfig

### Community 87 - "apply_curriculum_dropout"
Cohesion: 0.40
Nodes (4): apply_curriculum_dropout(), Tensor, Progressive curriculum dropout for Variant C. During training, CTM predictions…, Apply progressive curriculum dropout to CTM regression predictions. With…

### Community 88 - "associative_scan"
Cohesion: 0.67
Nodes (3): associative_scan(), Tensor, Parallel associative scan using log-space cumulative products. Solves the…

## Knowledge Gaps
- **131 isolated node(s):** `build_tree`, `config_`, `hist_builder_`, `split_finder_`, `rng_` (+126 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `compute_all_features()` connect `compute_all_features` to `train_volatility.py`, `ensemble_trainer.py`, `main`, `benchmark_ridge_vs_glauben.py`, `train.py`, `portfolio_optimizer.py`, `pipeline_v3.py`, `enhanced_features.py`, `data/__init__.py`, `diagnose_v2.py`, `volatility_backtest.py`, `test_data.py`, `GlaubenskriegPredictor`, `garch_baseline.py`, `pipeline_v2.py`, `train_gbdt_only.py`, `glauben_v2_noise_free.py`, `FeaturePipeline`?**
  _High betweenness centrality (0.236) - this node is a cross-community bridge._
- **Why does `CTMStockModel` connect `CTMStockModel` to `EnsembleWalkForwardTrainer`, `ensemble_trainer.py`, `LossConfig`, `train.py`, `RecurrentCTM`, `validate`, `Tensor`, `tests/conftest.py`, `glauben_v2_noise_free.py`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Why does `pytest_configure()` connect `test_tree.cpp` to `tests/conftest.py`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `CTMStockModel` (e.g. with `DecoupledPredictor` and `FrozenSSMPredictor`) actually correct?**
  _`CTMStockModel` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `AlpacaBroker` (e.g. with `AccountInfo` and `BaseBroker`) actually correct?**
  _`AlpacaBroker` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `TradingAccount` (e.g. with `BaseBroker` and `Order`) actually correct?**
  _`TradingAccount` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `MockBroker` (e.g. with `AccountInfo` and `BaseBroker`) actually correct?**
  _`MockBroker` has 18 INFERRED edges - model-reasoned connections that need verification._