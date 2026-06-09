"""CTM stock prediction: end-to-end training pipeline.

Usage:
    PYTHONPATH=. python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset



import yaml
from src.data.features import compute_all_features, compute_forward_returns
from src.data.dataset import create_sequences, train_val_test_split
from src.model.ctm_model import CTMStockModel
from src.model.multiasset_ctm import MultiAssetCTM
from src.model.loop_ctm import RecurrentCTM
from src.model.losses import LossConfig
from src.train.advanced_trainer import (
    LossWrapper,
    WalkForwardTrainerAdvanced,
    validate_advanced,
)
from src.train.curriculum import apply_curriculum_dropout


_KNOWN_TRAINER_KEYS = {
    "n_epochs", "batch_size", "lr", "weight_decay", "grad_clip",
    "patience", "step_size", "train_window", "val_window", "purge_period",
    "warmup_steps", "ramp_steps", "lr_warmup_epochs", "test_n_epochs",
}


def _warn_unknown_keys(cfg: dict, known_keys: set, section: str) -> None:
    unknown = set(cfg.keys()) - known_keys
    if unknown:
        import warnings
        warnings.warn(f"Unknown config keys in {section}: {unknown}")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "trainer" in cfg:
        _warn_unknown_keys(cfg["trainer"], _KNOWN_TRAINER_KEYS, "trainer")
    return cfg


def compute_class_weights(
    targets: pd.Series, eps: float = 1e-6
) -> torch.Tensor | None:
    """Compute inverse-frequency class weights from target returns.

    Returns None if only one class is present.
    """
    dir_labels = np.sign(targets).astype(int) + 1  # {-1,0,1} → {0,1,2}
    counts = np.bincount(dir_labels, minlength=3)
    total = len(dir_labels)
    weights = total / (3.0 * (counts + eps))
    if (counts > 0).sum() <= 1:
        return None
    return torch.tensor(weights, dtype=torch.float32)


def class_targets_fn(targets: torch.Tensor) -> torch.Tensor:
    """Map regression targets to directional class labels {0,1,2} for directional loss."""
    return torch.sign(targets).long() + 1


def _build_model_params(
    cfg: Dict[str, Any],
    input_dim: int,
    model_type: str,
    n_assets: int | None = None,
    n_loop: int | None = None,
    embedding_dim: int | None = None,
    use_cross_attention: bool | None = None,
    use_time_gate: bool = False,
) -> Dict[str, Any]:
    """Build model constructor params from config and overrides.

    Parameters
    ----------
    cfg : full config dict.
    input_dim : number of input features.
    model_type : "multiasset", "recurrent", or "ctm".
    n_assets : required for multiasset.
    n_loop : required for recurrent (>= 2).
    embedding_dim, use_cross_attention, use_time_gate : multiasset overrides.
    """
    model_cfg = cfg["model"]
    scaling_cfg = cfg.get("scaling", {})

    common = {
        "input_dim": input_dim,
        "model_dim": model_cfg.get("model_dim", 64),
        "state_dim": model_cfg.get("state_dim", 16),
        "conv_kernel": model_cfg.get("conv_kernel", 3),
        "n_layers": model_cfg.get("n_layers", 3),
        "output_dim": model_cfg.get("output_dim", 1),
        "dropout": model_cfg.get("dropout", 0.2),
    }

    if model_type == "multiasset":
        return {
            "n_assets": n_assets or scaling_cfg.get("n_assets", 1),
            "input_dim": input_dim,
            "model_dim": model_cfg.get("model_dim", 64),
            "state_dim": model_cfg.get("state_dim", 16),
            "n_layers": model_cfg.get("n_layers", 3),
            "output_dim": model_cfg.get("output_dim", 1),
            "embedding_dim": embedding_dim if embedding_dim is not None else scaling_cfg.get("embedding_dim", None),
            "use_cross_attention": use_cross_attention if use_cross_attention is not None else scaling_cfg.get("use_cross_attention", True),
            "dropout": model_cfg.get("dropout", 0.2),
            "conv_kernel": model_cfg.get("conv_kernel", 3),
            "use_decomp": model_cfg.get("use_decomp", False),
            "bidirectional": model_cfg.get("bidirectional", False),
            "parallel_scan": model_cfg.get("parallel_scan", True),
            "return_hidden": model_cfg.get("return_hidden", False),
            "use_time_gate": use_time_gate,
        }
    else:
        base = {
            **common,
            "use_decomp": model_cfg.get("use_decomp", False),
            "bidirectional": model_cfg.get("bidirectional", False),
            "parallel_scan": model_cfg.get("parallel_scan", True),
            "return_hidden": model_cfg.get("return_hidden", False),
        }
        if model_type == "recurrent":
            return {
                **base,
                "n_loop_iters": n_loop,
                "loop_dropout": model_cfg.get("loop_dropout", 0.1),
            }
        else:
            return base


def load_multi_asset_data(
    data_dir: str,
    seq_len: int,
    target_col: str = "forward_return",
    max_assets: int | None = None,
    periods: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load multiple stock CSV files and align them for MultiAssetCTM training.

    Each CSV must have columns: date,open,high,low,close,volume (date as index).

    Returns
    -------
    data_seq : (B, N, T, D) numpy array — features
    target_seq : (B, T, N) numpy array — forward_return targets
    asset_names : list of str — stock identifiers
    """
    import glob

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    if max_assets is not None:
        csv_files = csv_files[:max_assets]

    feature_dfs: list[pd.DataFrame] = []
    asset_names: list[str] = []

    for fpath in csv_files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        if df.empty or "close" not in df.columns:
            logging.warning("Skipping %s: invalid format", name)
            continue
        features = compute_all_features(df)
        close_t = torch.from_numpy(df["close"].values).float()
        features[target_col] = compute_forward_returns(close_t, periods=periods).numpy()
        features = features.dropna()
        if len(features) < seq_len + 10:
            logging.warning("Skipping %s: only %d rows after feature engineering", name, len(features))
            continue
        feature_dfs.append(features)
        asset_names.append(name)

    N = len(feature_dfs)
    if N < 2:
        raise ValueError(f"Need at least 2 assets for multi-asset training, got {N}")

    logging.info("Loaded %d assets from %s", N, data_dir)

    # Find common date intersection across all assets
    common_idx = feature_dfs[0].index
    for fdf in feature_dfs[1:]:
        common_idx = common_idx.intersection(fdf.index)

    logging.info(
        "Common date range: %s ~ %s (%d trading days)",
        common_idx.min(), common_idx.max(), len(common_idx),
    )

    # Align all assets to common dates
    feat_cols = [c for c in feature_dfs[0].columns if c != target_col]
    D = len(feat_cols)
    T = len(common_idx)

    feat_arrays: list[np.ndarray] = []
    targ_arrays: list[np.ndarray] = []
    for fdf in feature_dfs:
        aligned = fdf.loc[common_idx]
        feat_arrays.append(aligned[feat_cols].values.astype(np.float32))  # (T, D)
        targ_arrays.append(aligned[target_col].values.astype(np.float32))  # (T,)

    # Stack: (T, N, D) for features, (T, N) for targets
    feat_3d = np.stack(feat_arrays, axis=1)   # (T, N, D)
    targ_2d = np.stack(targ_arrays, axis=1)    # (T, N)

    # Create sequences via flatten → create_sequences → reshape
    # feat_3d: (T, N, D) → flatten to (T, N*D)
    flat_feat = feat_3d.reshape(T, N * D)
    seqs_feat = create_sequences(flat_feat, seq_len)          # (B, T, N*D)
    data_seq = seqs_feat.reshape(-1, seq_len, N, D).transpose(0, 2, 1, 3)  # (B, N, T, D)

    # Targets: (T, N) → sequences
    flat_targ = targ_2d.reshape(T, N)
    target_seq = create_sequences(flat_targ, seq_len)         # (B, T, N)

    logging.info("Multi-asset data: %s, targets: %s", data_seq.shape, target_seq.shape)
    # v4: Cross-sectional demeaning — isolate stock-specific alpha from market beta
    target_seq = target_seq - target_seq.mean(axis=-1, keepdims=True)
    logging.info("Targets cross-sectionally demeaned (shape: %s)", target_seq.shape)
    return data_seq, target_seq, asset_names


def main() -> None:
    parser = argparse.ArgumentParser(description="CTM stock prediction training")
    parser.add_argument("--config", default=None, help="Config YAML path (overrides --scale)")
    parser.add_argument("--scale", choices=["small", "large", "portfolio", "loop", "loop_large"], default=None,
                        help="Model scale: loads configs/scale_{scale}.yaml")
    parser.add_argument("--data", default=None, help="CSV path (overrides config)")
    parser.add_argument("--data-dir", default=None, help="Directory of CSV files for multi-asset training (requires --multi-asset)")
    parser.add_argument("--device", default=None, help="Device override (uses config YAML if not set)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", default="results/metrics.json", help="Metrics output path")
    parser.add_argument("--save-dir", default=None, help="Directory to save trained models")
    parser.add_argument("--multi-asset", action="store_true", help="Enable MultiAssetCTM for portfolio modeling")
    parser.add_argument("--n-assets", type=int, default=None, help="Number of assets (required for multi-asset mode)")
    parser.add_argument("--use-cross-attention", type=str, default=None, choices=["true", "false"], help="Enable cross-asset attention")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Asset embedding dimension")
    parser.add_argument("--ensemble", action="store_true", help="Enable CTM + GBDT ensemble mode")
    parser.add_argument("--p3", action="store_true", help="Enable P3 three-stage fusion (CTM→GBDT→modulator fine-tune)")
    parser.add_argument("--time-gate", action="store_true", help="Enable Variant A time-decay gate for CTM→GBDT progressive blending")
    parser.add_argument("--curriculum", action="store_true", help="Enable Variant C progressive curriculum dropout")
    parser.add_argument("--modulator-epochs", type=int, default=10, help="Stage 3 modulator fine-tuning epochs")
    parser.add_argument("--modulator-lr", type=float, default=1e-4, help="Stage 3 modulator learning rate")
    parser.add_argument("--modulator-patience", type=int, default=3, help="Stage 3 early stopping patience")
    parser.add_argument("--gbdt-trees", type=int, default=100, help="GBDT number of trees")
    parser.add_argument("--gbdt-depth", type=int, default=6, help="GBDT max depth")
    parser.add_argument("--gbdt-lr", type=float, default=0.1, help="GBDT learning rate")
    parser.add_argument("--gbdt-loss", type=str, default="mse", help="GBDT loss: mse, mae, huber, rankic")
    parser.add_argument("--gbdt-subsample", type=float, default=0.8, help="GBDT row subsample ratio")
    parser.add_argument("--gbdt-colsample", type=float, default=0.8, help="GBDT col subsample ratio")
    # ── TimeGate finetuning ──
    parser.add_argument("--finetune-gate", action="store_true",
                        help="Enable TimeDecayGate fine-tuning after CTM+GBDT training "
                             "(requires --time-gate)")
    parser.add_argument("--gate-finetune-epochs", type=int, default=50,
                        help="Gate finetuning epochs per window")
    parser.add_argument("--gate-finetune-lr", type=float, default=5e-4,
                        help="Gate finetuning learning rate")
    parser.add_argument("--gate-finetune-batch-size", type=int, default=32,
                        help="Gate finetuning batch size")
    parser.add_argument("--gate-finetune-patience", type=int, default=5,
                        help="Gate finetuning early stopping patience")
    parser.add_argument("--gate-freeze-backbone", action="store_true", default=True,
                        help="Freeze CTM backbone during gate finetuning (default: true)")
    parser.add_argument("--no-gate-freeze-backbone", action="store_false", dest="gate_freeze_backbone",
                        help="Unfreeze CTM backbone during gate finetuning")
    # Loss weight overrides (override config values when provided)
    parser.add_argument("--lambda-mse", type=float, default=None, help="MSE loss weight")
    parser.add_argument("--lambda-sharpe", type=float, default=None, help="Sharpe loss weight")
    parser.add_argument("--lambda-directional", type=float, default=None, help="Directional loss weight")
    parser.add_argument("--lambda-pinball", type=float, default=None, help="Pinball loss weight")
    parser.add_argument("--lambda-reg", type=float, default=None, help="L2 regularization weight")
    # Trainer hyperparameter overrides (override config YAML values)
    parser.add_argument("--train-window", type=int, default=None, help="Override train_window")
    parser.add_argument("--val-window", type=int, default=None, help="Override val_window")
    parser.add_argument("--step-size", type=int, default=None, help="Override step_size")
    parser.add_argument("--purge-period", type=int, default=None, help="Override purge_period")
    parser.add_argument("--n-epochs", type=int, default=None, help="Override n_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--n-loop", type=int, default=None,
                        help="Number of recurrent refinement loops (1=disable, 3=recommended)")
    parser.add_argument("--target-periods", type=int, default=5,
                        help="Forward return horizon in days (v4: 5-day for better SNR)")
    # Pretrained weight args for ensemble training
    parser.add_argument("--init-ctm-ckpt", type=str, default=None,
                        help="Path to pre-trained CTM checkpoint (.pt) for warm-start or inference")
    parser.add_argument("--init-gbdt-json", type=str, default=None,
                        help="Path to pre-trained GBDT model JSON for inference-only fusion")
    parser.add_argument("--skip-ctm-training", action="store_true",
                        help="Skip CTM training; use --init-ctm-ckpt weights as-is (inference mode)")
    parser.add_argument("--skip-gbdt-training", action="store_true",
                        help="Skip GBDT training; use --init-gbdt-json model as-is")
    parser.add_argument("--ensemble-inference", action="store_true",
                        help="Run inference-only ensemble fusion: load pre-trained CTM + GBDT, fuse, report metrics. "
                             "Requires --init-ctm-ckpt and --init-gbdt-json.")
    args = parser.parse_args()

    # ── Logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ── Reproducibility ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.info("Random seed set to %d", args.seed)

    if args.config is not None:
        config_path = args.config
    elif args.scale is not None:
        config_path = f"configs/scale_{args.scale}.yaml"
    else:
        config_path = "configs/default.yaml"
    cfg = load_config(config_path)

    # Apply CLI overrides to trainer config (if provided)
    trainer_cfg = cfg.setdefault("trainer", {})
    if args.train_window is not None:
        trainer_cfg["train_window"] = args.train_window
    if args.val_window is not None:
        trainer_cfg["val_window"] = args.val_window
    if args.step_size is not None:
        trainer_cfg["step_size"] = args.step_size
    if args.purge_period is not None:
        trainer_cfg["purge_period"] = args.purge_period
    if args.n_epochs is not None:
        trainer_cfg["n_epochs"] = args.n_epochs
    if args.batch_size is not None:
        trainer_cfg["batch_size"] = args.batch_size

    device = torch.device(args.device or cfg.get("device", "cpu"))

    # ── 1. Load data ──
    seq_len = cfg["model"]["seq_len"]
    target_col = cfg["data"].get("target_column", "forward_return")
    split = cfg["data"]["split"]
    train_frac = split.get("train", 0.7)
    val_frac = split.get("val", 0.15)

    is_portfolio = cfg.get("scale", "small") == "portfolio" or args.multi_asset

    if args.data_dir is not None and is_portfolio:
        # ── Multi-asset: load all CSVs from directory ──
        logging.info("Loading multi-asset data from %s", args.data_dir)
        n_assets_max = args.n_assets
        data_seq, target_seq, asset_names = load_multi_asset_data(
            args.data_dir, seq_len, target_col, max_assets=n_assets_max, periods=args.target_periods,
        )
        n_assets_portfolio = data_seq.shape[1]
        logging.info("Loaded %d assets, data: %s, targets: %s", n_assets_portfolio, data_seq.shape, target_seq.shape)

        # Temporal train/val/test split
        train_seq, val_seq, test_seq = train_val_test_split(data_seq, train_frac, val_frac)
        train_targ, val_targ, test_targ = train_val_test_split(target_seq, train_frac, val_frac)
        # Convert to tensors
        train_seq = torch.from_numpy(train_seq)
        train_targ = torch.from_numpy(train_targ)
        val_seq = torch.from_numpy(val_seq)
        val_targ = torch.from_numpy(val_targ)
        test_seq = torch.from_numpy(test_seq)
        test_targ = torch.from_numpy(test_targ)
        input_dim = data_seq.shape[-1]
        # Class weights (computed on training split only to prevent leakage)
        class_weights = compute_class_weights(pd.Series(train_targ.numpy().ravel()))
        if class_weights is not None:
            logging.info("Class weights: %s", class_weights.numpy().round(3))
        else:
            logging.info("No class weighting (single-class target)")
        logging.info("Multi-asset — input_dim=%d, n_assets=%d", input_dim, n_assets_portfolio)

    else:
        # ── Single-stock: original loading path ──
        data_path = args.data or cfg["data"]["csv_path"]
        logging.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Feature engineering
        logging.info("Computing features...")
        features_df = compute_all_features(df)

        close_prices = torch.from_numpy(df["close"].values).float()
        features_df["forward_return"] = compute_forward_returns(close_prices, periods=args.target_periods).numpy()  # v4: multi-day horizon
        features_df = features_df.dropna()

        feature_cols = [c for c in features_df.columns if c != target_col]
        feature_array = features_df[feature_cols].values.astype(np.float32)
        target_array = features_df[target_col].values.astype(np.float32)
        n_features_found = len(feature_cols)
        logging.info("%d features, %d rows", n_features_found, len(features_df))

        # Create sequences
        logging.info("Creating sequences (T=%d)...", seq_len)
        data_seq = create_sequences(feature_array, seq_len)
        target_seq = create_sequences(target_array, seq_len)
        target_seq = np.expand_dims(target_seq, -1)
        logging.info("%d sequences, shape: %s", len(data_seq), data_seq.shape)

        # Train/val/test split
        train_seq, val_seq, test_seq = train_val_test_split(data_seq, train_frac, val_frac)
        train_targ, val_targ, test_targ = train_val_test_split(target_seq, train_frac, val_frac)
        logging.info("Train: %d, Val: %d, Test: %d", len(train_seq), len(val_seq), len(test_seq))

        # Convert to tensors
        train_seq = torch.from_numpy(train_seq)
        train_targ = torch.from_numpy(train_targ)
        val_seq = torch.from_numpy(val_seq)
        val_targ = torch.from_numpy(val_targ)

        # Detect input_dim
        input_dim = train_seq.shape[-1]
        logging.info("Auto-detected input_dim=%d", input_dim)

        # Class weights (computed on training split only to prevent leakage)
        class_weights = compute_class_weights(pd.Series(train_targ.numpy().ravel()))
        if class_weights is not None:
            logging.info("Class weights: %s", class_weights.numpy().round(3))
        else:
            logging.info("No class weighting (single-class target)")

        # Portfolio reshaping (expand single-stock → N assets)
        if is_portfolio:
            n_assets_portfolio = args.n_assets or cfg.get("scaling", {}).get("n_assets", 1)
            test_seq = torch.from_numpy(test_seq)
            test_targ = torch.from_numpy(test_targ)
            logging.info("Reshaping for portfolio mode: n_assets=%d", n_assets_portfolio)
            train_seq = train_seq.unsqueeze(1).expand(-1, n_assets_portfolio, -1, -1)
            val_seq = val_seq.unsqueeze(1).expand(-1, n_assets_portfolio, -1, -1)
            test_seq = test_seq.unsqueeze(1).expand(-1, n_assets_portfolio, -1, -1)
            train_targ = train_targ.expand(-1, -1, n_assets_portfolio)
            val_targ = val_targ.expand(-1, -1, n_assets_portfolio)
            test_targ = test_targ.expand(-1, -1, n_assets_portfolio)
            logging.info("Reshaped data: %s, targets: %s", train_seq.shape, train_targ.shape)

    if not is_portfolio:
        n_assets_portfolio = 1
    else:
        n_assets_portfolio = train_seq.shape[1]

    # ── 8. Build model config ──
    model_cfg = cfg["model"]

    if is_portfolio:
        use_ca = None
        if args.use_cross_attention is not None:
            use_ca = args.use_cross_attention == "true"
        model_params = _build_model_params(
            cfg, input_dim, "multiasset",
            n_assets=args.n_assets,
            embedding_dim=args.embedding_dim,
            use_cross_attention=use_ca,
            use_time_gate=args.time_gate,
        )
        model_class = MultiAssetCTM
    else:
        n_loop_iters = args.n_loop if args.n_loop is not None else model_cfg.get("n_loop_iters", 1)
        if n_loop_iters > 1:
            model_params = _build_model_params(cfg, input_dim, "recurrent", n_loop=n_loop_iters)
            model_class = RecurrentCTM
        else:
            model_params = _build_model_params(cfg, input_dim, "ctm")
            model_class = CTMStockModel

    # ── 8b. Log model parameter count for overfitting awareness ──
    _dummy_model = model_class(**model_params).to(device)
    n_params = sum(p.numel() for p in _dummy_model.parameters() if p.requires_grad)
    n_train = len(train_seq)
    logging.info(
        "Model: %s, params=%d, n_train=%d, param/seq ratio=%.1f:1",
        model_class.__name__, n_params, n_train, n_params / max(n_train, 1)
    )
    del _dummy_model

    # ── 9. Build loss config (CLI args override config values) ──
    loss_cfg_data = cfg.get("loss", {})
    lambda_mse = args.lambda_mse if args.lambda_mse is not None else loss_cfg_data.get("lambda_mse", 1.0)
    lambda_sharpe = args.lambda_sharpe if args.lambda_sharpe is not None else loss_cfg_data.get("lambda_sharpe", 0.5)
    lambda_directional = args.lambda_directional if args.lambda_directional is not None else loss_cfg_data.get("lambda_directional", 1.0)
    lambda_pinball = args.lambda_pinball if args.lambda_pinball is not None else loss_cfg_data.get("lambda_pinball", 0.1)
    lambda_reg = args.lambda_reg if args.lambda_reg is not None else loss_cfg_data.get("lambda_reg", 0.01)
    skip_l2_reg = loss_cfg_data.get("skip_l2_reg", False)
    loss_config = LossConfig(
        lambda_mse=lambda_mse,
        lambda_sharpe=lambda_sharpe,
        lambda_directional=lambda_directional,
        lambda_pinball=lambda_pinball,
        lambda_reg=lambda_reg,
        pinball_tau=loss_cfg_data.get("pinball_tau", 0.05),
        skip_l2_reg=skip_l2_reg,
        class_weight=class_weights,
    )

    # ── 10. Load pre-trained weights (if provided) ──
    pretrained_ctm_sd = None
    if args.skip_ctm_training and args.init_ctm_ckpt is None:
        raise ValueError("--skip-ctm-training requires --init-ctm-ckpt")
    if args.skip_gbdt_training and args.init_gbdt_json is None:
        raise ValueError("--skip-gbdt-training requires --init-gbdt-json")
    if args.ensemble_inference and args.init_ctm_ckpt is None:
        raise ValueError("--ensemble-inference requires --init-ctm-ckpt")
    if args.ensemble_inference and args.init_gbdt_json is None:
        raise ValueError("--ensemble-inference requires --init-gbdt-json")

    if args.init_ctm_ckpt is not None:
        ckpt_path = args.init_ctm_ckpt
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Pre-trained CTM checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Support both full checkpoint (with "model_state_dict" key) and bare state_dict
        if "model_state_dict" in ckpt:
            pretrained_ctm_sd = ckpt["model_state_dict"]
            logging.info(
                "Loaded pre-trained CTM from full checkpoint (epoch=%s, config keys=%d)",
                ckpt.get("epoch", "?"), len(ckpt.get("config", {})),
            )
        else:
            pretrained_ctm_sd = ckpt
            logging.info("Loaded pre-trained CTM state_dict from %s", ckpt_path)

    # ── 10b. Walk-forward training ──
    trainer_cfg = cfg["trainer"]
    trainer = None
    if args.ensemble or args.ensemble_inference:
        # Ensemble inference mode implies ensemble
        is_ensemble = True
        try:
            from src.train.ensemble_trainer import EnsembleWalkForwardTrainer
        except ImportError as e:
            logging.warning("Could not import EnsembleWalkForwardTrainer: %s", e)
            logging.warning("Falling back to CTM-only mode.")
            is_ensemble = False
            args.ensemble = False
            args.ensemble_inference = False
        else:
            gbdt_cfg = {
                "num_trees": args.gbdt_trees,
                "max_depth": args.gbdt_depth,
                "learning_rate": args.gbdt_lr,
                "subsample_row": args.gbdt_subsample,
                "subsample_col": args.gbdt_colsample,
                "include_ctm_features": True,
                "random_seed": args.seed,
            }
            loss_fn = None
            if args.gbdt_loss == "composite":
                from src.train.loss_bridge import make_gbdt_loss_fn
                loss_fn = make_gbdt_loss_fn(loss_config)

            # Determine skip flags from CLI or inference mode
            skip_ctm = args.skip_ctm_training or args.ensemble_inference
            skip_gbdt = args.skip_gbdt_training or args.ensemble_inference

            # ── Gate finetuning args ──
            gate_ft_epochs = args.gate_finetune_epochs if args.finetune_gate else 0
            gate_ft_lr = args.gate_finetune_lr
            gate_ft_batch = args.gate_finetune_batch_size
            gate_ft_patience = args.gate_finetune_patience
            gate_freeze_backbone = args.gate_freeze_backbone

            if args.p3 and not args.ensemble_inference:
                from src.train.p3_trainer import P3EnsembleTrainer
                trainer = P3EnsembleTrainer(
                    model_class=model_class,
                    model_params=model_params,
                    loss_config=loss_config,
                    gbdt_config=gbdt_cfg,
                    gbdt_loss=args.gbdt_loss,
                    device=device,
                    loss_fn=loss_fn,
                    modulator_epochs=args.modulator_epochs,
                    modulator_lr=args.modulator_lr,
                    modulator_patience=args.modulator_patience,
                    pretrained_ctm_state_dict=pretrained_ctm_sd,
                    pretrained_gbdt_json=args.init_gbdt_json,
                    skip_ctm_training=skip_ctm,
                    skip_gbdt_training=skip_gbdt,
                    gate_finetune_epochs=gate_ft_epochs,
                    gate_finetune_lr=gate_ft_lr,
                    freeze_ctm_backbone_for_gate=gate_freeze_backbone,
                    gate_finetune_batch_size=gate_ft_batch,
                    gate_finetune_patience=gate_ft_patience,
                )
            else:
                trainer = EnsembleWalkForwardTrainer(
                    model_class=model_class,
                    model_params=model_params,
                    loss_config=loss_config,
                    gbdt_config=gbdt_cfg,
                    gbdt_loss=args.gbdt_loss,
                    device=device,
                    loss_fn=loss_fn,
                    pretrained_ctm_state_dict=pretrained_ctm_sd,
                    pretrained_gbdt_json=args.init_gbdt_json,
                    skip_ctm_training=skip_ctm,
                    skip_gbdt_training=skip_gbdt,
                    gate_finetune_epochs=gate_ft_epochs,
                    gate_finetune_lr=gate_ft_lr,
                    freeze_ctm_backbone_for_gate=gate_freeze_backbone,
                    gate_finetune_batch_size=gate_ft_batch,
                    gate_finetune_patience=gate_ft_patience,
                )
    if not args.ensemble and not args.ensemble_inference:
        trainer = WalkForwardTrainerAdvanced(
            model_class=model_class,
            model_params=model_params,
            loss_config=loss_config,
            device=device,
        )

    if trainer is None:
        raise RuntimeError("Trainer was not initialized (import error?)")

    # Extract reusable training hyperparameters
    train_window_val = trainer_cfg.get("train_window", max(len(train_seq) // 3, 1))
    val_window_val = trainer_cfg.get("val_window", max(len(val_seq) // 3, 1))
    purge_period = trainer_cfg.get("purge_period", max(seq_len * 2, 63))
    step_size = trainer_cfg.get("step_size", 21)
    n_epochs = trainer_cfg.get("n_epochs", 100)
    batch_size = trainer_cfg.get("batch_size", 32)
    lr = trainer_cfg.get("lr", 3e-4)
    weight_decay = trainer_cfg.get("weight_decay", 0.15)
    grad_clip = trainer_cfg.get("grad_clip", 1.0)
    patience = trainer_cfg.get("patience", 10)

    # Clamp lr_warmup_epochs to avoid exceeding available epochs
    lr_warmup_epochs = trainer_cfg.get("lr_warmup_epochs", 200)
    if lr_warmup_epochs > max(5, n_epochs - 10):
        clamped = max(5, n_epochs - 10)
        logging.warning(
            "lr_warmup_epochs=%d clamped to %d (n_epochs=%d)",
            lr_warmup_epochs, clamped, n_epochs,
        )
        trainer_cfg["lr_warmup_epochs"] = clamped

    if args.ensemble_inference:
        from src.train.ensemble_trainer import EnsembleWalkForwardTrainer
        assert isinstance(trainer, EnsembleWalkForwardTrainer), \
            "Ensemble inference requires EnsembleWalkForwardTrainer"
        logging.info("Running ensemble inference with pre-trained models...")
        val_data_np = val_seq.numpy() if torch.is_tensor(val_seq) else val_seq
        val_target_np = val_targ.numpy() if torch.is_tensor(val_targ) else val_targ
        val_tensor = torch.from_numpy(val_data_np).float()
        target_tensor = torch.from_numpy(val_target_np).float()
        summary = trainer.run_ensemble_inference(
            data=val_tensor,
            targets=target_tensor,
            batch_size=trainer_cfg.get("batch_size", 32),
        )
        logging.info(
            "Ensemble inference complete: fused_IC=%.4f, CTM sharpe=%.4f, GBDT sharpe=%.4f",
            summary.get("fused_ic", 0),
            summary.get("ctm_sharpe", 0),
            summary.get("gbdt_sharpe", 0),
        )
        results = [summary]  # Placeholder for downstream empty-check
    else:
        logging.info("Starting walk-forward training...")
        results = trainer.train_walk_forward(
            data=train_seq,
            targets=train_targ,
            train_window=train_window_val,
            val_window=val_window_val,
            purge_period=purge_period,
            step_size=step_size,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            patience=patience,
            warmup_steps=trainer_cfg.get("warmup_steps", 2000),
            ramp_steps=trainer_cfg.get("ramp_steps", 3000),
            lr_warmup_epochs=trainer_cfg.get("lr_warmup_epochs", 200),
            class_targets_fn=class_targets_fn,
        )

    if not results:
        logging.warning("No walk-forward windows fit the data. Try reducing train_window/val_window.")
        summary: Dict[str, Any] = {
            "n_windows": 0,
            "error": "No walk-forward windows fit data. Reduce train_window/val_window.",
        }
        out_path = cfg.get("output", {}).get("metrics_path", "results/metrics.json")
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logging.warning("Empty results summary saved to %s", out_path)
        return

    # ── Ensemble inference fast path: summary already built ──
    if args.ensemble_inference:
        summary = results[0]  # results[0] is the dict from run_ensemble_inference
        out_path = args.output if args.output != parser.get_default("output") else cfg.get("output", {}).get("metrics_path", "results/metrics.json")
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logging.info("Ensemble inference results saved to %s", out_path)
        return

    # ── 11. Summarize results ──
    logging.info("=== WALK-FORWARD RESULTS ===")
    sharpe_values = [r.best_sharpe for r in results]
    for i, r in enumerate(results):
        if args.ensemble:
            if args.p3:
                logging.info(
                    "  Window %d: start=%d, ctm=%.4f, gbdt_ic=%.4f, "
                    "ens=%.4f, p3=%.4f, p3_ens=%.4f, epochs=%d",
                    i, r.window_start,
                    r.ctm_sharpe or 0, r.gbdt_ic or 0,
                    r.ensemble_sharpe or 0,
                    r.p3_sharpe or 0, r.p3_ensemble_sharpe or 0,
                    r.epochs_run,
                )
            else:
                logging.info(
                    "  Window %d: start=%d, ctm_sharpe=%.4f, gbdt_ic=%.4f, "
                    "ensemble_sharpe=%.4f, ctm_weight=%.3f, epochs=%d",
                    i,
                    r.window_start,
                    r.ctm_sharpe or 0,
                    r.gbdt_ic or 0,
                    r.ensemble_sharpe or 0,
                    r.ctm_weight or 1.0,
                    r.epochs_run,
                )
        else:
            logging.info(
                "  Window %d: start=%d, best_sharpe=%.4f, epochs=%d",
                i,
                r.window_start,
                r.best_sharpe,
                r.epochs_run,
            )

    summary: Dict[str, Any] = {
        "n_windows": len(results),
        "mean_sharpe": float(np.mean(sharpe_values)),
        "std_sharpe": float(np.std(sharpe_values)),
        "max_sharpe": float(np.max(sharpe_values)),
        "min_sharpe": float(np.min(sharpe_values)),
        "model_params": model_params,
        "results": [],
    }
    _ctm_ic_all = [r.ctm_ic for r in results if r.ctm_ic is not None]
    if _ctm_ic_all:
        summary["mean_ctm_ic"] = float(np.mean(_ctm_ic_all))
        summary["std_ctm_ic"] = float(np.std(_ctm_ic_all))
    if args.ensemble:
        _ctm_s = [r.ctm_sharpe for r in results if r.ctm_sharpe is not None]
        _ctm_i = [r.ctm_ic for r in results if r.ctm_ic is not None]
        _ctm_d = [r.ctm_dir_acc for r in results if r.ctm_dir_acc is not None]
        _gbdt_s = [r.gbdt_sharpe for r in results if r.gbdt_sharpe is not None]
        _gbdt_i = [r.gbdt_ic for r in results if r.gbdt_ic is not None]
        _gbdt_d = [r.gbdt_dir_acc for r in results if r.gbdt_dir_acc is not None]
        _ens_s  = [r.ensemble_sharpe for r in results if r.ensemble_sharpe is not None]
        _ens_i  = [r.ensemble_ic for r in results if r.ensemble_ic is not None]
        _ens_d  = [r.ensemble_dir_acc for r in results if r.ensemble_dir_acc is not None]
        summary["mean_ctm_sharpe"] = float(np.mean(_ctm_s)) if _ctm_s else 0.0
        summary["mean_ctm_ic"] = float(np.mean(_ctm_i)) if _ctm_i else 0.0
        summary["std_ctm_ic"] = float(np.std(_ctm_i)) if _ctm_i else 0.0
        summary["mean_ctm_dir_acc"] = float(np.mean(_ctm_d)) if _ctm_d else 0.0
        summary["mean_gbdt_sharpe"] = float(np.mean(_gbdt_s)) if _gbdt_s else 0.0
        summary["std_gbdt_sharpe"] = float(np.std(_gbdt_s)) if _gbdt_s else 0.0
        summary["mean_gbdt_ic"] = float(np.mean(_gbdt_i)) if _gbdt_i else 0.0
        summary["mean_gbdt_dir_acc"] = float(np.mean(_gbdt_d)) if _gbdt_d else 0.0
        summary["mean_ensemble_sharpe"] = float(np.mean(_ens_s)) if _ens_s else 0.0
        summary["mean_ensemble_ic"] = float(np.mean(_ens_i)) if _ens_i else 0.0
        summary["mean_ensemble_dir_acc"] = float(np.mean(_ens_d)) if _ens_d else 0.0
        _ctm_w = [r.ctm_weight for r in results if r.ctm_weight is not None]
        summary["mean_ctm_weight"] = float(np.mean(_ctm_w)) if _ctm_w else 1.0
        if args.p3:
            _p3_s = [r.p3_sharpe for r in results if r.p3_sharpe is not None]
            _p3_i = [r.p3_ic for r in results if r.p3_ic is not None]
            _p3_e = [r.p3_ensemble_sharpe for r in results if r.p3_ensemble_sharpe is not None]
            summary["mean_p3_sharpe"] = float(np.mean(_p3_s)) if _p3_s else 0.0
            summary["mean_p3_ic"] = float(np.mean(_p3_i)) if _p3_i else 0.0
            summary["mean_p3_ensemble_sharpe"] = float(np.mean(_p3_e)) if _p3_e else 0.0
    for r in results:
        entry: Dict[str, Any] = {
            "window_start": r.window_start,
            "window_end": r.window_end,
            "best_sharpe": r.best_sharpe,
            "epochs_run": r.epochs_run,
        }
        if not args.ensemble:
            entry["ctm_ic"] = r.ctm_ic if r.ctm_ic is not None else 0.0
        if args.ensemble:
            entry["ctm_sharpe"] = r.ctm_sharpe if r.ctm_sharpe is not None else 0.0
            entry["ctm_ic"] = r.ctm_ic if r.ctm_ic is not None else 0.0
            entry["ctm_dir_acc"] = r.ctm_dir_acc if r.ctm_dir_acc is not None else 0.0
            entry["gbdt_sharpe"] = r.gbdt_sharpe if r.gbdt_sharpe is not None else 0.0
            entry["gbdt_ic"] = r.gbdt_ic if r.gbdt_ic is not None else 0.0
            entry["gbdt_dir_acc"] = r.gbdt_dir_acc if r.gbdt_dir_acc is not None else 0.0
            entry["ensemble_sharpe"] = r.ensemble_sharpe if r.ensemble_sharpe is not None else 0.0
            entry["ensemble_ic"] = r.ensemble_ic if r.ensemble_ic is not None else 0.0
            entry["ensemble_dir_acc"] = r.ensemble_dir_acc if r.ensemble_dir_acc is not None else 0.0
            entry["ctm_weight"] = r.ctm_weight if r.ctm_weight is not None else 1.0
            if args.p3:
                entry["p3_sharpe"] = r.p3_sharpe if r.p3_sharpe is not None else 0.0
                entry["p3_ic"] = r.p3_ic if r.p3_ic is not None else 0.0
                entry["p3_ensemble_sharpe"] = r.p3_ensemble_sharpe if r.p3_ensemble_sharpe is not None else 0.0
        summary["results"].append(entry)

    if hasattr(trainer, 'explain') and results:
        try:
            explanation = trainer.explain(results, top_k=10)
            if "error" not in explanation:
                summary["feature_importance"] = explanation["aggregated"]
                top5 = explanation["aggregated"][:5]
                logging.info("Top-5 features: %s",
                             [f["feature_idx"] for f in top5])
        except Exception as e:
            logging.warning("explain() failed: %s", e)

    # ── 12. Final evaluation on test set ──
    # NOTE: This trains a new model on train+val and evaluates on the held-out test set.
    # This is a simplified approach; the proper solution is to use the best walk-forward
    # model from each window to predict on the test set directly.
    logging.info("Running final evaluation on held-out test set...")
    if isinstance(test_seq, np.ndarray):
        test_seq_pt = torch.from_numpy(test_seq)
        test_targ_pt = torch.from_numpy(test_targ)
    else:
        test_seq_pt = test_seq
        test_targ_pt = test_targ

    # Combine train+val for final model training
    full_train_data = torch.cat([train_seq, val_seq], dim=0)
    full_train_targ = torch.cat([train_targ, val_targ], dim=0)

    # Split combined data: 80% train, 20% validation for early stopping
    n_full = len(full_train_data)
    n_final_train = int(n_full * 0.8)
    final_train_data = full_train_data[:n_final_train]
    final_train_targ = full_train_targ[:n_final_train]
    final_val_data = full_train_data[n_final_train:]
    final_val_targ = full_train_targ[n_final_train:]

    final_train_ds = TensorDataset(final_train_data, final_train_targ)
    final_val_ds = TensorDataset(final_val_data, final_val_targ)
    test_ds = TensorDataset(test_seq_pt, test_targ_pt)

    # Time-series data: shuffle=False for all loaders
    final_train_loader = DataLoader(final_train_ds, batch_size=batch_size, shuffle=False)
    final_val_loader = DataLoader(final_val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if args.p3 and hasattr(trainer, 'get_stage3_model'):
        stage3 = trainer.get_stage3_model()
        if stage3 is not None:
            final_model = stage3
        else:
            final_model = model_class(**model_params).to(device)
    else:
        final_model = model_class(**model_params).to(device)
        # P0 Fix 2: Warm-start test-prep from best walk-forward window state dict.
        # Without this, test-prep trains a randomly-initialised model on train+val
        # data, losing all information learned during walk-forward validation.
        _best_init_sd = None
        _best_idx = -1
        if results:
            if args.ensemble:
                ic_values = [
                    r.ensemble_ic if r.ensemble_ic is not None else float('-inf')
                    for r in results
                ]
            else:
                ic_values = [
                    r.ctm_ic if r.ctm_ic is not None else float('-inf')
                    for r in results
                ]
            if ic_values and any(v != float('-inf') for v in ic_values):
                _best_idx = int(np.nanargmax(ic_values))
                if hasattr(trainer, 'get_best_ctm_state_dict'):
                    _best_init_sd = trainer.get_best_ctm_state_dict(_best_idx)
        if _best_init_sd is not None:
            try:
                # strict=False: architectural differences across windows tolerated
                final_model.load_state_dict(_best_init_sd, strict=False)
                logging.info(
                    "Test-prep warm-start: loaded state dict from best "
                    "walk-forward window %d (IC=%.4f)",
                    _best_idx, max(ic_values)
                )
            except Exception as _exc:
                logging.warning(
                    "Warm-start state dict load failed (%s), "
                    "falling back to random initialisation", _exc
                )
    optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    final_loss_wrapper = LossWrapper(
        config=deepcopy(loss_config),
        model=final_model,
        class_targets_fn=class_targets_fn,
    ).to(device)

    test_n_epochs = trainer_cfg.get("test_n_epochs", min(n_epochs, 50))
    test_patience = max(5, patience // 2)
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    _cur_n_assets = getattr(final_model, 'n_assets', 1)
    _cur_output_dim = getattr(final_model, 'output_dim', model_params.get('output_dim', 1))
    for epoch in range(test_n_epochs):
        final_model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in final_train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = final_model(batch_x)
            # NOTE: curriculum dropout is NOT applied during test-prep —
            # it would corrupt the final model (late-horizon 80% dropout →
            # model collapses to near-zero predictions).
            loss = final_loss_wrapper(pred, batch_y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Early stopping: evaluate on validation portion
        final_model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in final_val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = final_model(batch_x)
                val_loss += final_loss_wrapper(pred, batch_y).item()
                val_batches += 1
        val_loss /= max(val_batches, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info("  Test-prep epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                          epoch + 1, test_n_epochs,
                          epoch_loss / max(n_batches, 1), val_loss)

        if patience_counter >= test_patience:
            logging.info("  Early stopping at epoch %d (best_val_loss=%.6f)",
                          epoch + 1, best_val_loss)
            break

    # Restore best model for final test evaluation
    if best_model_state is not None:
        final_model.load_state_dict(best_model_state)

    test_metrics = validate_advanced(final_model, test_loader, final_loss_wrapper, device)
    summary["test_metrics"] = {
        "test_loss": test_metrics["avg_loss"],
        "test_sharpe": test_metrics["sharpe_ratio"],
        "test_directional_accuracy": test_metrics["directional_accuracy"],
    }
    logging.info(
        "Test set — loss=%.6f, sharpe=%.4f, dir_acc=%.4f",
        test_metrics["avg_loss"],
        test_metrics["sharpe_ratio"],
        test_metrics["directional_accuracy"],
    )

    # ── 13. Save final model ──
    if args.save_dir:
        from src.utils.serialization import save_ensemble, save_ctm_model, save_ensemble_trainer_state
        try:
            gbdt_model = getattr(trainer, "_last_gbdt_model", None) if args.ensemble else None
            if args.ensemble and gbdt_model is not None:
                saved_paths = save_ensemble(
                    final_model, gbdt_model,
                    args.save_dir, config=cfg,
                )
                summary["saved_models"] = saved_paths
                # Save full trainer state bundle for resume/inference
                trainer_state_paths = save_ensemble_trainer_state(
                    ctm_model=final_model,
                    gbdt_model=gbdt_model,
                    save_dir=args.save_dir,
                    model_params=model_params,
                    config=cfg,
                    loss_config=loss_config,
                    gbdt_config={
                        "num_trees": args.gbdt_trees,
                        "max_depth": args.gbdt_depth,
                        "learning_rate": args.gbdt_lr,
                        "subsample_row": args.gbdt_subsample,
                        "subsample_col": args.gbdt_colsample,
                    },
                    gbdt_loss=args.gbdt_loss,
                    model_name="ensemble",
                )
                summary["saved_trainer_state"] = trainer_state_paths.get("trainer_state")
                logging.info("Ensemble trainer state saved to %s", trainer_state_paths.get("trainer_state"))
            else:
                saved_paths = save_ctm_model(
                    final_model, args.save_dir, config=cfg,
                )
                summary["saved_models"] = {"ctm_state_dict": saved_paths[0]}
                # Full checkpoint with optimizer + epoch for resume training
                full_ckpt_path = os.path.join(args.save_dir, "ctm_model_full.pt")
                torch.save({
                    "model_state_dict": final_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": test_n_epochs,
                    "config": cfg,
                    "model_params": model_params,
                }, full_ckpt_path)
                logging.info("Full checkpoint saved to %s (resumable)", full_ckpt_path)
            logging.info("Models saved to %s", args.save_dir)
        except Exception as e:
            logging.warning("Model saving failed: %s", e)

        # ── 13b. Save Stage-3 P3 model (if available) ──
        if args.p3 and hasattr(trainer, "get_stage3_model"):
            try:
                stage3_model = trainer.get_stage3_model()
                if stage3_model is not None:
                    stage3_path = os.path.join(args.save_dir, "p3_model.pt")
                    torch.save(stage3_model.state_dict(), stage3_path)
                    if "saved_models" not in summary:
                        summary["saved_models"] = {}
                    summary["saved_models"]["p3_state_dict"] = stage3_path
                    logging.info("Stage-3 P3 model saved to %s", stage3_path)
            except Exception as e:
                logging.warning("Stage-3 P3 model saving failed: %s", e)

    # ── 14. Save results ──
    if args.output != parser.get_default("output"):
        out_path = args.output
    else:
        out_path = cfg.get("output", {}).get("metrics_path", "results/metrics.json")

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Results saved to %s", out_path)
    logging.info("Mean Sharpe: %.4f ± %.4f", summary["mean_sharpe"], summary["std_sharpe"])


if __name__ == "__main__":
    main()
