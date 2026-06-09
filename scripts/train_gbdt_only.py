"""Standalone GBDT-only walk-forward training script.

Baseline for paper comparing CTM vs GBDT vs Ensemble.
No PyTorch neural network (CTM) — pure C++ GBDT (Hoffnung) on aggregated
stock features.  Each CSV in ``--data-dir`` is one asset.

Usage:
    PYTHONPATH=. python scripts/train_gbdt_only.py --data-dir /path/to/csvs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

# ── Make internal imports work regardless of CWD ─────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Configuration loading ────────────────────────────────────────
_KNOWN_TRAINER_KEYS = {
    "n_epochs", "batch_size", "lr", "weight_decay", "grad_clip",
    "patience", "step_size", "train_window", "val_window", "purge_period",
    "warmup_steps", "ramp_steps", "lr_warmup_epochs", "test_n_epochs",
}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "trainer" in cfg:
        unknown = set(cfg["trainer"].keys()) - _KNOWN_TRAINER_KEYS
        if unknown:
            import warnings
            warnings.warn(f"Unknown config keys in trainer: {unknown}")
    return cfg


# ── Metrics helpers ──────────────────────────────────────────────


def _compute_sharpe(preds: np.ndarray, annual_factor: float = 252.0) -> float:
    """Annualised Sharpe ratio from a numpy prediction array."""
    preds = np.asarray(preds, dtype=np.float64).ravel()
    if len(preds) < 2:
        return 0.0
    mean_ret = np.mean(preds)
    std_ret = np.std(preds, ddof=1)
    if std_ret < 1e-12:
        return 0.0
    return float((mean_ret / std_ret) * np.sqrt(annual_factor))


def _safe_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank IC with NaN/inf handling.  Returns 0.0 on failure."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return 0.0
    ic, _ = spearmanr(y_true[valid], y_pred[valid])
    return float(ic) if np.isfinite(ic) else 0.0


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions where sign(pred) == sign(target)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) == 0:
        return 0.0
    signs_match = np.sign(y_true) == np.sign(y_pred)
    zero_both = (y_true == 0.0) & (y_pred == 0.0)
    return float(np.nanmean(np.where(signs_match | zero_both, 1.0, 0.0)))


# ── Multi-asset data loader ──────────────────────────────────────


def load_multi_asset_data(
    data_dir: str,
    seq_len: int,
    target_col: str = "forward_return",
    max_assets: Optional[int] = None,
    periods: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load multiple stock CSV files and align them on common dates.

    Each CSV must have columns: date,open,high,low,close,volume
    (date column used as index).

    Returns
    -------
    data_seq : np.ndarray  shape (B, N, T, D)
    target_seq : np.ndarray  shape (B, T, N)
    asset_names : list[str]
    """
    import glob

    from src.data.features import compute_all_features, compute_forward_returns
    from src.data.dataset import create_sequences

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    if max_assets is not None:
        csv_files = csv_files[:max_assets]

    feature_dfs: List[pd.DataFrame] = []
    asset_names: List[str] = []

    for fpath in csv_files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        if df.empty or "close" not in df.columns:
            logging.warning("Skipping %s: invalid format", name)
            continue
        features = compute_all_features(df)
        close_t = torch.from_numpy(df["close"].values).float()
        features[target_col] = compute_forward_returns(close_t, periods=periods).numpy()  # v4: multi-day
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

    # ── Common date intersection ──
    common_idx = feature_dfs[0].index
    for fdf in feature_dfs[1:]:
        common_idx = common_idx.intersection(fdf.index)

    logging.info(
        "Common date range: %s ~ %s (%d trading days)",
        common_idx.min(), common_idx.max(), len(common_idx),
    )

    feat_cols = [c for c in feature_dfs[0].columns if c != target_col]
    D = len(feat_cols)
    T = len(common_idx)

    feat_arrays: List[np.ndarray] = []
    targ_arrays: List[np.ndarray] = []
    for fdf in feature_dfs:
        aligned = fdf.loc[common_idx]
        feat_arrays.append(aligned[feat_cols].values.astype(np.float32))   # (T, D)
        targ_arrays.append(aligned[target_col].values.astype(np.float32))  # (T,)

    # Stack: (T, N, D) and (T, N)
    feat_3d = np.stack(feat_arrays, axis=1)  # (T, N, D)
    targ_2d = np.stack(targ_arrays, axis=1)  # (T, N)

    # Sequences via flatten → create_sequences → reshape
    flat_feat = feat_3d.reshape(T, N * D)
    seqs_feat = create_sequences(flat_feat, seq_len)                     # (B, T, N*D)
    data_seq = seqs_feat.reshape(-1, seq_len, N, D).transpose(0, 2, 1, 3)  # (B, N, T, D)

    flat_targ = targ_2d.reshape(T, N)
    target_seq = create_sequences(flat_targ, seq_len)                     # (B, T, N)

    logging.info("Multi-asset data: %s, targets: %s", data_seq.shape, target_seq.shape)
    # v4: Cross-sectional demeaning
    target_seq = target_seq - target_seq.mean(axis=-1, keepdims=True)
    return data_seq, target_seq, asset_names


# ── Main pipeline ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GBDT-only walk-forward training baseline (no CTM)"
    )
    parser.add_argument("--data-dir", required=True, help="Path to directory of CSV files")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path (seq_len, split ratios)")
    parser.add_argument("--device", default="cpu", help="Device (always cpu for GBDT)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-periods", type=int, default=5, help="Forward return horizon (v4: 5-day)")
    parser.add_argument("--n-assets", type=int, default=None, help="Max number of assets to use")
    parser.add_argument("--gbdt-trees", type=int, default=200, help="GBDT number of trees")
    parser.add_argument("--gbdt-depth", type=int, default=6, help="GBDT max depth")
    parser.add_argument("--gbdt-loss", type=str, default="mse", help="GBDT loss: mse, mae, huber")
    parser.add_argument("--gbdt-lr", type=float, default=0.1, help="GBDT learning rate")
    parser.add_argument("--output", default="results/gbdt_metrics.json", help="Metrics output path")
    parser.add_argument("--save-dir", default=None, help="Directory to save the final GBDT model JSON")
    # Walk-forward overrides
    parser.add_argument("--train-window", type=int, default=None, help="Override train_window from config")
    parser.add_argument("--val-window", type=int, default=None, help="Override val_window from config")
    parser.add_argument("--step-size", type=int, default=None, help="Override step_size from config")
    parser.add_argument("--purge-period", type=int, default=None, help="Override purge_period from config")
    args = parser.parse_args()

    # ── Logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ── Reproducibility ──
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("Random seed: %d", args.seed)

    # ── Config ──
    cfg = load_config(args.config)
    seq_len = cfg["model"]["seq_len"]
    trainer_cfg = cfg.setdefault("trainer", {})

    train_window = args.train_window if args.train_window is not None else trainer_cfg.get("train_window", 1000)
    val_window = args.val_window if args.val_window is not None else trainer_cfg.get("val_window", 200)
    step_size = args.step_size if args.step_size is not None else trainer_cfg.get("step_size", 21)
    purge_period = args.purge_period if args.purge_period is not None else trainer_cfg.get("purge_period", 126)

    # ── Load data ──
    logging.info("Loading data from %s", args.data_dir)
    data_seq, target_seq, asset_names = load_multi_asset_data(
        args.data_dir, seq_len, max_assets=args.n_assets, periods=args.target_periods,
    )
    B, N, T_seq, D = data_seq.shape
    logging.info("Data loaded: B=%d sequences, N=%d assets, T=%d steps, D=%d features", B, N, T_seq, D)

    # ── Import GBDT (C++) ──
    try:
        from gbdt import GBDT, GBDTConfig
        logging.info("GBDT C++ module loaded (from gbdt package)")
    except ImportError:
        try:
            from gbdt_python import GBDT, GBDTConfig
            logging.info("GBDT C++ module loaded (from gbdt_python)")
        except ImportError:
            logging.error(
                "Cannot import GBDT. Set PYTHONPATH to include Hoffnung build directory.\n"
                "Example: export PYTHONPATH=/path/to/hoffnung/build:/path/to/hoffnung/build/python"
            )
            sys.exit(1)

    # ── Import feature builder ──
    from src.data.gbdt_features import build_gbdt_feature_matrix
    from src.train._walk_forward_utils import walk_forward_windows

    logging.info(
        "Walk-forward: train_window=%d, val_window=%d, step_size=%d, purge_period=%d",
        train_window, val_window, step_size, purge_period,
    )

    # ── Walk-forward training ──
    window_results: List[Dict[str, Any]] = []
    last_model = None

    for pos, train_end, purge_end, val_end in walk_forward_windows(
        B, train_window, val_window, purge_period, step_size,
    ):
        logging.info(
            "Window %d: pos=%d, train=[%d,%d), val=[%d,%d)",
            len(window_results), pos, pos, train_end, purge_end, val_end,
        )

        # 1. Slice temporal windows
        train_data = data_seq[pos:train_end]           # (W_train, N, T, D)
        val_data = data_seq[purge_end:val_end]          # (W_val, N, T, D)

        W_train = train_data.shape[0]
        W_val = val_data.shape[0]

        # 2. Flatten assets into sample axis: (W, N, T, D) → (W*N, T, D)
        train_flat = train_data.reshape(W_train * N, T_seq, D)
        val_flat = val_data.reshape(W_val * N, T_seq, D)

        # 3. Extract targets for last timestep: (W, T, N) → (W, N) → (W*N,)
        train_targ_full = target_seq[pos:train_end]      # (W_train, T, N)
        val_targ_full = target_seq[purge_end:val_end]     # (W_val, T, N)

        y_train = train_targ_full[:, -1, :].ravel()       # (W_train*N,)
        y_val = val_targ_full[:, -1, :].ravel()           # (W_val*N,)

        # 4. Build tabular GBDT features (no CTM hidden states)
        X_train = build_gbdt_feature_matrix(train_flat, include_ctm_features=False)
        X_val = build_gbdt_feature_matrix(val_flat, include_ctm_features=False)

        logging.info(
            "  Features: X_train=%s, X_val=%s (raw=%d agg → %d GBDT features)",
            X_train.shape, X_val.shape, D, X_train.shape[1],
        )

        # 5. Train GBDT via native C++ fit
        config = GBDTConfig()
        config.num_trees = args.gbdt_trees
        config.max_depth = args.gbdt_depth
        config.learning_rate = args.gbdt_lr
        config.loss_type = args.gbdt_loss
        config.subsample_row = 0.8
        config.subsample_col = 0.8
        config.random_seed = args.seed

        X_train_f32 = np.asarray(X_train, dtype=np.float32)
        y_train_f32 = np.asarray(y_train, dtype=np.float32)
        X_val_f32 = np.asarray(X_val, dtype=np.float32)
        y_val_f32 = np.asarray(y_val, dtype=np.float32)

        model = GBDT(config)
        model.fit(X_train_f32, y_train_f32, X_val_f32, y_val_f32)
        last_model = model

        # 6. Predict on validation set
        preds_flat = model.predict(X_val_f32)               # (W_val*N,)

        # 7. Per-asset metrics: reshape → (W_val, N) → compute per column → average
        preds_2d = preds_flat.reshape(W_val, N)             # (W_val, N)
        y_val_2d = y_val_f32.reshape(W_val, N)

        asset_sharpes = []
        asset_ics = []
        asset_dirs = []
        for a in range(N):
            p_a = preds_2d[:, a]
            t_a = y_val_2d[:, a]
            strategy_ret = np.sign(p_a) * t_a
            asset_sharpes.append(_compute_sharpe(strategy_ret))
            asset_ics.append(_safe_ic(t_a, p_a))
            asset_dirs.append(_directional_accuracy(t_a, p_a))

        window_sharpe = float(np.nanmean(asset_sharpes))
        window_ic = float(np.nanmean(asset_ics))
        window_dir = float(np.nanmean(asset_dirs))

        logging.info(
            "  Metrics: sharpe=%.4f, IC=%.4f, dir_acc=%.4f",
            window_sharpe, window_ic, window_dir,
        )

        window_results.append({
            "window": len(window_results),
            "window_start": int(pos),
            "window_end": int(val_end),
            "sharpe": window_sharpe,
            "ic": window_ic,
            "directional_accuracy": window_dir,
        })

    # ── Handle zero-windows edge case ──
    if not window_results:
        logging.warning("No walk-forward windows fit. Reduce train_window/val_window.")
        summary: Dict[str, Any] = {
            "n_windows": 0,
            "error": "No valid walk-forward windows. Reduce train_window/val_window.",
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        logging.warning("Empty results summary saved to %s", args.output)
        return

    # ── Aggregate summary ──
    sharpes = [r["sharpe"] for r in window_results]
    ics = [r["ic"] for r in window_results]
    dirs = [r["directional_accuracy"] for r in window_results]

    summary = {
        "n_windows": len(window_results),
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "mean_ic": float(np.mean(ics)),
        "std_ic": float(np.std(ics)),
        "mean_directional_accuracy": float(np.mean(dirs)),
        "std_directional_accuracy": float(np.std(dirs)),
        "windows": window_results,
    }

    # ── Save final GBDT model as JSON ──
    if args.save_dir and last_model is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        json_str = last_model.to_json()
        model_path = os.path.join(args.save_dir, "gbdt_model.json")
        with open(model_path, "w") as f:
            f.write(json_str)
        logging.info("GBDT model saved to %s", model_path)

    # ── Save metrics ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Metrics saved to %s", args.output)
    logging.info(
        "Summary: sharpe=%.4f ± %.4f, IC=%.4f ± %.4f, dir_acc=%.4f ± %.4f",
        summary["mean_sharpe"], summary["std_sharpe"],
        summary["mean_ic"], summary["std_ic"],
        summary["mean_directional_accuracy"], summary["std_directional_accuracy"],
    )


if __name__ == "__main__":
    main()
