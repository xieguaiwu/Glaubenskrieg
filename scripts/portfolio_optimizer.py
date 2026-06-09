#!/usr/bin/env python3
"""
Portfolio-Level Sharpe Optimization Experiment
==============================================
Compares per-stock MSE training against portfolio-aware training to test
whether optimizing Sharpe directly extracts more signal from noisy OHLCV
features.

Four approaches:
  1. LGB-MSE (BASELINE)     — standard regression, top-K portfolio
  2. LGB-LambdaRank          — listwise ranking loss, top-K portfolio
  3. LGB-MSE + Sharpe-tuned  — MSE training, hyperparams selected by Sharpe
  4. NN-Sharpe               — 2-layer MLP trained with differentiable Sharpe loss

Key research question:
  Does portfolio-aware training extract more signal from the same noisy
  features than per-stock training? Or does it just overfit differently?

Output: results/portfolio_optimizer.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Project imports ──────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.data.features import compute_all_features

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("portfolio_opt")

# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════

DEFAULT_DATA_DIR = "./new_data/data/us_stocks"
DEFAULT_OUTPUT = "./results/portfolio_optimizer.json"
N_STOCKS = 100               # Top N most liquid stocks
FORWARD_PERIODS = 5          # 5-day forward return
SEQ_LEN = 63                 # Feature window (used for context, not directly)
TOP_K_FRAC = 0.20            # Long top 20%, short bottom 20%
WALK_FORWARD = {
    "train_years": 4,        # 1008 days
    "val_years": 1,          # 252 days
    "purge_months": 6,       # 126 days
    "step_months": 6,        # 126 days
}

# LightGBM base params
LGB_BASE = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

# LightGBM hyperparameter grid for Sharpe-tuned approach
LGB_HP_GRID = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.02, "num_leaves": 15},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03, "num_leaves": 31},
    {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05, "num_leaves": 31},
    {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.01, "num_leaves": 15},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "num_leaves": 63},
]

# Neural network config
NN_CONFIG = {
    "hidden_dim": 32,
    "n_epochs": 50,
    "batch_days": 32,        # Days per mini-batch
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "temperature": 1.0,      # Softmax temperature for smooth portfolio weights
    "patience": 10,
}


# ══════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════

def sharpe_ratio(returns: np.ndarray, annualize: bool = True) -> float:
    """Annualized Sharpe ratio."""
    returns = np.asarray(returns, dtype=np.float64).ravel()
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-12:
        return 0.0
    sr = mu / sigma
    return float(sr * np.sqrt(252) if annualize else sr)


def sortino_ratio(returns: np.ndarray) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    returns = np.asarray(returns, dtype=np.float64).ravel()
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    downside = returns[returns < 0]
    if len(downside) < 2:
        return float(np.inf) if mu > 0 else 0.0
    sigma_d = np.std(downside, ddof=1)
    if sigma_d < 1e-12:
        return 0.0
    return float((mu / sigma_d) * np.sqrt(252))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown as negative fraction."""
    peak = np.maximum.accumulate(equity)
    return float(np.min((equity - peak) / np.maximum(peak, 1e-12)))


def ic_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cross-sectional Spearman rank IC."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 10:
        return 0.0
    ic, _ = spearmanr(y_true[valid], y_pred[valid])
    return float(ic) if np.isfinite(ic) else 0.0


def long_short_sharpe(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    top_frac: float = TOP_K_FRAC,
) -> Dict[str, float]:
    """
    Form a long-short portfolio from cross-sectional predictions.
    Long top fraction, short bottom fraction. Returns Sharpe and other metrics.
    
    Parameters
    ----------
    predictions : (T, N) array — predicted returns for each stock at each time
    actual_returns : (T, N) array — realized forward returns
    top_frac : fraction to go long/short
    
    Returns
    -------
    dict with sharpe, sortino, max_dd, mean_ret, vol, turnover
    """
    T, N = predictions.shape
    if T < 20 or N < 5:
        return {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0,
                "mean_ret": 0.0, "vol": 0.0, "turnover": 0.0}
    
    K = max(1, int(N * top_frac))
    portfolio_returns = np.zeros(T)
    prev_long = None
    
    for t in range(T):
        pred_t = predictions[t]
        actual_t = actual_returns[t]
        
        # Skip days with too many NaNs
        valid = np.isfinite(pred_t) & np.isfinite(actual_t)
        if valid.sum() < 2 * K:
            portfolio_returns[t] = 0.0
            continue
        
        # Rank predictions (higher → better)
        order = np.argsort(pred_t[valid])
        valid_indices = np.where(valid)[0]
        
        short_idx = valid_indices[order[:K]]
        long_idx = valid_indices[order[-K:]]
        
        short_ret = np.nanmean(actual_t[short_idx])
        long_ret = np.nanmean(actual_t[long_idx])
        portfolio_returns[t] = long_ret - short_ret
    
    # Remove leading zeros
    portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]
    if len(portfolio_returns) < 10:
        return {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0,
                "mean_ret": 0.0, "vol": 0.0, "turnover": 0.0}
    
    # Equity curve for max DD
    equity = np.cumprod(1 + portfolio_returns)
    
    return {
        "sharpe": sharpe_ratio(portfolio_returns),
        "sortino": sortino_ratio(portfolio_returns),
        "max_dd": max_drawdown(equity),
        "mean_ret": float(np.mean(portfolio_returns)),
        "vol": float(np.std(portfolio_returns, ddof=1)),
        "turnover": 0.0,  # Computed separately if needed
    }


# ══════════════════════════════════════════════════════════════════
# Data Loading & Feature Engineering
# ══════════════════════════════════════════════════════════════════

def load_stock_data(
    data_dir: str,
    n_top: int = N_STOCKS,
    min_history: int = 2000,
) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[str], pd.DatetimeIndex
]:
    """
    Load all CSVs, filter to stocks with sufficient history,
    then select top-N by average volume (liquidity).
    Compute features, align dates.
    
    Returns
    -------
    stocks_raw : dict symbol → OHLCV DataFrame
    features : dict symbol → feature DataFrame
    symbols : list of symbol strings (sorted)
    common_dates : shared DatetimeIndex
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        # Try alternate path
        alt = Path("../new_data/data/us_stocks")
        if alt.exists():
            data_path = alt
        else:
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    csv_files = sorted(data_path.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {data_path}")
    
    # Load and filter by history length
    stocks_raw: Dict[str, pd.DataFrame] = {}
    for fp in csv_files:
        sym = fp.stem
        try:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if len(df) < min_history:
                continue
            if "volume" not in df.columns:
                continue
            stocks_raw[sym] = df
        except Exception as e:
            logger.warning(f"Failed to load {fp.name}: {e}")
    
    logger.info(f"Loaded {len(stocks_raw)} stocks with ≥{min_history} rows")
    
    # Compute average volume for each stock (last year), select top N
    volumes: List[Tuple[str, float]] = []
    for sym, df in stocks_raw.items():
        avg_vol = df["volume"].iloc[-252:].mean()
        volumes.append((sym, avg_vol))
    
    volumes.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [s for s, _ in volumes[:n_top]]
    stocks_raw = {s: stocks_raw[s] for s in top_symbols}
    logger.info(f"Selected top {len(stocks_raw)} stocks by liquidity (from {len(volumes)} eligible)")
    
    # Compute features for each stock
    features: Dict[str, pd.DataFrame] = {}
    for i, (sym, df) in enumerate(stocks_raw.items()):
        try:
            feats = compute_all_features(df)
            features[sym] = feats
        except Exception as e:
            logger.warning(f"Feature computation failed for {sym}: {e}")
            continue
        if (i + 1) % 50 == 0:
            logger.info(f"  Features computed: {i+1}/{len(stocks_raw)}")
    
    # Align to common date range
    common_dates = None
    for f in features.values():
        if common_dates is None:
            common_dates = f.index
        else:
            common_dates = common_dates.intersection(f.index)
    
    # Keep only stocks with full feature coverage
    symbols = sorted(features.keys())
    features = {s: features[s].loc[common_dates] for s in symbols}
    stocks_raw = {s: stocks_raw[s].loc[common_dates] for s in symbols}
    
    logger.info(
        f"Aligned: {len(common_dates)} days × {len(symbols)} stocks, "
        f"{common_dates[0].date()} ~ {common_dates[-1].date()}"
    )
    
    return stocks_raw, features, symbols, common_dates


def build_panel(
    features: Dict[str, pd.DataFrame],
    stocks_raw: Dict[str, pd.DataFrame],
    symbols: List[str],
    common_dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build aligned panel arrays.
    
    Returns
    -------
    X : (T, N, F) feature array
    y_ret : (T, N) forward return array
    y_vol : (T, N) realized volatility array
    """
    T = len(common_dates)
    N = len(symbols)
    
    feature_cols = list(features[symbols[0]].columns)
    F = len(feature_cols)
    
    X = np.zeros((T, N, F), dtype=np.float32)
    for j, sym in enumerate(symbols):
        X[:, j, :] = features[sym].values.astype(np.float32)
    
    # Forward returns: close[t+FORWARD] / close[t] - 1
    y_ret = np.zeros((T, N), dtype=np.float32)
    for j, sym in enumerate(symbols):
        prices = stocks_raw[sym]["close"].values
        n = len(prices)
        valid_len = n - FORWARD_PERIODS
        if valid_len > 0:
            y_ret[:valid_len, j] = (
                prices[FORWARD_PERIODS:] / prices[:valid_len] - 1.0
            ).astype(np.float32)
        # Last FORWARD_PERIODS days are NaN (no future data)
        y_ret[valid_len:, j] = np.nan
    
    return X, y_ret, feature_cols


# ══════════════════════════════════════════════════════════════════
# Walk-Forward Window Generation
# ══════════════════════════════════════════════════════════════════

@dataclass
class Window:
    """A single walk-forward window."""
    idx: int
    train_start: int
    train_end: int      # exclusive
    purge_end: int      # exclusive (train_end + purge)
    val_start: int
    val_end: int        # exclusive
    

def generate_windows(
    n_days: int,
    train_days: int = 1008,
    purge_days: int = 126,
    val_days: int = 252,
    step_days: int = 126,
) -> List[Window]:
    """Generate walk-forward windows."""
    windows = []
    idx = 0
    pos = 0
    while pos + train_days + purge_days + val_days <= n_days:
        train_end = pos + train_days
        purge_end = train_end + purge_days
        val_end = purge_end + val_days
        windows.append(Window(idx, pos, train_end, purge_end, purge_end, val_end))
        pos += step_days
        idx += 1
    return windows


def window_to_indices(window: Window, n_stocks: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert day-level window to flat sample indices for tabular training.
    Flat indexing: sample i = day d, stock s → index = d * n_stocks + s
    """
    train_idx = np.arange(
        window.train_start * n_stocks,
        window.train_end * n_stocks,
    )
    val_idx = np.arange(
        window.val_start * n_stocks,
        window.val_end * n_stocks,
    )
    return train_idx, val_idx


# ══════════════════════════════════════════════════════════════════
# Approach 1: LGB-MSE Baseline
# ══════════════════════════════════════════════════════════════════

def train_lgb_mse(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = None,
) -> lgb.LGBMRegressor:
    """Train LightGBM regressor with MSE loss."""
    p = dict(LGB_BASE)
    if params:
        p.update(params)
    
    mask_train = ~np.isnan(y_train)
    mask_val = ~np.isnan(y_val)
    
    if mask_train.sum() < 200 or mask_val.sum() < 50:
        raise ValueError(f"Insufficient data: {mask_train.sum()} train, {mask_val.sum()} val")
    
    model = lgb.LGBMRegressor(**p)
    model.fit(
        X_train[mask_train], y_train[mask_train],
        eval_set=[(X_val[mask_val], y_val[mask_val])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def evaluate_window_ls(
    predictions_2d: np.ndarray,   # (T_val, N)
    actual_returns_2d: np.ndarray,  # (T_val, N)
    name: str = "",
) -> Dict:
    """Evaluate a window's long-short portfolio metrics."""
    metrics = long_short_sharpe(predictions_2d, actual_returns_2d)
    # Also compute cross-sectional IC
    ics = []
    T, N = predictions_2d.shape
    for t in range(T):
        valid = np.isfinite(predictions_2d[t]) & np.isfinite(actual_returns_2d[t])
        if valid.sum() >= 10:
            ics.append(ic_spearman(actual_returns_2d[t], predictions_2d[t]))
    
    metrics["ic_mean"] = float(np.mean(ics)) if ics else 0.0
    metrics["ic_std"] = float(np.std(ics)) if ics else 0.0
    metrics["name"] = name
    return metrics


# ══════════════════════════════════════════════════════════════════
# Approach 2: LGB-LambdaRank
# ══════════════════════════════════════════════════════════════════

def _discretize_labels(
    y: np.ndarray,
    groups: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """
    Convert continuous labels to integer relevance labels (0 to n_bins-1)
    by quantile binning within each group (day).
    Higher return → higher label.
    """
    y_disc = np.full_like(y, -1, dtype=int)
    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = groups == g
        y_g = y[mask]
        valid = ~np.isnan(y_g)
        if valid.sum() < n_bins:
            continue
        y_valid = y_g[valid]
        # Bin into n_bins quantiles
        bin_edges = np.percentile(y_valid, np.linspace(0, 100, n_bins + 1))
        # Make edges unique for digitize
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            y_disc[mask][valid] = 0
            continue
        # digitize returns 1..len(bins)-1, subtract 1 for 0-indexed
        assigned = np.digitize(y_valid, bin_edges[:-1]) - 1
        assigned = np.clip(assigned, 0, n_bins - 1)
        # Assign back to full array
        disc_indices = np.where(mask)[0][valid]
        y_disc[disc_indices] = assigned
    return y_disc


def train_lgb_lambdarank(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_groups: np.ndarray,
    params: dict = None,
) -> lgb.LGBMRanker:
    """
    Train LightGBM ranker with LambdaRank objective.
    Groups indicate which samples belong to the same query (day).
    Continuous labels are discretized into 5 quantile bins per day.
    """
    p = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10, 20],
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if params:
        p.update(params)
    
    # Discretize labels
    y_train_disc = _discretize_labels(y_train, train_groups)
    y_val_disc = _discretize_labels(y_val, val_groups)
    
    # Remove NaN / unlabeled entries
    mask_train = (y_train_disc >= 0) & ~np.isnan(X_train).any(axis=1)
    mask_val = (y_val_disc >= 0) & ~np.isnan(X_val).any(axis=1)
    
    if mask_train.sum() < 200 or mask_val.sum() < 50:
        raise ValueError(f"Insufficient data for LambdaRank: {mask_train.sum()} train, {mask_val.sum()} val")
    
    X_tr = X_train[mask_train]
    y_tr = y_train_disc[mask_train]
    groups_tr = train_groups[mask_train]
    _, group_sizes = np.unique(groups_tr, return_counts=True)
    
    X_vl = X_val[mask_val]
    y_vl = y_val_disc[mask_val]
    groups_vl = val_groups[mask_val]
    _, group_sizes_v = np.unique(groups_vl, return_counts=True)
    
    model = lgb.LGBMRanker(**p)
    model.fit(
        X_tr, y_tr,
        group=group_sizes,
        eval_set=[(X_vl, y_vl)],
        eval_group=[group_sizes_v],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


# ══════════════════════════════════════════════════════════════════
# Approach 3: LGB + Sharpe HP Tuning
# ══════════════════════════════════════════════════════════════════

def train_lgb_sharpe_tuned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_stocks: int,
    hp_grid: List[dict] = None,
) -> Tuple[lgb.LGBMRegressor, dict]:
    """
    Train LGB with MSE, then select best hyperparameters by
    long-short portfolio Sharpe on validation set.
    """
    if hp_grid is None:
        hp_grid = LGB_HP_GRID
    
    mask_train = ~np.isnan(y_train)
    mask_val = ~np.isnan(y_val)
    
    best_model = None
    best_score = -np.inf
    best_hp = {}
    results = []
    
    T_val = len(np.unique(np.arange(len(y_val))[mask_val] // n_stocks))
    
    for hp in hp_grid:
        try:
            p = dict(LGB_BASE)
            p.update(hp)
            
            model = lgb.LGBMRegressor(**p)
            model.fit(
                X_train[mask_train], y_train[mask_train],
                eval_set=[(X_val[mask_val], y_val[mask_val])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            
            # Predict on validation set
            y_pred_flat = model.predict(X_val[mask_val])
            
            # Reshape to (T_val, N) for portfolio evaluation
            # Need to reconstruct the 2D shape
            val_indices = np.where(mask_val)[0]
            day_indices = val_indices // n_stocks
            stock_indices = val_indices % n_stocks
            
            unique_days = np.unique(day_indices)
            pred_2d = np.full((len(unique_days), n_stocks), np.nan)
            actual_2d = np.full((len(unique_days), n_stocks), np.nan)
            
            for k, d in enumerate(unique_days):
                mask_d = day_indices == d
                pred_2d[k, stock_indices[mask_d]] = y_pred_flat[mask_d]
                actual_2d[k, stock_indices[mask_d]] = y_val[val_indices[mask_d]]
            
            ls_metrics = long_short_sharpe(pred_2d, actual_2d)
            score = ls_metrics["sharpe"]
            
            results.append({
                "hp": hp,
                "sharpe": score,
                "sortino": ls_metrics["sortino"],
                "max_dd": ls_metrics["max_dd"],
            })
            
            if score > best_score:
                best_score = score
                best_model = model
                best_hp = hp
                
        except Exception as e:
            logger.warning(f"HP combo failed: {hp}, error: {e}")
            continue
    
    if best_model is None:
        # Fallback to default
        best_model = train_lgb_mse(X_train, y_train, X_val, y_val)
        best_hp = dict(LGB_BASE)
    
    return best_model, {"best_hp": best_hp, "best_sharpe": best_score, "all_results": results}


# ══════════════════════════════════════════════════════════════════
# Approach 4: Neural Network with Differentiable Sharpe Loss
# ══════════════════════════════════════════════════════════════════

class PortfolioMLP(nn.Module):
    """2-layer MLP for cross-sectional return prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, F) → (N, 1) predictions"""
        return self.net(x).squeeze(-1)


def smooth_portfolio_weights(
    predictions: torch.Tensor,  # (N,)
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Convert predictions to differentiable long-short portfolio weights.
    
    w_i = softmax(pred/tau)_i - 1/N
    
    This is differentiable and centered (sums to 0). Higher predictions
    get positive weights (long), lower get negative (short).
    """
    N = predictions.shape[0]
    if temperature <= 0:
        temperature = 1.0
    scaled = predictions / temperature
    # Numerical stability: subtract max
    scaled = scaled - scaled.max()
    soft_weights = F.softmax(scaled, dim=0)
    # Center: subtract 1/N so weights sum to 0
    weights = soft_weights - 1.0 / N
    return weights


def differentiable_sharpe_loss(
    predictions: torch.Tensor,    # (T, N)
    actual_returns: torch.Tensor, # (T, N)
    temperature: float = 1.0,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute negative Sharpe ratio of a smooth long-short portfolio.
    
    For each day, compute portfolio return = sum(w_i * actual_return_i)
    where w_i = softmax(pred_i/τ) - 1/N (differentiable weights).
    
    Returns
    -------
    loss : scalar, -Sharpe
    sharpe_val : scalar, actual Sharpe (for logging)
    """
    T, N = predictions.shape
    
    # Remove days with NaN returns
    valid_mask = ~torch.isnan(actual_returns).any(dim=1)
    if valid_mask.sum() < 10:
        return torch.tensor(0.0, device=predictions.device), torch.tensor(0.0)
    
    pred_valid = predictions[valid_mask]
    ret_valid = actual_returns[valid_mask]
    T_valid = pred_valid.shape[0]
    
    portfolio_rets = torch.zeros(T_valid, device=predictions.device)
    
    for t in range(T_valid):
        w = smooth_portfolio_weights(pred_valid[t], temperature)
        portfolio_rets[t] = torch.dot(w, ret_valid[t])
    
    mu = portfolio_rets.mean()
    sigma = portfolio_rets.std() + epsilon
    
    sharpe_val = mu / sigma
    # Annualize for interpretable numbers (not for gradient)
    loss = -sharpe_val  # Minimize negative Sharpe = maximize Sharpe
    
    return loss, sharpe_val.detach()


def train_portfolio_nn(
    X_train: np.ndarray,         # (T_train, N, F)
    y_train: np.ndarray,         # (T_train, N)
    X_val: np.ndarray,           # (T_val, N, F)
    y_val: np.ndarray,           # (T_val, N)
    n_epochs: int = 50,
    batch_days: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    temperature: float = 1.0,
    patience: int = 10,
    seed: int = 42,
) -> Tuple[nn.Module, dict]:
    """
    Train a 2-layer MLP with differentiable Sharpe loss.
    
    Each mini-batch contains a subset of days; for each day,
    we predict all N stocks cross-sectionally and compute
    the smooth portfolio Sharpe.
    """
    torch.manual_seed(seed)
    
    T_train, N, F = X_train.shape
    T_val = X_val.shape[0]
    
    device = torch.device("cpu")
    model = PortfolioMLP(input_dim=F, hidden_dim=NN_CONFIG["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Impute NaN in features using column means from training set
    X_train_imp = X_train.copy()
    X_val_imp = X_val.copy()
    for f in range(F):
        col = X_train_imp[:, :, f]
        col_mean = np.nanmean(col)
        if np.isnan(col_mean):
            col_mean = 0.0
        X_train_imp[:, :, f] = np.nan_to_num(col, nan=col_mean)
        X_val_imp[:, :, f] = np.nan_to_num(X_val_imp[:, :, f], nan=col_mean)
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train_imp).float()  # (T, N, F)
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val_imp).float()
    y_val_t = torch.from_numpy(y_val).float()
    
    # Remove NaN days from training (NaN in target returns)
    train_valid = ~torch.isnan(y_train_t).any(dim=1)
    val_valid = ~torch.isnan(y_val_t).any(dim=1)
    
    X_train_clean = X_train_t[train_valid]
    y_train_clean = y_train_t[train_valid]
    X_val_clean = X_val_t[val_valid]
    y_val_clean = y_val_t[val_valid]
    
    T_clean = X_train_clean.shape[0]
    if T_clean < batch_days:
        batch_days = max(1, T_clean // 2)
    
    history = {"train_loss": [], "val_sharpe": [], "val_loss": []}
    best_val_sharpe = -np.inf
    best_state = None
    no_improve = 0
    
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        
        # Shuffle day indices
        day_order = torch.randperm(T_clean)
        
        for start in range(0, T_clean, batch_days):
            end = min(start + batch_days, T_clean)
            batch_indices = day_order[start:end]
            
            X_batch = X_train_clean[batch_indices]  # (B, N, F)
            y_batch = y_train_clean[batch_indices]  # (B, N)
            
            # Forward: predict all stocks for all batch days
            B = X_batch.shape[0]
            X_flat = X_batch.reshape(-1, F)         # (B*N, F)
            pred_flat = model(X_flat)                # (B*N,)
            pred_batch = pred_flat.reshape(B, N)     # (B, N)
            
            loss, _ = differentiable_sharpe_loss(pred_batch, y_batch, temperature)
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            Xv_flat = X_val_clean.reshape(-1, F)
            predv_flat = model(Xv_flat)
            predv = predv_flat.reshape(X_val_clean.shape[0], N)
            val_loss, val_sr = differentiable_sharpe_loss(
                predv, y_val_clean, temperature
            )
        
        avg_train_loss = float(np.mean(epoch_losses))
        val_sr_val = float(val_sr) * np.sqrt(252)  # Annualize for logging
        
        history["train_loss"].append(avg_train_loss)
        history["val_sharpe"].append(val_sr_val)
        history["val_loss"].append(float(val_loss))
        
        if (epoch + 1) % 10 == 0:
            logger.debug(
                f"  NN epoch {epoch+1:3d}/{n_epochs} | "
                f"train_loss={avg_train_loss:.4f} | val_sharpe={val_sr_val:.3f}"
            )
        
        # Early stopping
        if val_sr_val > best_val_sharpe:
            best_val_sharpe = val_sr_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.debug(f"  NN early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, {
        "best_val_sharpe": best_val_sharpe,
        "n_epochs_trained": epoch + 1 - no_improve if no_improve < patience else epoch + 1,
        "history": history,
    }


# ══════════════════════════════════════════════════════════════════
# Cross-sectional prediction helper
# ══════════════════════════════════════════════════════════════════

def predictions_to_2d(
    pred_flat: np.ndarray,
    mask: np.ndarray,
    n_stocks: int,
    n_days: int,
) -> np.ndarray:
    """Convert flat predictions back to (T, N) with NaN padding."""
    result = np.full((n_days, n_stocks), np.nan, dtype=np.float32)
    indices = np.where(mask)[0]
    days = indices // n_stocks
    stocks = indices % n_stocks
    result[days, stocks] = pred_flat
    return result


# ══════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════

def run_experiment(
    data_dir: str = DEFAULT_DATA_DIR,
    n_stocks: int = N_STOCKS,
    output_path: str = DEFAULT_OUTPUT,
    skip_nn: bool = False,
    seed: int = 42,
) -> Dict:
    """Run the full portfolio optimizer comparison experiment."""
    
    t_start = time.time()
    results = {
        "config": {
            "data_dir": data_dir,
            "n_stocks": n_stocks,
            "forward_periods": FORWARD_PERIODS,
            "top_k_frac": TOP_K_FRAC,
            "walk_forward": WALK_FORWARD,
            "seed": seed,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # ═══════════ 1. Load Data ═══════════
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preparing data")
    logger.info("=" * 60)
    
    stocks_raw, features, symbols, common_dates = load_stock_data(data_dir, n_stocks)
    n_stocks_actual = len(symbols)
    n_days = len(common_dates)
    
    X, y_ret, feature_cols = build_panel(features, stocks_raw, symbols, common_dates)
    F = X.shape[2]
    
    results["data"] = {
        "n_stocks": n_stocks_actual,
        "n_days": n_days,
        "n_features": F,
        "feature_cols": feature_cols,
        "date_range": [str(common_dates[0].date()), str(common_dates[-1].date())],
        "symbols": symbols[:10] + ["..."] + symbols[-5:] if len(symbols) > 15 else symbols,
    }
    logger.info(f"Panel: {n_days}d × {n_stocks_actual}s × {F}f")
    
    # ═══════════ 2. Walk-Forward Windows ═══════════
    logger.info("=" * 60)
    logger.info("STEP 2: Generating walk-forward windows")
    logger.info("=" * 60)
    
    train_days = WALK_FORWARD["train_years"] * 252
    purge_days = WALK_FORWARD["purge_months"] * 21
    val_days = WALK_FORWARD["val_years"] * 252
    step_days = WALK_FORWARD["step_months"] * 21
    
    windows = generate_windows(n_days, train_days, purge_days, val_days, step_days)
    logger.info(f"Generated {len(windows)} walk-forward windows")
    
    results["windows"] = {
        "n_windows": len(windows),
        "train_days": train_days,
        "purge_days": purge_days,
        "val_days": val_days,
        "step_days": step_days,
    }
    
    # ═══════════ 3. Run Approaches ═══════════
    # Map flat index to day index (for LambdaRank groups)
    all_day_indices = np.repeat(np.arange(n_days), n_stocks_actual)
    
    # Storage for window-level results
    all_windows = []
    
    for wi, window in enumerate(windows):
        logger.info(f"\n{'='*60}")
        logger.info(f"WINDOW {wi+1}/{len(windows)}: "
                    f"train [{window.train_start}:{window.train_end}) "
                    f"val [{window.val_start}:{window.val_end})")
        logger.info(f"{'='*60}")
        
        train_idx, val_idx = window_to_indices(window, n_stocks_actual)
        
        # Flatten features and targets
        X_flat = X.reshape(-1, F)
        y_ret_flat = y_ret.reshape(-1)
        
        X_tr = X_flat[train_idx]
        y_tr = y_ret_flat[train_idx]
        X_vl = X_flat[val_idx]
        y_vl = y_ret_flat[val_idx]
        
        # Day groups for LambdaRank
        groups_tr = all_day_indices[train_idx]
        groups_vl = all_day_indices[val_idx]
        
        # Validation data in 2D for portfolio evaluation
        T_val = window.val_end - window.val_start
        X_val_2d = X[window.val_start:window.val_end]  # (T_val, N, F)
        y_val_2d = y_ret[window.val_start:window.val_end]  # (T_val, N)
        
        window_result = {
            "window_id": wi,
            "train_range": [int(window.train_start), int(window.train_end)],
            "val_range": [int(window.val_start), int(window.val_end)],
            "n_train_samples": len(train_idx),
            "n_val_samples": len(val_idx),
        }
        
        # --- Approach 1: Baseline LGB-MSE ---
        logger.info("  [1/4] LGB-MSE Baseline...")
        t0 = time.time()
        try:
            model_mse = train_lgb_mse(X_tr, y_tr, X_vl, y_vl)
            # Predict on validation set (flat)
            mask_val = ~np.isnan(y_vl)
            pred_flat_mse = model_mse.predict(X_vl[mask_val])
            pred_2d_mse = predictions_to_2d(pred_flat_mse, mask_val,
                                             n_stocks_actual, T_val)
            mse_metrics = evaluate_window_ls(pred_2d_mse, y_val_2d, "LGB-MSE")
            mse_metrics["runtime_s"] = time.time() - t0
            window_result["lgb_mse"] = mse_metrics
            logger.info(f"    Sharpe={mse_metrics['sharpe']:.3f}, "
                       f"IC={mse_metrics['ic_mean']:.4f}, "
                       f"Sortino={mse_metrics['sortino']:.3f}")
        except Exception as e:
            logger.error(f"  LGB-MSE failed: {e}")
            window_result["lgb_mse"] = {"error": str(e)}
        
        # --- Approach 2: LGB-LambdaRank ---
        logger.info("  [2/4] LGB-LambdaRank...")
        t0 = time.time()
        try:
            model_lambdarank = train_lgb_lambdarank(
                X_tr, y_tr, groups_tr, X_vl, y_vl, groups_vl
            )
            mask_val = ~np.isnan(y_vl)
            pred_flat_lr = model_lambdarank.predict(X_vl[mask_val])
            pred_2d_lr = predictions_to_2d(pred_flat_lr, mask_val,
                                            n_stocks_actual, T_val)
            lr_metrics = evaluate_window_ls(pred_2d_lr, y_val_2d, "LGB-LambdaRank")
            lr_metrics["runtime_s"] = time.time() - t0
            window_result["lgb_lambdarank"] = lr_metrics
            logger.info(f"    Sharpe={lr_metrics['sharpe']:.3f}, "
                       f"IC={lr_metrics['ic_mean']:.4f}")
        except Exception as e:
            logger.error(f"  LGB-LambdaRank failed: {e}")
            window_result["lgb_lambdarank"] = {"error": str(e)}
        
        # --- Approach 3: LGB + Sharpe HP Tuning ---
        logger.info("  [3/4] LGB + Sharpe HP Tuning...")
        t0 = time.time()
        try:
            model_tuned, tuning_info = train_lgb_sharpe_tuned(
                X_tr, y_tr, X_vl, y_vl, n_stocks_actual
            )
            mask_val = ~np.isnan(y_vl)
            pred_flat_tuned = model_tuned.predict(X_vl[mask_val])
            pred_2d_tuned = predictions_to_2d(pred_flat_tuned, mask_val,
                                               n_stocks_actual, T_val)
            tuned_metrics = evaluate_window_ls(pred_2d_tuned, y_val_2d, "LGB-SharpeTuned")
            tuned_metrics["runtime_s"] = time.time() - t0
            tuned_metrics["tuning"] = tuning_info
            window_result["lgb_sharpe_tuned"] = tuned_metrics
            logger.info(f"    Sharpe={tuned_metrics['sharpe']:.3f}, "
                       f"best_hp={tuning_info.get('best_hp', {})}")
        except Exception as e:
            logger.error(f"  LGB-SharpeTuned failed: {e}")
            window_result["lgb_sharpe_tuned"] = {"error": str(e)}
        
        # --- Approach 4: NN with Sharpe Loss ---
        if not skip_nn:
            logger.info("  [4/4] NN-Sharpe...")
            t0 = time.time()
            try:
                # Prepare 2D training data for NN
                X_train_2d = X[window.train_start:window.train_end]  # (T_train, N, F)
                y_train_2d = y_ret[window.train_start:window.train_end]  # (T_train, N)
                
                model_nn, nn_info = train_portfolio_nn(
                    X_train_2d, y_train_2d,
                    X_val_2d, y_val_2d,
                    n_epochs=NN_CONFIG["n_epochs"],
                    batch_days=NN_CONFIG["batch_days"],
                    lr=NN_CONFIG["lr"],
                    weight_decay=NN_CONFIG["weight_decay"],
                    temperature=NN_CONFIG["temperature"],
                    patience=NN_CONFIG["patience"],
                    seed=seed,
                )
                
                # Predict on validation set (handle NaN in features)
                model_nn.eval()
                X_val_nonan = np.nan_to_num(X_val_2d, nan=0.0)
                with torch.no_grad():
                    Xv_t = torch.from_numpy(X_val_nonan).float()
                    Tv, Nv, Fv = Xv_t.shape
                    pred_flat_nn = model_nn(Xv_t.reshape(-1, Fv)).numpy()
                    pred_2d_nn = pred_flat_nn.reshape(Tv, Nv)
                
                nn_metrics = evaluate_window_ls(pred_2d_nn, y_val_2d, "NN-Sharpe")
                nn_metrics["runtime_s"] = time.time() - t0
                nn_metrics["training"] = nn_info
                window_result["nn_sharpe"] = nn_metrics
                logger.info(f"    Sharpe={nn_metrics['sharpe']:.3f}, "
                           f"IC={nn_metrics['ic_mean']:.4f}, "
                           f"epochs={nn_info['n_epochs_trained']}")
            except Exception as e:
                logger.error(f"  NN-Sharpe failed: {e}")
                import traceback
                traceback.print_exc()
                window_result["nn_sharpe"] = {"error": str(e)}
        
        # --- Baselines: Random & Mean-Reversion ---
        T_val = window.val_end - window.val_start
        y_actual_2d = y_ret[window.val_start:window.val_end]
        
        # Random predictions
        rng = np.random.RandomState(seed + wi)
        random_pred_2d = rng.randn(T_val, n_stocks_actual)
        random_metrics = evaluate_window_ls(random_pred_2d, y_actual_2d, "Random")
        window_result["random"] = random_metrics
        
        # Mean-reversion: predict opposite of yesterday's return
        ret_feat_idx = feature_cols.index("simple_return") if "simple_return" in feature_cols else 0
        meanrev_pred_2d = -X[window.val_start:window.val_end, :, ret_feat_idx]
        meanrev_metrics = evaluate_window_ls(meanrev_pred_2d, y_actual_2d, "MeanRev")
        window_result["meanrev"] = meanrev_metrics
        
        all_windows.append(window_result)
    
    results["window_results"] = all_windows
    
    # ═══════════ 4. Aggregate & Compare ═══════════
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)
    
    approaches = ["random", "meanrev", "lgb_mse", "lgb_lambdarank", "lgb_sharpe_tuned"]
    if not skip_nn:
        approaches.append("nn_sharpe")
    
    aggregate = {}
    
    for approach in approaches:
        sharpes = []
        sortinos = []
        ics = []
        maxdds = []
        vols = []
        runtimes = []
        
        for w in all_windows:
            if approach in w and "error" not in w[approach]:
                m = w[approach]
                sharpes.append(m.get("sharpe", 0.0))
                sortinos.append(m.get("sortino", 0.0))
                ics.append(m.get("ic_mean", 0.0))
                maxdds.append(m.get("max_dd", 0.0))
                vols.append(m.get("vol", 0.0))
                runtimes.append(m.get("runtime_s", 0.0))
        
        if sharpes:
            aggregate[approach] = {
                "n_windows": len(sharpes),
                "sharpe_mean": float(np.mean(sharpes)),
                "sharpe_std": float(np.std(sharpes)),
                "sharpe_min": float(np.min(sharpes)),
                "sharpe_max": float(np.max(sharpes)),
                "sortino_mean": float(np.mean(sortinos)),
                "sortino_std": float(np.std(sortinos)),
                "ic_mean": float(np.mean(ics)),
                "ic_std": float(np.std(ics)),
                "max_dd_mean": float(np.mean(maxdds)),
                "vol_mean": float(np.mean(vols)),
                "runtime_total_s": float(np.sum(runtimes)),
                "runtime_avg_s": float(np.mean(runtimes)),
            }
            logger.info(
                f"  {approach:25s}: Sharpe={aggregate[approach]['sharpe_mean']:.3f}±"
                f"{aggregate[approach]['sharpe_std']:.3f}, "
                f"IC={aggregate[approach]['ic_mean']:.4f}, "
                f"Sortino={aggregate[approach]['sortino_mean']:.3f}"
            )
        else:
            aggregate[approach] = {"error": "No successful windows"}
    
    results["aggregate"] = aggregate
    
    # ═══════════ 5. Statistical Comparison ═══════════
    # Paired t-test: each non-baseline approach vs baseline
    from scipy.stats import ttest_rel
    
    baseline_sharpes = []
    for w in all_windows:
        if "lgb_mse" in w and "error" not in w["lgb_mse"]:
            baseline_sharpes.append(w["lgb_mse"]["sharpe"])
    
    comparisons = {}
    for approach in approaches:
        if approach == "lgb_mse":
            continue
        
        approach_sharpes = []
        for w in all_windows:
            if approach in w and "error" not in w[approach]:
                approach_sharpes.append(w[approach]["sharpe"])
        
        if len(baseline_sharpes) >= 3 and len(approach_sharpes) >= 3:
            # Align on windows that have both
            b_aligned = []
            a_aligned = []
            for w in all_windows:
                if ("lgb_mse" in w and "error" not in w["lgb_mse"] and
                    approach in w and "error" not in w[approach]):
                    b_aligned.append(w["lgb_mse"]["sharpe"])
                    a_aligned.append(w[approach]["sharpe"])
            
            if len(b_aligned) >= 3:
                t_stat, p_val = ttest_rel(a_aligned, b_aligned)
                delta = np.mean(np.array(a_aligned) - np.array(b_aligned))
                comparisons[approach] = {
                    "delta_sharpe": float(delta),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "significant_05": bool(p_val < 0.05),
                    "n_paired_windows": len(b_aligned),
                }
                logger.info(
                    f"  {approach} vs baseline: ΔSharpe={delta:+.3f}, "
                    f"t={t_stat:.3f}, p={p_val:.4f}"
                )
    
    results["comparisons"] = comparisons
    
    # ═══════════ 6. Summary ═══════════
    elapsed = time.time() - t_start
    results["runtime_total_s"] = elapsed
    results["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    
    # Print final summary table
    _print_summary(results)
    
    return results


def _print_summary(results: Dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("  PORTFOLIO OPTIMIZER — FINAL RESULTS")
    print("=" * 80)
    
    data = results.get("data", {})
    print(f"  Data: {data.get('n_stocks', '?')} stocks × "
          f"{data.get('n_days', '?')} days, "
          f"{data.get('n_features', '?')} features")
    print(f"  Date range: {data.get('date_range', ['?', '?'])}")
    print(f"  Walk-forward windows: {results['windows']['n_windows']}")
    print()
    
    agg = results.get("aggregate", {})
    comparisons = results.get("comparisons", {})
    
    # Header
    print(f"  {'Approach':<25s} {'Sharpe':>8s}  {'IC':>7s}  {'Sortino':>8s}  "
          f"{'MaxDD':>7s}  {'ΔSharpe':>9s}  {'p-val':>6s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*6}")
    
    # Baselines (no Δ vs baseline comparison needed)
    for name, key in [("Random", "random"), ("Mean-Reversion", "meanrev")]:
        a = agg.get(key, {})
        if "sharpe_mean" in a:
            print(f"  {name:<25s} "
                  f"{a['sharpe_mean']:>7.3f}±{a['sharpe_std']:.2f}  "
                  f"{a['ic_mean']:>6.4f}  "
                  f"{a['sortino_mean']:>7.3f}  "
                  f"{a['max_dd_mean']:>6.3f}  "
                  f"{'---':>9s}  {'---':>6s}")
    
    # Separator
    print(f"  {'-'*25}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*6}")
    
    # Baseline model
    bl = agg.get("lgb_mse", {})
    if "sharpe_mean" in bl:
        print(f"  {'LGB-MSE (baseline)':<25s} "
              f"{bl['sharpe_mean']:>7.3f}±{bl['sharpe_std']:.2f}  "
              f"{bl['ic_mean']:>6.4f}  "
              f"{bl['sortino_mean']:>7.3f}  "
              f"{bl['max_dd_mean']:>6.3f}")
    
    # Other approaches
    for name, key in [("LGB-LambdaRank", "lgb_lambdarank"),
                       ("LGB-SharpeTuned", "lgb_sharpe_tuned"),
                       ("NN-Sharpe", "nn_sharpe")]:
        a = agg.get(key, {})
        if "sharpe_mean" not in a:
            continue
        comp = comparisons.get(key, {})
        delta_str = f"{comp.get('delta_sharpe', 0):+.3f}" if comp else "N/A"
        p_str = f"{comp.get('p_value', 1):.4f}" if comp else "N/A"
        sig = " *" if comp.get("significant_05") else ""
        print(f"  {name:<25s} "
              f"{a['sharpe_mean']:>7.3f}±{a['sharpe_std']:.2f}  "
              f"{a['ic_mean']:>6.4f}  "
              f"{a['sortino_mean']:>7.3f}  "
              f"{a['max_dd_mean']:>6.3f}  "
              f"{delta_str:>9s}  "
              f"{p_str:>6s}{sig}")
    
    print(f"\n  * p < 0.05")
    print(f"  Runtime: {results.get('runtime_total_s', 0):.0f}s")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Portfolio-Level Sharpe Optimization Experiment"
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Directory containing stock CSV files"
    )
    parser.add_argument(
        "--n-stocks", type=int, default=N_STOCKS,
        help="Number of top stocks by liquidity"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output JSON path"
    )
    parser.add_argument(
        "--skip-nn", action="store_true",
        help="Skip neural network training (faster)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    if not HAS_LGB:
        logger.error("LightGBM is required. Install: pip install lightgbm")
        sys.exit(1)
    
    run_experiment(
        data_dir=args.data_dir,
        n_stocks=args.n_stocks,
        output_path=args.output,
        skip_nn=args.skip_nn,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
