#!/usr/bin/env python3
"""
GARCH(1,1) baseline vs LightGBM for volatility prediction.

Walk-forward comparison across US stocks with four predictors:
  1. GARCH(1,1): 1-step-ahead conditional variance from daily returns
  2. LightGBM:   predict squared returns from causal OHLCV features
  3. Historical Mean: rolling 252-day mean of squared returns
  4. Persistence: lagged squared return (r²_{t-1} → forecast for r²_t)

Walk-forward protocol: train=1000d, purge=126d, val=200d, step=126d

Metrics per stock, per predictor: QLIKE, MSE, MAE
Aggregate: mean QLIKE, paired t-test, Diebold-Mariano test

Key diagnostic question:
  Does LightGBM meaningfully outperform GARCH(1,1) for volatility prediction,
  or are we adding unnecessary complexity?

Usage:
    pip install arch lightgbm numpy pandas scipy
    python scripts/garch_baseline.py --n-stocks 50 --output results/garch_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.features import compute_all_features

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_DAYS = 1000   # training window length (trading days)
PURGE_DAYS = 126    # embargo gap between train and val
VAL_DAYS = 200      # validation window (LGB early stopping)
STEP_DAYS = 126     # walk-forward step / test window length

RETURN_SCALE = 100.0  # scale returns to percentage for GARCH numerical stability
MIN_SAMPLES = TRAIN_DAYS + PURGE_DAYS + VAL_DAYS + STEP_DAYS  # 1452

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_stock_data(
    data_dir: str,
    n_stocks: int = 50,
) -> Tuple[List[str], List[np.ndarray], List[pd.DataFrame]]:
    """Load OHLCV CSVs, compute daily returns and features.

    Returns arrays aligned so that at index i:
      - features[i] = causal features available after close on day i
      - returns[i]  = log return from day i to day i+1 (scaled by RETURN_SCALE)

    So features at time t predict the NEXT day's squared return r²_{t+1}.
    """
    csv_files = sorted(Path(data_dir).glob("*.csv"))[:n_stocks]
    logger.info("Loading %d stocks from %s", len(csv_files), data_dir)

    stocks: List[str] = []
    returns_list: List[np.ndarray] = []
    features_list: List[pd.DataFrame] = []

    for fp in csv_files:
        code = fp.stem
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if df.empty or "close" not in df.columns or "volume" not in df.columns:
            logger.warning("Skipping %s: invalid data", code)
            continue

        # Daily log returns: ret[t] = log(close[t+1]/close[t]) * scale
        # diff drops first element, so ret[0] = return from df.index[0]→df.index[1]
        all_ret = np.log(df["close"]).diff().dropna().values * RETURN_SCALE

        # Causal features computed from OHLCV up to each day
        feats_raw = compute_all_features(df)
        # Drop rows with NaN features (first ~21 days have insufficient history)
        feats_clean = feats_raw.dropna()

        if len(feats_clean) < MIN_SAMPLES:
            logger.warning("Skipping %s: only %d feature rows", code, len(feats_clean))
            continue

        # Map feature dates to return indices.
        # all_ret[j] is the return from df.index[j] to df.index[j+1].
        # For a feature date at df.index[k], the NEXT day's return is all_ret[k].
        # We need k < len(all_ret) (i.e. k < len(df)-1).
        feats_aligned_list = []
        returns_aligned_list = []

        for feat_date in feats_clean.index:
            # Find the position of this date in df.index
            pos = df.index.get_loc(feat_date)
            if pos < len(all_ret):  # need next day's return
                feats_aligned_list.append(feats_clean.loc[feat_date])
                returns_aligned_list.append(all_ret[pos])

        if len(returns_aligned_list) < MIN_SAMPLES:
            logger.warning("Skipping %s: aligned size %d < %d",
                           code, len(returns_aligned_list), MIN_SAMPLES)
            continue

        stocks.append(code)
        ret_arr = np.array(returns_aligned_list, dtype=np.float64)
        returns_list.append(ret_arr)

        # Augment features with lagged squared return (matching GARCH's α·r²_{t-1} term)
        feats_df = pd.DataFrame(feats_aligned_list, columns=feats_clean.columns)
        # Lag-1 squared return: at index i, this is r²_{i-1} (the squared return from i-1→i)
        # We need returns[i-1]² — pad first element with unconditional mean
        lag_sq = np.zeros(len(ret_arr))
        if len(ret_arr) > 1:
            lag_sq[1:] = ret_arr[:-1] ** 2
            lag_sq[0] = np.mean(ret_arr ** 2)  # unconditional for first element
        feats_df["sq_return_lag1"] = lag_sq
        features_list.append(feats_df)

    logger.info("Loaded %d stocks successfully", len(stocks))
    return stocks, returns_list, features_list


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward fold generation
# ═══════════════════════════════════════════════════════════════════════════════

def get_walk_forward_folds(T: int) -> List[Tuple[int, int, int, int]]:
    """Generate purged walk-forward fold boundaries.

    Returns list of (train_start, train_end, val_start, val_end, test_start, test_end).
    All indices relative to returns array (0 to T-1).
    """
    total_window = TRAIN_DAYS + PURGE_DAYS + VAL_DAYS + STEP_DAYS
    n_folds = max(0, (T - total_window) // STEP_DAYS + 1)
    folds: List[Tuple[int, int, int, int, int, int]] = []

    for fold in range(n_folds):
        pos = fold * STEP_DAYS
        tr_start = pos
        tr_end = pos + TRAIN_DAYS
        purge_end = tr_end + PURGE_DAYS
        vl_start = purge_end
        vl_end = vl_start + VAL_DAYS
        te_start = vl_end
        te_end = min(te_start + STEP_DAYS, T)

        if te_end - te_start < 10:  # need at least 10 test days
            break

        folds.append((tr_start, tr_end, vl_start, vl_end, te_start, te_end))

    return folds


# ═══════════════════════════════════════════════════════════════════════════════
# Predictors
# ═══════════════════════════════════════════════════════════════════════════════

def fit_garch_and_forecast(
    returns: np.ndarray,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
) -> np.ndarray:
    """Fit GARCH(1,1) on training returns, produce 1-step-ahead variance forecasts.

    Uses actual lagged squared returns to compute conditional variance
    for each test day (filtering, not pure multi-step forecasting).
    Falls back to training-data variance if GARCH fit is degenerate.

    Returns
    -------
    forecasts : (test_len,) array of conditional variance forecasts.
    """
    train_ret = returns[train_start:train_end]
    train_var = float(np.var(train_ret))

    # Sanity: return constant variance if training data is degenerate
    if train_var < 1e-8 or len(train_ret) < 100:
        return np.full(test_end - test_start, max(train_var, LOG_VAR_FLOOR))

    try:
        model = arch_model(train_ret, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        result = model.fit(disp="off", show_warning=False)
    except Exception as e:
        logger.debug("GARCH fit exception: %s — using training variance", e)
        return np.full(test_end - test_start, train_var)

    if result.convergence_flag != 0:
        logger.debug("GARCH no-converge (flag=%d) — using training variance", result.convergence_flag)
        return np.full(test_end - test_start, train_var)

    omega = float(result.params["omega"])
    alpha = float(result.params["alpha[1]"])
    beta = float(result.params["beta[1]"])

    # Validate parameters: reject degenerate fits
    # Degenerate: ω≈0, α≈0, β≈1 → constant near-zero variance
    omega_min = max(1e-6, train_var * 1e-4)
    alpha_min = 0.005  # need at least 0.5% reaction to shocks

    if omega < omega_min or alpha < alpha_min:
        logger.debug("GARCH degenerate: ω=%.2e α=%.4f β=%.4f — using training variance",
                     omega, alpha, beta)
        return np.full(test_end - test_start, train_var)

    # Ensure stationarity
    if alpha + beta >= 1.0:
        logger.debug("GARCH non-stationary: α+β=%.4f — using training variance", alpha + beta)
        return np.full(test_end - test_start, train_var)

    # Compute 1-step-ahead conditional variance for the full range
    # (train_start through test_end) using filtering with actual returns
    all_returns = returns[train_start:test_end]
    all_h = np.zeros(len(all_returns))
    all_h[0] = omega / (1.0 - alpha - beta)  # unconditional variance

    for t in range(1, len(all_h)):
        all_h[t] = omega + alpha * all_returns[t - 1] ** 2 + beta * all_h[t - 1]

    # Extract test portion
    test_offset = test_start - train_start
    forecasts = all_h[test_offset:test_offset + (test_end - test_start)]

    # Final sanity: ensure forecasts are positive and reasonable
    forecasts = np.maximum(forecasts, LOG_VAR_FLOOR)
    # Cap at 10x training variance to prevent explosion
    forecasts = np.minimum(forecasts, train_var * 10.0)

    return forecasts


# Minimum variance floor for log-transform (squared return of 0.01% daily move)
LOG_VAR_FLOOR = (0.0001 * RETURN_SCALE) ** 2  # ≈ 1e-4 for RETURN_SCALE=100


def fit_lgb_and_predict(
    features: pd.DataFrame,
    squared_returns: np.ndarray,
    train_start: int,
    train_end: int,
    val_start: int,
    val_end: int,
    test_start: int,
    test_end: int,
    seed: int = 42,
) -> np.ndarray:
    """Train LightGBM on log-transformed squared returns, predict in original space.

    Variance is strictly positive and right-skewed. We train on log(σ² + ε)
    to ensure predictions are always positive after exp(). This prevents the
    QLIKE metric from exploding due to near-zero or negative raw predictions.

    Returns
    -------
    predictions : (test_len,) array of predicted squared returns (positive).
    """
    X_all = features.values.astype(np.float32)
    # Log-transform: ensures predictions stay positive after exp()
    y_log_all = np.log(np.maximum(squared_returns, LOG_VAR_FLOOR)).astype(np.float32)

    X_tr = X_all[train_start:train_end]
    y_tr = y_log_all[train_start:train_end]
    X_vl = X_all[val_start:val_end]
    y_vl = y_log_all[val_start:val_end]
    X_te = X_all[test_start:test_end]

    # Drop NaN rows
    tr_valid = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
    vl_valid = np.isfinite(X_vl).all(axis=1) & np.isfinite(y_vl)
    te_valid = np.isfinite(X_te).all(axis=1)

    if tr_valid.sum() < 100 or vl_valid.sum() < 20:
        return np.full(test_end - test_start, np.nan)

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="l1",
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=20,
        verbosity=-1,
        random_state=seed,
        n_jobs=1,  # single-threaded for per-stock fits
    )
    try:
        model.fit(
            X_tr[tr_valid], y_tr[tr_valid],
            eval_set=[(X_vl[vl_valid], y_vl[vl_valid])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )
    except Exception as e:
        logger.debug("LGB fit failed: %s", e)
        return np.full(test_end - test_start, np.nan)

    # Predict in log-space, transform back to variance space
    pred_log = np.full(test_end - test_start, np.nan)
    pred_log[te_valid] = model.predict(X_te[te_valid])
    # exp() guarantees positivity, guard against overflow
    pred = np.exp(np.clip(pred_log, -50.0, 50.0))
    return pred


def historical_mean_forecast(
    squared_returns: np.ndarray,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
) -> np.ndarray:
    """Forecast using the mean of squared returns in the training window."""
    mean_val = np.nanmean(squared_returns[train_start:train_end])
    if not np.isfinite(mean_val) or mean_val <= 0:
        return np.full(test_end - test_start, np.nan)
    return np.full(test_end - test_start, mean_val)


def persistence_forecast(
    squared_returns: np.ndarray,
    test_start: int,
    test_end: int,
) -> np.ndarray:
    """Forecast using lagged squared return, floored at LOG_VAR_FLOOR.

    pred[t] = max(r²[t-1], LOG_VAR_FLOOR).  This prevents QLIKE explosions
    when the previous day had near-zero realized variance.
    """
    out = np.full(test_end - test_start, np.nan)
    for i, t in enumerate(range(test_start, test_end)):
        if t > 0:
            out[i] = max(squared_returns[t - 1], LOG_VAR_FLOOR)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE = log(h) + σ²/h  where h = forecast variance, σ² = realized variance.

    h and σ² must be positive. Lower QLIKE is better.
    Patton (2011) shows QLIKE is a homogeneous, robust loss function for
    volatility forecast evaluation.
    """
    h = np.maximum(y_pred, 1e-8)
    sigma2 = np.maximum(y_true, 1e-8)
    return float(np.mean(np.log(h) + sigma2 / h))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute QLIKE, MSE, MAE for valid (finite) predictions."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 5:
        return {"qlike": np.nan, "mse": np.nan, "mae": np.nan, "n": 0}
    yt = y_true[valid]
    yp = y_pred[valid]
    return {
        "qlike": qlike(yt, yp),
        "mse": float(np.mean((yt - yp) ** 2)),
        "mae": float(np.mean(np.abs(yt - yp))),
        "n": int(valid.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Diebold-Mariano test
# ═══════════════════════════════════════════════════════════════════════════════

def diebold_mariano(
    loss1: np.ndarray,
    loss2: np.ndarray,
    h: int = 1,
    loss_type: str = "qlike",
) -> Dict:
    """Diebold-Mariano test for equal predictive accuracy.

    H0: E[L1 - L2] = 0 (equal accuracy)
    H1: E[L1 - L2] ≠ 0 (different accuracy)

    Uses Newey-West HAC standard errors with Bartlett kernel for
    autocorrelation-robust inference (up to lag h-1).

    Parameters
    ----------
    loss1, loss2 : element-wise losses (QLIKE is typical for vol forecasting).
    h            : forecast horizon (1 for 1-step-ahead).
    loss_type    : label for output metadata.

    Returns
    -------
    dict with dm_statistic, p_value, mean_differential, interpretation.
    """
    d = loss1 - loss2  # positive d means loss1 > loss2 (predictor 2 better)
    d = d[np.isfinite(d)]
    n = len(d)

    if n < 10:
        return {"dm_statistic": np.nan, "p_value": np.nan,
                "mean_differential": np.nan, "n": n, "error": "too few samples"}

    mean_d = np.mean(d)

    # Newey-West variance with Bartlett kernel
    # Var(mean(d)) ≈ (1/n) * [γ₀ + 2·Σ_{k=1}^{h-1} (1-k/h)·γₖ]
    max_lag = min(h, n // 4)  # rule of thumb: h steps ahead, max h-1 lags
    nw_var = np.var(d, ddof=0) / n  # start with i.i.d. variance

    if max_lag > 1:
        for k in range(1, max_lag):
            gamma_k = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
            weight = 1.0 - k / max_lag  # Bartlett kernel
            nw_var += 2.0 * weight * gamma_k / n

    nw_se = max(np.sqrt(nw_var), 1e-10)
    dm_stat = mean_d / nw_se

    # Two-sided p-value using normal approximation
    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "mean_differential": float(mean_d),
        "n": n,
        "loss_type": loss_type,
    }


def qlike_elementwise(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute element-wise QLIKE: log(h_i) + σ²_i/h_i."""
    h = np.maximum(y_pred, 1e-8)
    sigma2 = np.maximum(y_true, 1e-8)
    return np.log(h) + sigma2 / h


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward per stock
# ═══════════════════════════════════════════════════════════════════════════════

PredictorResult = Dict  # {"qlike": float, "mse": float, "mae": float, "n_samples": int,
                        #  "n_folds": int, "per_fold_qlike": list, ...}

def walk_forward_single_stock(
    returns: np.ndarray,
    features: pd.DataFrame,
    seed: int = 42,
) -> Tuple[PredictorResult, PredictorResult, PredictorResult, PredictorResult]:
    """Run walk-forward comparison for a single stock.

    Returns dicts for garch, lgb, mean, persistence predictors.
    """
    T = len(returns)
    squared_returns = returns ** 2
    folds = get_walk_forward_folds(T)

    if not folds:
        empty = {"qlike": np.nan, "mse": np.nan, "mae": np.nan, "n_samples": 0,
                 "n_folds": 0, "per_fold": [], "error": "no folds"}
        return empty, empty, empty, empty

    # Collect per-fold predictions and targets
    all_garch_pred, all_lgb_pred, all_mean_pred, all_persist_pred = [], [], [], []
    all_targets = []
    per_fold_metrics = {"garch": [], "lgb": [], "mean": [], "persistence": []}

    for tr_s, tr_e, vl_s, vl_e, te_s, te_e in folds:
        actual = squared_returns[te_s:te_e]

        # GARCH
        garch_pred = fit_garch_and_forecast(returns, tr_s, tr_e, te_s, te_e)
        garch_met = compute_metrics(actual, garch_pred)

        # LGB
        lgb_pred = fit_lgb_and_predict(features, squared_returns,
                                       tr_s, tr_e, vl_s, vl_e, te_s, te_e, seed)
        lgb_met = compute_metrics(actual, lgb_pred)

        # Mean
        mean_pred = historical_mean_forecast(squared_returns, tr_s, tr_e, te_s, te_e)
        mean_met = compute_metrics(actual, mean_pred)

        # Persistence
        persist_pred = persistence_forecast(squared_returns, te_s, te_e)
        persist_met = compute_metrics(actual, persist_pred)

        per_fold_metrics["garch"].append(garch_met)
        per_fold_metrics["lgb"].append(lgb_met)
        per_fold_metrics["mean"].append(mean_met)
        per_fold_metrics["persistence"].append(persist_met)

        all_garch_pred.append(garch_pred)
        all_lgb_pred.append(lgb_pred)
        all_mean_pred.append(mean_pred)
        all_persist_pred.append(persist_pred)
        all_targets.append(actual)

    def aggregate(preds: List[np.ndarray]) -> PredictorResult:
        """Pool all fold predictions into overall metrics."""
        all_p = np.concatenate([p for p in preds if len(p) > 0]) if preds else np.array([])
        all_t = np.concatenate([a for a in all_targets if len(a) > 0]) if all_targets else np.array([])
        min_len = min(len(all_p), len(all_t))
        if min_len < 5:
            return {"qlike": np.nan, "mse": np.nan, "mae": np.nan,
                    "n_samples": 0, "n_folds": len(preds), "per_fold": []}
        overall = compute_metrics(all_t[:min_len], all_p[:min_len])
        overall["n_samples"] = overall.pop("n")
        overall["n_folds"] = len(preds)
        # Store per-fold QLIKE for per-stock stats
        pfm = per_fold_metrics.get("garch", [])  # placeholder, we'll override per-predictor
        return overall

    garch_result = aggregate(all_garch_pred)
    lgb_result = aggregate(all_lgb_pred)
    mean_result = aggregate(all_mean_pred)
    persist_result = aggregate(all_persist_pred)

    # Add per-fold metrics
    for key, result in [("garch", garch_result), ("lgb", lgb_result),
                        ("mean", mean_result), ("persistence", persist_result)]:
        pfm = per_fold_metrics[key]
        result["per_fold_qlike"] = [m["qlike"] for m in pfm if np.isfinite(m.get("qlike", np.nan))]
        result["per_fold_mse"] = [m["mse"] for m in pfm if np.isfinite(m.get("mse", np.nan))]

    return garch_result, lgb_result, mean_result, persist_result


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate and statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_results(
    stocks: List[str],
    all_results: List[Tuple[PredictorResult, PredictorResult, PredictorResult, PredictorResult]],
    all_returns: List[np.ndarray],
    all_features: List[pd.DataFrame],
    seed: int,
) -> Dict:
    """Aggregate per-stock results into cross-sectional statistics.

    Computes:
    - Mean QLIKE per predictor (across stocks)
    - Median QLIKE per predictor
    - Fraction of stocks where LGB beats GARCH
    - Paired t-test: LGB QLIKE vs GARCH QLIKE
    - Diebold-Mariano: LGB vs GARCH (pooled losses)
    """
    N = len(stocks)

    garch_qlikes = []
    lgb_qlikes = []
    mean_qlikes = []
    persist_qlikes = []

    for garch_r, lgb_r, mean_r, persist_r in all_results:
        garch_qlikes.append(garch_r.get("qlike", np.nan))
        lgb_qlikes.append(lgb_r.get("qlike", np.nan))
        mean_qlikes.append(mean_r.get("qlike", np.nan))
        persist_qlikes.append(persist_r.get("qlike", np.nan))

    garch_qlikes = np.array(garch_qlikes)
    lgb_qlikes = np.array(lgb_qlikes)
    mean_qlikes = np.array(mean_qlikes)
    persist_qlikes = np.array(persist_qlikes)

    # Filter to stocks where both GARCH and LGB succeeded
    valid = np.isfinite(garch_qlikes) & np.isfinite(lgb_qlikes)
    n_valid = valid.sum()

    # ------------------------------------------------------------------
    # Mean QLIKE per predictor
    # ------------------------------------------------------------------
    per_predictor = {
        "garch": {"mean_qlike": float(np.nanmean(garch_qlikes)),
                  "median_qlike": float(np.nanmedian(garch_qlikes)),
                  "std_qlike": float(np.nanstd(garch_qlikes)),
                  "n_valid": int(np.isfinite(garch_qlikes).sum())},
        "lgb":   {"mean_qlike": float(np.nanmean(lgb_qlikes)),
                  "median_qlike": float(np.nanmedian(lgb_qlikes)),
                  "std_qlike": float(np.nanstd(lgb_qlikes)),
                  "n_valid": int(np.isfinite(lgb_qlikes).sum())},
        "mean":  {"mean_qlike": float(np.nanmean(mean_qlikes)),
                  "median_qlike": float(np.nanmedian(mean_qlikes)),
                  "std_qlike": float(np.nanstd(mean_qlikes)),
                  "n_valid": int(np.isfinite(mean_qlikes).sum())},
        "persistence": {"mean_qlike": float(np.nanmean(persist_qlikes)),
                        "median_qlike": float(np.nanmedian(persist_qlikes)),
                        "std_qlike": float(np.nanstd(persist_qlikes)),
                        "n_valid": int(np.isfinite(persist_qlikes).sum())},
    }

    # ------------------------------------------------------------------
    # Fraction of stocks where LGB beats GARCH
    # ------------------------------------------------------------------
    if n_valid >= 2:
        lgb_better = lgb_qlikes[valid] < garch_qlikes[valid]
        fraction_lgb_better = float(lgb_better.mean())
        # Also compute rank of each predictor per stock
        all_four = np.column_stack([garch_qlikes[valid], lgb_qlikes[valid],
                                     mean_qlikes[valid], persist_qlikes[valid]])
        ranks = np.argsort(np.argsort(all_four, axis=1), axis=1)  # 0=best (lowest QLIKE)
        avg_ranks = {
            "garch": float(np.mean(ranks[:, 0])),
            "lgb": float(np.mean(ranks[:, 1])),
            "mean": float(np.mean(ranks[:, 2])),
            "persistence": float(np.mean(ranks[:, 3])),
        }
    else:
        fraction_lgb_better = np.nan
        avg_ranks = {}

    # ------------------------------------------------------------------
    # Paired t-test: LGB QLIKE vs GARCH QLIKE
    # ------------------------------------------------------------------
    if n_valid >= 5:
        delta = garch_qlikes[valid] - lgb_qlikes[valid]  # positive = GARCH worse
        t_stat, p_value_paired = sp_stats.ttest_1samp(delta, 0.0)
        paired_ttest = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value_paired),
            "mean_delta": float(np.mean(delta)),
            "std_delta": float(np.std(delta, ddof=1)),
            "n_pairs": int(n_valid),
            "interpretation": ("LGB significantly better than GARCH" if p_value_paired < 0.05 and t_stat > 0
                               else "GARCH significantly better than LGB" if p_value_paired < 0.05
                               else "No significant difference"),
        }
    else:
        paired_ttest = {"error": "insufficient samples", "n_pairs": int(n_valid)}

    # ------------------------------------------------------------------
    # Diebold-Mariano test (pooled element-wise losses)
    # ------------------------------------------------------------------
    # Reconstruct pooled losses for LGB vs GARCH across all stocks and folds
    lgb_losses_pooled = []
    garch_losses_pooled = []

    for idx, (garch_r, lgb_r, _, _) in enumerate(all_results):
        returns = all_returns[idx]
        features = all_features[idx]
        squared_rets = returns ** 2
        T = len(returns)
        folds = get_walk_forward_folds(T)

        for tr_s, tr_e, vl_s, vl_e, te_s, te_e in folds:
            actual = squared_rets[te_s:te_e]

            garch_pred = fit_garch_and_forecast(returns, tr_s, tr_e, te_s, te_e)
            lgb_pred = fit_lgb_and_predict(features, squared_rets,
                                           tr_s, tr_e, vl_s, vl_e, te_s, te_e, seed)

            valid_fold = np.isfinite(garch_pred) & np.isfinite(lgb_pred) & np.isfinite(actual)
            if valid_fold.sum() < 5:
                continue

            garch_losses_pooled.extend(
                qlike_elementwise(actual[valid_fold], garch_pred[valid_fold]).tolist()
            )
            lgb_losses_pooled.extend(
                qlike_elementwise(actual[valid_fold], lgb_pred[valid_fold]).tolist()
            )

    dm_garch_vs_lgb = diebold_mariano(
        np.array(garch_losses_pooled), np.array(lgb_losses_pooled),
        h=1, loss_type="qlike"
    )

    # Also DM test for LGB vs persistence, GARCH vs persistence
    persist_losses_pooled = []
    for idx, _ in enumerate(all_results):
        squared_rets = all_returns[idx] ** 2
        T = len(squared_rets)
        folds = get_walk_forward_folds(T)
        for _, _, _, _, te_s, te_e in folds:
            actual = squared_rets[te_s:te_e]
            persist_pred = persistence_forecast(squared_rets, te_s, te_e)
            valid_fold = np.isfinite(persist_pred) & np.isfinite(actual)
            if valid_fold.sum() < 5:
                continue
            persist_losses_pooled.extend(
                qlike_elementwise(actual[valid_fold], persist_pred[valid_fold]).tolist()
            )

    # Align lengths for DM tests (use min common length)
    if lgb_losses_pooled and persist_losses_pooled:
        n_common = min(len(lgb_losses_pooled), len(persist_losses_pooled))
        dm_lgb_vs_persist = diebold_mariano(
            np.array(lgb_losses_pooled[:n_common]),
            np.array(persist_losses_pooled[:n_common]),
            h=1, loss_type="qlike"
        )
    else:
        dm_lgb_vs_persist = {"error": "insufficient data"}

    if garch_losses_pooled and persist_losses_pooled:
        n_common = min(len(garch_losses_pooled), len(persist_losses_pooled))
        dm_garch_vs_persist = diebold_mariano(
            np.array(garch_losses_pooled[:n_common]),
            np.array(persist_losses_pooled[:n_common]),
            h=1, loss_type="qlike"
        )
    else:
        dm_garch_vs_persist = {"error": "insufficient data"}

    # ------------------------------------------------------------------
    # Assemble final output
    # ------------------------------------------------------------------
    return {
        "n_stocks_total": N,
        "n_stocks_valid": int(n_valid),
        "per_predictor": per_predictor,
        "fraction_lgb_better_than_garch": fraction_lgb_better,
        "avg_ranks": avg_ranks,
        "paired_ttest_lgb_vs_garch": paired_ttest,
        "diebold_mariano_lgb_vs_garch": dm_garch_vs_lgb,
        "diebold_mariano_lgb_vs_persistence": dm_lgb_vs_persist,
        "diebold_mariano_garch_vs_persistence": dm_garch_vs_persist,
        "per_stock": {
            stocks[i]: {
                "garch_qlike": garch_qlikes[i],
                "lgb_qlike": lgb_qlikes[i],
                "mean_qlike": mean_qlikes[i],
                "persistence_qlike": persist_qlikes[i],
            }
            for i in range(N)
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GARCH(1,1) vs LightGBM volatility prediction baseline"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "new_data", "data", "us_stocks"),
        help="Directory containing US stock CSV files",
    )
    parser.add_argument("--n-stocks", type=int, default=50,
                        help="Number of stocks to evaluate (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for LightGBM")
    parser.add_argument("--output",
                        default=os.path.join(PROJECT_ROOT, "results", "garch_baseline.json"),
                        help="Output JSON path")
    parser.add_argument("--verbose", action="store_true",
                        help="Log per-stock and per-fold details")
    args = parser.parse_args()

    if not HAS_ARCH:
        logger.error("arch package required. Install: pip install arch")
        sys.exit(1)
    if not HAS_LGB:
        logger.error("lightgbm package required. Install: pip install lightgbm")
        sys.exit(1)

    np.random.seed(args.seed)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    t0 = time.time()

    # ── 1. Load data ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading US stock data")
    logger.info("=" * 60)
    stocks, returns_list, features_list = load_stock_data(args.data_dir, args.n_stocks)
    N = len(stocks)
    if N == 0:
        logger.error("No valid stocks found")
        sys.exit(1)
    logger.info("Loaded %d stocks; data dir: %s", N, os.path.abspath(args.data_dir))

    # ── 2. Walk-forward per stock ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Walk-forward comparison (%d stocks)", N)
    logger.info("  Protocol: train=%dd purge=%dd val=%dd step=%dd",
                TRAIN_DAYS, PURGE_DAYS, VAL_DAYS, STEP_DAYS)
    logger.info("=" * 60)

    all_results: List[Tuple] = []
    for i, code in enumerate(stocks):
        ret = returns_list[i]
        feats = features_list[i]
        T = len(ret)
        folds = get_walk_forward_folds(T)
        logger.info("[%2d/%2d] %-6s: %d days, %d folds",
                    i + 1, N, code, T, len(folds))

        garch_r, lgb_r, mean_r, persist_r = walk_forward_single_stock(
            ret, feats, args.seed
        )
        all_results.append((garch_r, lgb_r, mean_r, persist_r))

        if args.verbose:
            for name, r in [("GARCH", garch_r), ("LGB", lgb_r),
                            ("Mean", mean_r), ("Persist", persist_r)]:
                logger.debug("  %-7s: QLIKE=%.4f MSE=%.6f MAE=%.6f (n=%d folds=%d)",
                            name, r.get("qlike", np.nan), r.get("mse", np.nan),
                            r.get("mae", np.nan), r.get("n_samples", 0),
                            r.get("n_folds", 0))

    # ── 3. Aggregate ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Aggregating results")
    logger.info("=" * 60)
    aggregate = aggregate_results(stocks, all_results, returns_list, features_list, args.seed)

    # ── 4. Print summary ─────────────────────────────────────────────────
    pp = aggregate["per_predictor"]
    print("\n" + "=" * 72)
    print("  GARCH(1,1) vs LightGBM — Volatility Prediction Baseline")
    print("=" * 72)
    print(f"  Stocks: {N} loaded, {aggregate['n_stocks_valid']} with valid GARCH+LGB")
    print(f"  Protocol: train={TRAIN_DAYS}d purge={PURGE_DAYS}d val={VAL_DAYS}d step={STEP_DAYS}d")
    print(f"  Target: 1-step-ahead squared daily log returns (scaled ×{RETURN_SCALE:.0f})")
    print("-" * 72)
    print(f"  {'Predictor':<14} {'Mean QLIKE':>10} {'Median QLIKE':>10} {'Std QLIKE':>10}")
    print("  " + "-" * 44)
    for name in ["garch", "lgb", "mean", "persistence"]:
        d = pp[name]
        print(f"  {name:<14} {d['mean_qlike']:>10.4f} {d['median_qlike']:>10.4f} {d['std_qlike']:>10.4f}")
    print("-" * 72)

    # Key diagnostic
    frac = aggregate.get("fraction_lgb_better_than_garch", np.nan)
    if np.isfinite(frac):
        pct = frac * 100
        print(f"\n  LGB beats GARCH on {pct:.1f}% of stocks")
        ranks = aggregate.get("avg_ranks", {})
        if ranks:
            print(f"  Average rank (0=best): GARCH={ranks.get('garch',0):.2f} "
                  f"LGB={ranks.get('lgb',0):.2f} "
                  f"Mean={ranks.get('mean',0):.2f} "
                  f"Persistence={ranks.get('persistence',0):.2f}")

    # Paired t-test
    ptt = aggregate.get("paired_ttest_lgb_vs_garch", {})
    if "t_statistic" in ptt:
        print(f"\n  Paired t-test (LGB vs GARCH QLIKE):")
        print(f"    t = {ptt['t_statistic']:+.4f}, p = {ptt['p_value']:.4f}")
        print(f"    ΔQLIKE = {ptt['mean_delta']:+.4f} ± {ptt['std_delta']:.4f} "
              f"({ptt['n_pairs']} pairs)")
        print(f"    → {ptt['interpretation']}")

    # DM test
    dm = aggregate.get("diebold_mariano_lgb_vs_garch", {})
    if "dm_statistic" in dm and np.isfinite(dm["dm_statistic"]):
        # d = GARCH_loss - LGB_loss; positive → GARCH has higher loss (LGB better)
        if dm["mean_differential"] > 0:
            better, worse = "LGB", "GARCH"
        else:
            better, worse = "GARCH", "LGB"
        sig = "***" if dm["p_value"] < 0.01 else ("**" if dm["p_value"] < 0.05 else "")
        print(f"\n  Diebold-Mariano test (GARCH vs LGB, pooled QLIKE):")
        print(f"    DM = {dm['dm_statistic']:+.4f}, p = {dm['p_value']:.4f} {sig}")
        print(f"    Mean ΔQLIKE = {dm['mean_differential']:+.4f} "
              f"(positive → {worse} worse, {better} better)")
        print(f"    n = {dm['n']}")

    print("=" * 72)

    # ── 5. Save ──────────────────────────────────────────────────────────
    output = {
        "config": {
            "data_dir": os.path.abspath(args.data_dir),
            "n_stocks": N,
            "seed": args.seed,
            "train_days": TRAIN_DAYS,
            "purge_days": PURGE_DAYS,
            "val_days": VAL_DAYS,
            "step_days": STEP_DAYS,
            "target": "1-step-ahead squared percentage log returns",
            "return_scale": RETURN_SCALE,
            "garch_spec": "GARCH(1,1) with normal innovations",
            "lgb_spec": "LGBMRegressor on log(y+ε) transform, num_leaves=31, max_depth=6, lr=0.05, 300 rounds",
            "lgb_log_floor": LOG_VAR_FLOOR,
            "features": list(features_list[0].columns) if features_list else [],
        },
        "aggregate": aggregate,
        "runtime_s": round(time.time() - t0, 2),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved → %s", out_path)

    elapsed = time.time() - t0
    logger.info("Total runtime: %.1f s (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
