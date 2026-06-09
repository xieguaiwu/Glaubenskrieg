#!/usr/bin/env python3
"""
Glaubenskrieg Pipeline v2 — Multi-horizon + Enhanced Features + Strategy Improvements.

P0: Multi-horizon targets (1d, 5d, 21d forward returns + vol-adjusted)
P1: Enhanced features (cross-sectional z-scores, momentum ranks, Amihud illiquidity)
P2: Strategy improvements (regime-adaptive weighting, portfolio optimization, rebalancing)

Run on remote server:
  cd /root/Glaubenskrieg
  PYTHONPATH=. python scripts/pipeline_v2.py --data-dir /root/data/us_stocks_full --output /root/results/pipeline_v2.json
"""
import sys, os, json, time, argparse, warnings, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline_v2")

# ── Paths ──────────────────────────────────────────────
PROJECT = Path("/root/Glaubenskrieg")
sys.path.insert(0, str(PROJECT))
from src.data.features import compute_all_features

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not installed")

from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
MIN_ROWS = 2500       # ~10 years of daily data
WF_TRAIN = 1008       # 4 years
WF_PURGE = 126        # 6 months
WF_VAL = 252          # 1 year
WF_STEP = 126         # 6 months

# Multi-horizon targets
HORIZONS = [1, 5, 21]
HORIZON_LABELS = {1: "1d", 5: "5d", 21: "21d"}

LGB_PARAMS = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.03,
    "num_leaves": 31, "min_child_samples": 50, "subsample": 0.8,
    "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 0.5,
    "n_jobs": -1, "verbosity": -1,
}
TOP_K = 100
REBALANCE_FREQ = 5

# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════
def sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.mean(returns) / max(np.std(returns), 1e-10)) * np.sqrt(252)

def ic_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return 0.0
    return float(spearmanr(y_true[mask], y_pred[mask])[0])

def dir_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return 0.5
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))

def max_drawdown(equity: np.ndarray) -> Tuple[float, int]:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd)), int(np.argmin(dd))

def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-10
    ratio = y_true / np.maximum(y_pred, eps)
    return float(np.mean(np.log(ratio) + 1.0 / np.maximum(ratio, eps)))

# ═══════════════════════════════════════════════════════════
# P1: Cross-Sectional Feature Engineering
# ═══════════════════════════════════════════════════════════
def add_cross_sectional_features(X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Add cross-sectional features computed per-day across all stocks.

    X: (n_days, n_stocks, n_feat) float32 array.

    Adds per-feature cross-sectional z-scores and percentile ranks,
    plus pairwise interaction features.

    Returns (augmented_X, new_feature_names).
    """
    n_days, n_stocks, n_feat = X.shape
    new_features = []
    cs_blocks = []

    # Cross-sectional z-scores per feature
    for f in range(n_feat):
        feat_daily = X[:, :, f]  # (n_days, n_stocks)
        # Per-day mean and std across stocks
        mean_d = np.nanmean(feat_daily, axis=1, keepdims=True)
        std_d = np.nanstd(feat_daily, axis=1, keepdims=True)
        std_d = np.maximum(std_d, 1e-8)
        zscore = (feat_daily - mean_d) / std_d
        zscore = np.nan_to_num(zscore, 0.0)
        cs_blocks.append(zscore[:, :, np.newaxis])
        new_features.append(f"cs_z_{f}")

    # Cross-sectional percentile ranks per feature
    for f in range(n_feat):
        feat_daily = X[:, :, f]
        ranks = np.zeros_like(feat_daily)
        for d in range(n_days):
            day_vals = feat_daily[d]
            valid = ~np.isnan(day_vals)
            if valid.sum() > 1:
                ranks[d, valid] = pd.Series(day_vals[valid]).rank(pct=True).values
        cs_blocks.append(ranks[:, :, np.newaxis])
        new_features.append(f"cs_rank_{f}")

    # Pairwise rank products (top interactions from feature importance analysis)
    # RSI × Volume Ratio, SMA_5 × RSI, Bollinger × RSI
    # These capture regime-specific alpha
    for (a, b, name) in [
        (4, 5, "rsi_x_volratio"),     # rsi_14 × volume_ratio
        (2, 4, "sma5_x_rsi"),         # sma_5 dev × rsi_14
        (6, 4, "bollinger_x_rsi"),    # bollinger_position × rsi_14
    ]:
        if a < n_feat and b < n_feat:
            inter = X[:, :, a] * X[:, :, b]
            cs_blocks.append(inter[:, :, np.newaxis])
            new_features.append(name)

    X_aug = np.concatenate([X] + cs_blocks, axis=2)
    return X_aug, new_features


def add_momentum_rank_features(prices: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Compute multi-period momentum percentile ranks.

    prices: (n_days, n_stocks) close prices.

    Returns (mom_features, feature_names) where mom_features is (n_days, n_stocks, n_periods).
    """
    n_days, n_stocks = prices.shape
    periods = [5, 21, 63, 126]
    mom_feats = np.zeros((n_days, n_stocks, len(periods)), dtype=np.float32)
    names = []

    for pi, period in enumerate(periods):
        # Momentum: price[t] / price[t-period] - 1
        mom = np.zeros((n_days, n_stocks), dtype=np.float32)
        mom[period:] = prices[period:] / prices[:-period] - 1.0

        # Cross-sectional percentile rank per day
        for d in range(period, n_days):
            day_mom = mom[d]
            valid = ~np.isnan(day_mom)
            if valid.sum() > 1:
                mom[d, valid] = pd.Series(day_mom[valid]).rank(pct=True).values
        mom_feats[:, :, pi] = mom
        names.append(f"mom_rank_{period}d")

    return mom_feats, names


def add_illiquidity_feature(
    prices: np.ndarray, volumes: np.ndarray
) -> Tuple[np.ndarray, str]:
    """
    Amihud (2002) illiquidity: |return| / (price × volume).

    Higher values = more illiquid (larger price impact per dollar traded).

    Returns (illiq_feature, name) where illiq_feature is (n_days, n_stocks).
    """
    n_days, n_stocks = prices.shape
    rets = np.zeros_like(prices)
    rets[1:] = prices[1:] / prices[:-1] - 1.0

    dollar_vol = prices * np.maximum(volumes, 1.0)
    illiq = np.abs(rets) / np.maximum(dollar_vol, 1e-12)

    # 21-day rolling median (robust to outliers)
    illiq_ma = np.full_like(illiq, np.nan)
    for d in range(21, n_days):
        illiq_ma[d] = np.nanmedian(illiq[d-20:d+1], axis=0)

    # Cross-sectional rank
    illiq_rank = np.zeros_like(illiq_ma)
    for d in range(21, n_days):
        valid = ~np.isnan(illiq_ma[d])
        if valid.sum() > 1:
            illiq_rank[d, valid] = pd.Series(illiq_ma[d][valid]).rank(pct=True).values

    return illiq_rank[:, :, np.newaxis], "amihud_illiq"


# ═══════════════════════════════════════════════════════════
# P0: Multi-Horizon Target Computation
# ═══════════════════════════════════════════════════════════
def compute_multi_horizon_targets(
    prices: np.ndarray, horizons: List[int]
) -> Dict[int, np.ndarray]:
    """
    Compute forward returns for multiple horizons.

    prices: (n_days, n_stocks) close prices.
    horizons: list of forward periods [1, 5, 21].

    Returns dict {horizon: (n_days, n_stocks) forward returns}.
    Targets are NaN for the last `horizon` days.
    """
    n_days, n_stocks = prices.shape
    targets = {}
    for h in horizons:
        rets = np.full((n_days, n_stocks), np.nan, dtype=np.float32)
        rets[:-h] = prices[h:] / prices[:-h] - 1.0
        targets[h] = rets
    return targets


def compute_vol_adjusted_returns(
    prices: np.ndarray, horizon: int = 5, vol_period: int = 21
) -> np.ndarray:
    """
    Volatility-adjusted forward returns: forward_return / realized_vol.

    This penalizes predictions in high-volatility periods,
    aligning with risk-adjusted portfolio construction.
    """
    n_days, n_stocks = prices.shape
    rets = np.full((n_days, n_stocks), np.nan, dtype=np.float32)
    rets[:-horizon] = prices[horizon:] / prices[:-horizon] - 1.0

    # Realized vol
    simple_rets = np.zeros_like(prices)
    simple_rets[1:] = prices[1:] / prices[:-1] - 1.0
    rv = np.full_like(prices, np.nan)
    for d in range(vol_period, n_days):
        rv[d] = np.nanstd(simple_rets[d-vol_period+1:d+1], axis=0)

    vol_adj = rets / np.maximum(rv, 1e-6)
    return vol_adj


# ═══════════════════════════════════════════════════════════
# Walk-Forward Engine
# ═══════════════════════════════════════════════════════════
def generate_windows(n_total: int, train: int, purge: int, val: int, step: int):
    windows = []
    start = 0
    while start + train + purge + val <= n_total:
        windows.append((start, start + train, start + train + purge, start + train + purge + val))
        start += step
    return windows


def walk_forward_eval(
    X: np.ndarray, y: np.ndarray, n_stocks: int,
    model_type: str = "lgb", seed: int = 42,
) -> dict:
    """
    Walk-forward evaluation for a single (X, y) pair.

    Returns dict with per-fold and aggregate metrics.
    """
    n_days = X.shape[0]
    n_feat = X.shape[2]
    windows = generate_windows(n_days, WF_TRAIN, WF_PURGE, WF_VAL, WF_STEP)
    logger.info(f"  WF: {len(windows)} windows | {n_days} days × {n_stocks} stocks × {n_feat} feats | model={model_type}")

    X_flat = X.reshape(-1, n_feat)
    y_flat = y.reshape(-1)
    folds = []

    for wi, (t0, t1, t2, t3) in enumerate(windows):
        train_idx = np.arange(t0 * n_stocks, t1 * n_stocks)
        val_idx = np.arange(t2 * n_stocks, t3 * n_stocks)

        X_tr = X_flat[train_idx]
        y_tr = y_flat[train_idx]
        X_vl = X_flat[val_idx]
        y_vl = y_flat[val_idx]

        mask_tr = ~np.isnan(y_tr)
        mask_vl = ~np.isnan(y_vl)
        if np.any(np.isnan(X_tr)) or np.any(np.isinf(X_tr)):
            mask_tr &= ~np.isnan(X_tr).any(axis=1) & ~np.isinf(X_tr).any(axis=1)
        if np.any(np.isnan(X_vl)) or np.any(np.isinf(X_vl)):
            mask_vl &= ~np.isnan(X_vl).any(axis=1) & ~np.isinf(X_vl).any(axis=1)

        if mask_tr.sum() < 200 or mask_vl.sum() < 50:
            continue

        if model_type == "lgb" and HAS_LGB:
            model = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
            model.fit(X_tr[mask_tr], y_tr[mask_tr])
            y_pred = model.predict(X_vl[mask_vl])
        elif model_type == "ridge":
            model = Ridge(alpha=1.0, random_state=seed)
            model.fit(X_tr[mask_tr], y_tr[mask_tr])
            y_pred = model.predict(X_vl[mask_vl])
        else:
            continue

        fold_ic = ic_score(y_vl[mask_vl], y_pred)
        fold_dir = dir_accuracy(y_vl[mask_vl], y_pred)

        # Strategy Sharpe: long top 20%, short bottom 20%
        n_v = len(y_pred)
        cutoff = max(int(n_v * 0.2), 1)
        order = np.argsort(y_pred)
        y_true_sorted = y_vl[mask_vl]
        strat_ret = np.concatenate([
            -y_true_sorted[order[:cutoff]],
            y_true_sorted[order[-cutoff:]],
        ])
        fold_sharpe = sharpe(strat_ret) if len(strat_ret) > 1 else 0.0

        folds.append({
            "window": wi, "ic": float(fold_ic), "dir_acc": float(fold_dir),
            "sharpe": float(fold_sharpe),
            "train_n": int(mask_tr.sum()), "val_n": int(mask_vl.sum()),
        })

    if not folds:
        return {"n_windows": 0, "ic_mean": 0, "ic_std": 0, "sharpe_mean": 0, "dir_acc_mean": 0, "folds": []}

    ics = [f["ic"] for f in folds]
    sharpes = [f["sharpe"] for f in folds]
    dirs = [f["dir_acc"] for f in folds]

    return {
        "n_windows": len(folds),
        "ic_mean": float(np.mean(ics)),
        "ic_std": float(np.std(ics)),
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes)),
        "dir_acc_mean": float(np.mean(dirs)),
        "folds": folds,
    }


# ═══════════════════════════════════════════════════════════
# P2: Regime-Adaptive & Portfolio Strategies
# ═══════════════════════════════════════════════════════════
class RegimeDetector:
    """Detect market regime from daily returns using volatility and trend signals."""

    def __init__(self, vol_lookback: int = 63, trend_lookback: int = 126):
        self.vol_lb = vol_lookback
        self.trend_lb = trend_lookback

    def classify(self, returns: np.ndarray, day: int) -> str:
        """
        Classify regime at given day.

        Returns 'high_vol', 'low_vol_trending', or 'low_vol_mean_reverting'.
        """
        if day < self.trend_lb:
            return "normal"

        # Recent volatility
        recent_vol = np.nanstd(returns[day-self.vol_lb:day])
        long_vol = np.nanstd(returns[day-self.trend_lb:day])
        vol_ratio = recent_vol / max(long_vol, 1e-6)

        # Trend: SMA(63) vs SMA(126)
        sma_short = np.nanmean(returns[day-63:day])
        sma_long = np.nanmean(returns[day-self.trend_lb:day])
        trend = sma_short - sma_long

        if vol_ratio > 1.3:
            return "high_vol"
        elif trend > 0:
            return "low_vol_trending"
        else:
            return "low_vol_mean_reverting"


def regime_adaptive_backtest(
    prices: np.ndarray, predictions: np.ndarray,
    rebalance_freq: int = 5, top_k: int = 100,
) -> dict:
    """
    P2: Regime-adaptive portfolio backtest.

    - Select top-K stocks by prediction signal
    - Weight by inverse volatility in high-vol regime
    - Equal weight in low-vol trending regime
    - Reduce exposure in mean-reverting regime

    Returns backtest metrics dict.
    """
    n_days, n_stocks = prices.shape
    daily_rets = np.zeros_like(prices)
    daily_rets[1:] = prices[1:] / prices[:-1] - 1.0

    # Market return (equal-weight all stocks)
    market_rets = np.nanmean(daily_rets, axis=1)
    detector = RegimeDetector()

    eq_weights = np.full(n_stocks, 1.0 / n_stocks)

    # Strategy tracking
    bench_equity = np.ones(n_days)
    strat_equity = np.ones(n_days)
    regime_history = []
    current_weights = eq_weights.copy()

    for d in range(1, n_days):
        # Benchmark: equal weight
        bench_equity[d] = bench_equity[d-1] * (1 + np.dot(eq_weights, daily_rets[d]))

        # Rebalance
        if d % rebalance_freq == 0 and d > 126:
            regime = detector.classify(market_rets, d)
            regime_history.append({"day": d, "regime": regime})

            # Select top-K by prediction
            pred_today = predictions[d-1]  # use yesterday's prediction
            valid_pred = ~np.isnan(pred_today)
            if valid_pred.sum() >= top_k:
                top_idx = np.argsort(pred_today)[-top_k:]
                weights_raw = np.zeros(n_stocks)
                weights_raw[top_idx] = 1.0 / top_k
            else:
                weights_raw = eq_weights.copy()

            # Regime-adaptive scaling
            if regime == "high_vol":
                # Inverse volatility weighting within top-K
                lookback_rets = daily_rets[max(0,d-63):d]
                hist_vol = np.nanstd(lookback_rets, axis=0)
                hist_vol = np.maximum(hist_vol, 1e-6)
                inv_vol = np.where(weights_raw > 0, 1.0/hist_vol, 0)
                weights_raw = inv_vol / max(inv_vol.sum(), 1e-10)
            elif regime == "low_vol_mean_reverting":
                # Reduce exposure by 50%
                weights_raw *= 0.5
            # low_vol_trending: keep as-is

            current_weights = weights_raw

        strat_equity[d] = strat_equity[d-1] * (1 + np.dot(current_weights, daily_rets[d]))

    # Metrics
    strat_rets = np.diff(strat_equity) / strat_equity[:-1]
    bench_rets = np.diff(bench_equity) / bench_equity[:-1]

    strat_sharpe = sharpe(strat_rets)
    bench_sharpe = sharpe(bench_rets)
    strat_mdd, _ = max_drawdown(strat_equity)
    bench_mdd, _ = max_drawdown(bench_equity)
    strat_ann = np.mean(strat_rets) * 252
    bench_ann = np.mean(bench_rets) * 252

    # Regime-specific performance
    regime_perf = {}
    regimes_seen = set(r["regime"] for r in regime_history)
    for reg in regimes_seen:
        reg_days = [r["day"] for r in regime_history if r["regime"] == reg]
        if reg_days:
            reg_rets = []
            for rd in reg_days:
                end = min(rd + rebalance_freq, n_days)
                if end > rd + 1:
                    r_seg = strat_rets[rd:end-1]
                    reg_rets.extend(r_seg)
            if reg_rets:
                reg_rets = np.array(reg_rets)
                regime_perf[reg] = {
                    "sharpe": sharpe(reg_rets),
                    "n_periods": len(reg_rets),
                    "mean_return": float(np.mean(reg_rets) * 252),
                }

    return {
        "strategy_sharpe": float(strat_sharpe),
        "benchmark_sharpe": float(bench_sharpe),
        "sharpe_delta": float(strat_sharpe - bench_sharpe),
        "strategy_return": float(strat_equity[-1] - 1),
        "benchmark_return": float(bench_equity[-1] - 1),
        "strategy_annual_return": float(strat_ann),
        "benchmark_annual_return": float(bench_ann),
        "strategy_max_dd": float(strat_mdd),
        "regime_performance": regime_perf,
        "regime_counts": {r: sum(1 for rh in regime_history if rh["regime"] == r) for r in regimes_seen},
    }


# ═══════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Glaubenskrieg Pipeline v2")
    parser.add_argument("--data-dir", default="/root/data/us_stocks_full")
    parser.add_argument("--output", default="/root/results/pipeline_v2.json")
    parser.add_argument("--equity", default="/root/results/pipeline_v2_equity.csv")
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-enhanced", action="store_true")
    args = parser.parse_args()

    _t_start = time.time()
    results = {"config": vars(args), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ═══════════════════════════════════════════════════
    # 1. Load & Filter
    # ═══════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 1: Load & Filter Data")
    logger.info("=" * 60)

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.csv"))
    stocks = {}
    for fp in files:
        sym = fp.stem
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if len(df) >= args.min_rows:
            stocks[sym] = df
    logger.info(f"Loaded {len(stocks)} stocks with ≥{args.min_rows} rows (from {len(files)} total)")

    # ═══════════════════════════════════════════════════
    # 2. Compute Base Features
    # ═══════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 2: Compute Base OHLCV Features")
    logger.info("=" * 60)

    feats = {}
    for i, (sym, df) in enumerate(stocks.items()):
        feats[sym] = compute_all_features(df)
        if (i + 1) % 100 == 0:
            logger.info(f"  Features: {i+1}/{len(stocks)} stocks")

    # Align dates
    common = None
    for f in feats.values():
        common = f.index if common is None else common.intersection(f.index)

    symbols = sorted(feats.keys())
    n_stocks = len(symbols)
    n_days = len(common)
    base_feature_cols = list(feats[symbols[0]].columns)
    logger.info(f"Base features: {len(base_feature_cols)} cols: {base_feature_cols}")

    # Build base feature array
    X_base = np.zeros((n_days, n_stocks, len(base_feature_cols)), dtype=np.float32)
    for j, sym in enumerate(symbols):
        X_base[:, j, :] = feats[sym].loc[common].values.astype(np.float32)

    # Extract prices for target computation
    prices = np.zeros((n_days, n_stocks), dtype=np.float32)
    volumes = np.zeros((n_days, n_stocks), dtype=np.float32)
    for j, sym in enumerate(symbols):
        prices[:, j] = stocks[sym].loc[common, 'close'].values.astype(np.float32)
        volumes[:, j] = stocks[sym].loc[common, 'volume'].values.astype(np.float32)

    results["data"] = {
        "n_stocks": n_stocks, "n_days": n_days,
        "date_range": [str(common[0].date()), str(common[-1].date())],
        "base_feature_cols": base_feature_cols,
    }
    logger.info(f"Data shape: {n_days} days × {n_stocks} stocks × {len(base_feature_cols)} features")
    logger.info(f"Date range: {common[0].date()} ~ {common[-1].date()}")

    # ═══════════════════════════════════════════════════
    # 3. P1: Enhanced Features
    # ═══════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 3: P1 — Enhanced Features")
    logger.info("=" * 60)

    X = X_base.copy()
    all_feature_cols = list(base_feature_cols)

    if not args.skip_enhanced:
        # Cross-sectional features
        X_cs, cs_names = add_cross_sectional_features(X_base)
        X = np.concatenate([X, X_cs[:, :, len(base_feature_cols):]], axis=2)
        all_feature_cols.extend(cs_names)
        logger.info(f"  +{len(cs_names)} cross-sectional features (total: {len(all_feature_cols)})")

        # Momentum rank features
        X_mom, mom_names = add_momentum_rank_features(prices)
        X = np.concatenate([X, X_mom], axis=2)
        all_feature_cols.extend(mom_names)
        logger.info(f"  +{len(mom_names)} momentum rank features (total: {len(all_feature_cols)})")

        # Amihud illiquidity
        X_illiq, illiq_name = add_illiquidity_feature(prices, volumes)
        X = np.concatenate([X, X_illiq], axis=2)
        all_feature_cols.append(illiq_name)
        logger.info(f"  +1 illiquidity feature (total: {len(all_feature_cols)})")
    else:
        logger.info("  Skipped (--skip-enhanced)")

    results["features"] = {
        "n_total": X.shape[2],
        "n_base": len(base_feature_cols),
        "n_enhanced": X.shape[2] - len(base_feature_cols),
        "all_columns": all_feature_cols,
    }

    # ═══════════════════════════════════════════════════
    # 4. P0: Multi-Horizon Targets
    # ═══════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 4: P0 — Multi-Horizon Targets")
    logger.info("=" * 60)

    targets = compute_multi_horizon_targets(prices, HORIZONS)
    # Add volatility-adjusted return (5d horizon)
    targets_voladj = compute_vol_adjusted_returns(prices, horizon=5)

    logger.info(f"  Targets computed for horizons: {list(HORIZONS)} + vol_adj_5d")

    # ═══════════════════════════════════════════════════
    # 5. Walk-Forward Evaluation (per horizon × model)
    # ═══════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 5: Walk-Forward Evaluation")
    logger.info("=" * 60)

    horizon_results = {}

    # Standard horizons
    for h in HORIZONS:
        label = HORIZON_LABELS[h]
        logger.info(f"\n--- Horizon: {label} ({h}d forward) ---")

        h_res = {}
        for model_type in ["lgb", "ridge"]:
            wf = walk_forward_eval(X, targets[h], n_stocks, model_type=model_type, seed=args.seed)
            h_res[model_type] = wf
            logger.info(f"  {model_type:6s}: IC={wf['ic_mean']:.4f}±{wf['ic_std']:.4f}  Sharpe={wf['sharpe_mean']:.2f}  DirAcc={wf['dir_acc_mean']:.3f}  ({wf['n_windows']} windows)")
        horizon_results[label] = h_res

    # Volatility-adjusted target
    logger.info(f"\n--- Horizon: vol_adj_5d ---")
    voladj_res = {}
    for model_type in ["lgb", "ridge"]:
        wf = walk_forward_eval(X, targets_voladj, n_stocks, model_type=model_type, seed=args.seed)
        voladj_res[model_type] = wf
        logger.info(f"  {model_type:6s}: IC={wf['ic_mean']:.4f}±{wf['ic_std']:.4f}  Sharpe={wf['sharpe_mean']:.2f}  DirAcc={wf['dir_acc_mean']:.3f}  ({wf['n_windows']} windows)")
    horizon_results["vol_adj_5d"] = voladj_res

    results["horizons"] = horizon_results

    # ═══════════════════════════════════════════════════
    # 6. P2: Regime-Adaptive Backtest
    # ═══════════════════════════════════════════════════
    if not args.skip_backtest:
        logger.info("=" * 60)
        logger.info("STEP 6: P2 — Regime-Adaptive Backtest")
        logger.info("=" * 60)

        # Use LightGBM 5d predictions (best horizon typically)
        best_label = "5d"
        best_h = 5

        # Retrain on all data up to midpoint, generate predictions for second half
        mid_day = n_days // 2
        train_days = mid_day
        test_days = n_days - mid_day

        logger.info(f"  Training on days 0-{train_days}, testing on {train_days}-{n_days}")

        # Generate OOS predictions (walk-forward style for second half)
        oos_preds = np.full((n_days, n_stocks), np.nan, dtype=np.float32)
        X_flat = X.reshape(-1, X.shape[2])
        y_flat = targets[best_h].reshape(-1)

        # Walk-forward over second half
        bt_windows = generate_windows(n_days, WF_TRAIN, WF_PURGE, WF_VAL, WF_STEP)
        for wi, (t0, t1, t2, t3) in enumerate(bt_windows):
            if t0 < train_days:  # Only use windows where validation is in test period
                train_idx = np.arange(t0 * n_stocks, t1 * n_stocks)
                val_idx = np.arange(t2 * n_stocks, t3 * n_stocks)

                X_tr = X_flat[train_idx]
                y_tr = y_flat[train_idx]
                X_vl = X_flat[val_idx]
                y_vl = y_flat[val_idx]

                mask_tr = ~np.isnan(y_tr)
                if mask_tr.sum() < 200:
                    continue
                if np.any(np.isnan(X_tr)) or np.any(np.isinf(X_tr)):
                    mask_tr &= ~np.isnan(X_tr).any(axis=1)

                if HAS_LGB:
                    model = lgb.LGBMRegressor(random_state=args.seed, **LGB_PARAMS)
                    model.fit(X_tr[mask_tr], y_tr[mask_tr])
                    y_pred = model.predict(X_vl)

                    # Place predictions at validation days
                    for d in range(t2, t3):
                        day_offset = d - t2
                        stock_start = day_offset * n_stocks
                        stock_end = stock_start + n_stocks
                        if stock_end <= len(y_pred):
                            oos_preds[d] = y_pred[stock_start:stock_end]

        logger.info(f"  OOS predictions generated for {np.sum(~np.isnan(oos_preds[:,0]))} days")

        # Regime-adaptive backtest
        bt_result = regime_adaptive_backtest(prices, oos_preds, rebalance_freq=REBALANCE_FREQ, top_k=TOP_K)
        results["backtest"] = bt_result

        logger.info(f"  Strategy Sharpe:  {bt_result['strategy_sharpe']:.4f}")
        logger.info(f"  Benchmark Sharpe: {bt_result['benchmark_sharpe']:.4f}")
        logger.info(f"  Δ Sharpe:         {bt_result['sharpe_delta']:+.4f}")
        logger.info(f"  Regime counts:    {bt_result['regime_counts']}")

        # Save equity curve (simplified)
        eq_df = pd.DataFrame({
            "date": common,
            "strategy_equity": np.ones(n_days),  # placeholder, actual equity in bt_result
        })
        # Don't save large equity arrays in JSON, keep in CSV separately
        eq_df.to_csv(args.equity, index=False)
        logger.info(f"  Equity placeholder saved to {args.equity}")

    # ═══════════════════════════════════════════════════
    # 7. Summary & Save
    # ═══════════════════════════════════════════════════
    elapsed = time.time() - _t_start
    results["runtime_s"] = round(elapsed, 1)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Summary Table ──
    print("\n" + "=" * 80)
    print("  GLAUBENSKRIEG PIPELINE v2 — RESULTS")
    print("=" * 80)
    print(f"  Data: {n_stocks} stocks × {n_days} days ({common[0].date()} ~ {common[-1].date()})")
    print(f"  Features: {X.shape[2]} ({len(base_feature_cols)} base + {X.shape[2]-len(base_feature_cols)} enhanced)")
    print(f"  Walk-Forward: {WF_TRAIN}/{WF_PURGE}/{WF_VAL}/{WF_STEP} (train/purge/val/step)")
    print()

    print(f"  {'Horizon':<14} {'Model':<8} {'IC':>8} {'±':>8} {'Sharpe':>8} {'DirAcc':>7} {'Wins':>5}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*5}")
    for label, h_res in horizon_results.items():
        for model_type, wf in h_res.items():
            print(f"  {label:<14} {model_type:<8} {wf['ic_mean']:>8.4f} {wf['ic_std']:>8.4f} {wf['sharpe_mean']:>8.2f} {wf['dir_acc_mean']:>7.3f} {wf['n_windows']:>5}")

    if "backtest" in results:
        bt = results["backtest"]
        print(f"\n  BACKTEST (regime-adaptive, top {TOP_K}, rebalance {REBALANCE_FREQ}d):")
        print(f"    Strategy Sharpe:  {bt['strategy_sharpe']:.4f}")
        print(f"    Benchmark Sharpe: {bt['benchmark_sharpe']:.4f}")
        print(f"    Δ Sharpe:         {bt['sharpe_delta']:+.4f}")
        print(f"    Regime counts:    {bt['regime_counts']}")

    print(f"\n  Runtime: {elapsed:.0f}s")
    print(f"  Output:  {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
