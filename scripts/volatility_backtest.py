#!/usr/bin/env python3
"""
Volatility Trading Backtest System for Glaubenskrieg.

Walk-forward LightGBM volatility prediction → weekly rebalancing with
inverse-volatility position sizing on A-share stocks.

Strategy:
  1. Predict forward 21-day realized vol for each stock using LightGBM
  2. Weekly rebalancing: select top-N by liquidity, weight ∝ 1/σ_pred
  3. Compare to equal-weight benchmark + vol-targeted variant

Walk-forward protocol (purged):
  train=1000d | purge=126d | val=200d | test=126d

Usage:
    python scripts/volatility_backtest.py
    python scripts/volatility_backtest.py --n-stocks 50 --top-k 20
    python scripts/volatility_backtest.py --seed 123 --output results/custom.json
"""

import argparse, json, logging, os, sys, time, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.features import compute_all_features

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

TARGET_HORIZON  = 21       # forward volatility horizon (trading days)
TRAIN_DAYS      = 1000     # walk-forward training window
PURGE_DAYS      = 126      # purge gap between train and val
VAL_DAYS        = 200      # validation window for early stopping
STEP_DAYS       = 126      # test window / step size
REBALANCE_FREQ  = 5        # rebalance every N trading days
TARGET_ANN_VOL  = 0.15     # target annualized vol for vol-scaling
LEVERAGE_CAP    = 2.0      # max leverage for vol-targeting
LEVERAGE_FLOOR  = 0.25     # min leverage for vol-targeting
VOL_LOOKBACK    = 63       # trailing days for realized vol estimation


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading & feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

def compute_forward_realized_vol(
    df: pd.DataFrame, horizon: int = TARGET_HORIZON
) -> pd.Series:
    """Compute forward-looking realized volatility.

    At day t, forward_vol[t] = std(daily_returns[t+1 : t+1+horizon]).
    The last ``horizon`` entries are NaN (not enough future data).

    Parameters
    ----------
    df : DataFrame with 'close' column.
    horizon : number of trading days forward.

    Returns
    -------
    Series indexed like df.
    """
    daily_ret = df["close"].pct_change()
    forward_vol = daily_ret.rolling(horizon).std().shift(-horizon)
    forward_vol.name = "forward_vol_21"
    return forward_vol


def load_and_prepare_data(
    data_dir: str, n_stocks: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           pd.DatetimeIndex, List[str], List[str]]:
    """Load CSVs, compute features/forward vol/liquidity/close, align by date UNION.

    Uses the full date range (union of all stocks) with NaN filling for stocks
    that don't have data on certain dates.  This preserves the maximum time
    series length rather than truncating to a narrow intersection.

    Returns
    -------
    features   : (T, N, D) float32 — feature matrix (NaN where stock has no data).
    targets    : (T, N) float32   — forward 21-day realized vol.
    liquidity  : (T, N) float32   — 20-day avg volume.
    close_prices: (T, N) float32  — daily close prices.
    dates      : DatetimeIndex of length T.
    stocks     : list of stock codes (length N).
    feat_names : list of feature column names (length D).
    """
    csv_files = sorted(Path(data_dir).glob("*.csv"))[:n_stocks]
    logger.info("Loading %d stocks from %s", len(csv_files), data_dir)

    min_required = TRAIN_DAYS + PURGE_DAYS + VAL_DAYS + STEP_DAYS

    feature_dfs: Dict[str, pd.DataFrame] = {}
    target_series: Dict[str, pd.Series] = {}
    liquidity_series: Dict[str, pd.Series] = {}
    close_series: Dict[str, pd.Series] = {}

    for fp in csv_files:
        code = fp.stem
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if df.empty or "close" not in df.columns or "volume" not in df.columns:
            logger.warning("Skipping %s: invalid data", code)
            continue

        # Causal features from OHLCV
        feats = compute_all_features(df)
        # Forward vol target (causal features → forward-looking target avoids leakage)
        fwd_vol = compute_forward_realized_vol(df, TARGET_HORIZON)
        # Liquidity proxy
        liq = df["volume"].rolling(20).mean()

        # Intersect valid indices (causal features drop early NaN, fwd_vol drops late NaN)
        code_idx = feats.dropna().index.intersection(fwd_vol.dropna().index)
        code_idx = code_idx.intersection(liq.dropna().index)
        if len(code_idx) < min_required:
            logger.warning("Skipping %s: only %d valid rows (need >= %d)",
                           code, len(code_idx), min_required)
            continue

        feature_dfs[code] = feats.loc[code_idx]
        target_series[code] = fwd_vol.loc[code_idx]
        liquidity_series[code] = liq.loc[code_idx]
        close_series[code] = df["close"].loc[code_idx]

    stocks = list(feature_dfs.keys())
    N = len(stocks)
    if N == 0:
        raise RuntimeError("No valid stocks found")
    logger.info("%d stocks pass filters", N)

    # Build date UNION (not intersection) to maximise time series length.
    # Stocks without data on a given date get NaN — the walk-forward
    # training filters NaN rows per fold so this is safe.
    all_dates = feature_dfs[stocks[0]].index
    for s in stocks[1:]:
        all_dates = all_dates.union(feature_dfs[s].index)
    all_dates = all_dates.sort_values()
    T = len(all_dates)
    logger.info("Full date range (union): %s → %s (%d days)",
                all_dates[0].strftime("%Y-%m-%d"),
                all_dates[-1].strftime("%Y-%m-%d"), T)

    # Report coverage: for each date, how many stocks have data
    coverage = np.zeros(T, dtype=int)
    for s in stocks:
        mask = all_dates.isin(feature_dfs[s].index)
        coverage += mask.astype(int)
    logger.info("Coverage: min=%d stocks/date, median=%d, max=%d",
                int(coverage.min()), int(np.median(coverage)), int(coverage.max()))

    # Feature names
    all_cols = list(feature_dfs[stocks[0]].columns)
    feat_names = [c for c in all_cols if c != "forward_vol_21"]
    D = len(feat_names)

    # Build aligned arrays (NaN-filled, then overwrite where data exists)
    features = np.full((T, N, D), np.nan, dtype=np.float32)
    targets = np.full((T, N), np.nan, dtype=np.float32)
    liquidity = np.full((T, N), np.nan, dtype=np.float32)
    close_prices = np.full((T, N), np.nan, dtype=np.float32)

    for j, s in enumerate(stocks):
        fdf = feature_dfs[s]
        # Align this stock's index to the full date union
        stock_dates = fdf.index.intersection(all_dates)
        idx_map = all_dates.get_indexer(stock_dates)
        valid_idx = idx_map[idx_map >= 0]

        features[valid_idx, j, :] = fdf.loc[stock_dates, feat_names].values.astype(np.float32)
        targets[valid_idx, j] = target_series[s].loc[stock_dates].values.astype(np.float32)
        liquidity[valid_idx, j] = liquidity_series[s].loc[stock_dates].values.astype(np.float32)
        close_prices[valid_idx, j] = close_series[s].loc[stock_dates].values.astype(np.float32)

    logger.info("Feature matrix: %d dates × %d stocks × %d features", T, N, D)
    return features, targets, liquidity, close_prices, all_dates, stocks, feat_names


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward training
# ═══════════════════════════════════════════════════════════════════════════════

def build_lgb_model(seed: int = 42) -> lgb.LGBMRegressor:
    """Create a fast LightGBM regressor for volatility prediction.

    Model is deliberately lightweight to keep walk-forward training
    tractable with 200 stocks × 1000 training days per fold.
    """
    return lgb.LGBMRegressor(
        objective="regression",
        metric="l1",
        boosting_type="gbdt",
        num_leaves=15,
        max_depth=4,
        learning_rate=0.07,
        n_estimators=150,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=50,
        verbosity=-1,
        random_state=seed,
        n_jobs=-1,
        force_col_wise=True,   # faster for wide-ish data with many samples
    )


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE loss: log(h) + σ²/h  where h = y_pred², σ² = y_true². Lower is better."""
    h = np.maximum(y_pred ** 2, 1e-8)
    return float(np.mean(np.log(h) + y_true ** 2 / h))


def walk_forward_predict(
    features: np.ndarray,       # (T, N, D)
    targets: np.ndarray,        # (T, N)
    train_days: int = TRAIN_DAYS,
    purge_days: int = PURGE_DAYS,
    val_days: int = VAL_DAYS,
    step_days: int = STEP_DAYS,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Dict]]:
    """Purged walk-forward: train LGBMRegressor per fold, return out-of-sample predictions.

    Fold layout:
      train [pos, pos+train_days)
      purge [pos+train_days, pos+train_days+purge_days)  ← skipped (no leakage)
      val   [pos+train_days+purge_days, pos+train_days+purge_days+val_days)
      test  [pos+train_days+purge_days+val_days, pos+...)

    Returns
    -------
    predictions : (T, N) float32 — NaN outside test windows.
    fold_info   : list of per-fold metrics dicts.
    """
    T, N, D = features.shape
    total_window = train_days + purge_days + val_days + step_days

    predictions = np.full((T, N), np.nan, dtype=np.float32)
    fold_info: List[Dict] = []

    n_folds = max(0, (T - total_window) // step_days + 1)
    logger.info("Walk-forward: up to %d folds (total_window=%d days)", n_folds, total_window)

    for fold in range(n_folds):
        pos = fold * step_days
        tr_end = pos + train_days
        vs = tr_end + purge_days          # validation start
        ve = vs + val_days                # validation end = test start
        te_end = min(ve + step_days, T)

        if ve >= T:
            break

        # Gather training data (flatten across stocks)
        tr_f = features[pos:tr_end].reshape(-1, D)
        tr_t = targets[pos:tr_end].reshape(-1)
        v_f  = features[vs:ve].reshape(-1, D)
        v_t  = targets[vs:ve].reshape(-1)

        tr_valid = np.isfinite(tr_f).all(axis=1) & np.isfinite(tr_t)
        vl_valid = np.isfinite(v_f).all(axis=1) & np.isfinite(v_t)
        X_tr, y_tr = tr_f[tr_valid], tr_t[tr_valid]
        X_vl, y_vl = v_f[vl_valid], v_t[vl_valid]

        if len(X_tr) < 200 or len(X_vl) < 50:
            logger.warning("Fold %d: insufficient data (train=%d, val=%d)",
                           fold, len(X_tr), len(X_vl))
            continue

        # Train
        model = build_lgb_model(seed)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(10, verbose=False), lgb.log_evaluation(0)],
        )

        # Predict on test window
        te_f = features[ve:te_end].reshape(-1, D)
        te_t = targets[ve:te_end].reshape(-1)
        te_valid = np.isfinite(te_f).all(axis=1)
        X_te = te_f[te_valid]

        if len(X_te) == 0:
            logger.warning("Fold %d: no valid test samples", fold)
            continue

        yp_all = model.predict(X_te)
        te_pred = np.full(te_f.shape[0], np.nan, dtype=np.float32)
        te_pred[te_valid] = yp_all
        predictions[ve:te_end] = te_pred.reshape(te_end - ve, N)

        fold_info.append({
            "fold": fold,
            "train_dates": [int(pos), int(tr_end)],
            "test_dates": [int(ve), int(te_end)],
            "n_train": int(tr_valid.sum()),
            "n_val": int(vl_valid.sum()),
            "n_test": int(te_valid.sum()),
            "mse": float(np.mean((te_t[te_valid] - yp_all) ** 2)),
            "mae": float(np.mean(np.abs(te_t[te_valid] - yp_all))),
            "qlike": qlike(te_t[te_valid], yp_all),
        })
        logger.info("  Fold %2d: train [%d,%d) test [%d,%d) | "
                    "MSE=%.6f MAE=%.6f QLIKE=%.4f | samples=%d",
                    fold, pos, tr_end, ve, te_end,
                    fold_info[-1]["mse"], fold_info[-1]["mae"],
                    fold_info[-1]["qlike"], fold_info[-1]["n_test"])

    if fold_info:
        qs = [f["qlike"] for f in fold_info]
        ms = [f["mse"] for f in fold_info]
        ma = [f["mae"] for f in fold_info]
        logger.info("Walk-forward: %d folds | QLIKE=%.4f±%.4f MSE=%.6f MAE=%.6f",
                    len(fold_info), np.mean(qs), np.std(qs),
                    np.mean(ms), np.mean(ma))

    return predictions, fold_info


# ═══════════════════════════════════════════════════════════════════════════════
# Trading simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_strategy(
    predictions: np.ndarray,       # (T, N) predicted forward vol (NaN where unpredicted)
    close_prices: np.ndarray,      # (T, N)
    liquidity: np.ndarray,         # (T, N) 20-day avg volume
    dates: pd.DatetimeIndex,       # (T,)
    stocks: List[str],             # (N,)
    top_k: int = 50,
    rebalance_freq: int = REBALANCE_FREQ,
    transaction_cost: float = 0.0,
    oos_start_idx: int = 0,        # first OOS date index (0-based into dates)
) -> Dict:
    """Simulate weekly rebalancing with inverse-volatility weighting.

    At each rebalance:
      1. Select top_k stocks by trailing liquidity.
      2. Weight each stock ∝ 1 / predicted_vol (equal-weight fallback if no pred).
      3. Hold until next rebalance.

    Also tracks:
      - Equal-weight benchmark (top_k by liquidity, equal weights).
      - Vol-targeted variant (benchmark × leverage to target annualized vol).

    Parameters
    ----------
    oos_start_idx : first date index where trading strategy begins.
                    Before this, strategy = benchmark (no predictions yet).

    Returns
    -------
    dict with equity_curve (DataFrame), turnover (array), oos_start_idx.
    """
    T, N = predictions.shape
    if T < rebalance_freq + 1:
        raise ValueError(f"Need at least {rebalance_freq + 1} dates, got {T}")

    # Daily forward returns: returns[t] = close[t+1]/close[t] - 1
    # So daily_rets[t] is earned from date[t] to date[t+1]
    daily_rets = close_prices[1:] / close_prices[:-1] - 1.0  # (T-1, N)

    # Rebalance schedule: start from day 0, use equal-weight before OOS
    rebalance_dates_idx = list(range(0, T - 1, rebalance_freq))
    if not rebalance_dates_idx:
        rebalance_dates_idx = [0]
    # Always extend to cover the last trading day
    if rebalance_dates_idx[-1] < T - 1:
        rebalance_dates_idx.append(T - 1)

    # Mark start of OOS period (first index that has predictions)
    first_pred_date = None
    for t in range(T):
        if np.any(np.isfinite(predictions[t])):
            first_pred_date = t
            break
    effective_oos = first_pred_date if first_pred_date is not None else oos_start_idx
    logger.info("First prediction date: index %d", effective_oos)

    n_reb = len(rebalance_dates_idx) - 1

    # Track equity values (len T-1, aligned to daily_rets)
    strat_value = np.ones(T - 1)
    bench_value = np.ones(T - 1)
    voltarget_value = np.ones(T - 1)
    turnover_arr = np.zeros(n_reb)

    current_weights = np.zeros(N)
    bench_weights = np.zeros(N)
    vol_tracker: List[float] = []  # recent benchmark returns for vol estimation

    for i in range(n_reb):
        reb_day = rebalance_dates_idx[i]
        next_reb = rebalance_dates_idx[i + 1]

        # Select top_k by liquidity
        liq = liquidity[reb_day].copy()
        liq[np.isnan(liq)] = 0.0
        top_idx = np.argsort(liq)[::-1][:top_k]
        top_idx = top_idx[liq[top_idx] > 0]
        if len(top_idx) == 0:
            continue

        # Strategy weights: inverse-vol if predictions available, else equal-weight
        pred_vol = predictions[reb_day].copy()
        pred_vol = np.abs(pred_vol)
        pred_vol[np.isnan(pred_vol) | (pred_vol < 1e-8)] = np.nan

        has_predictions = reb_day >= effective_oos and np.any(np.isfinite(pred_vol[top_idx]))

        if has_predictions:
            inv_vol = 1.0 / pred_vol[top_idx]
            inv_vol[np.isnan(inv_vol)] = 0.0
            sum_inv = inv_vol.sum()
            if sum_inv > 1e-10:
                new_weights = np.zeros(N)
                new_weights[top_idx] = inv_vol / sum_inv
            else:
                new_weights = np.zeros(N)
                new_weights[top_idx] = 1.0 / len(top_idx)
        else:
            new_weights = np.zeros(N)
            new_weights[top_idx] = 1.0 / len(top_idx)

        # Benchmark: equal-weight
        new_bench = np.zeros(N)
        new_bench[top_idx] = 1.0 / len(top_idx)

        # Turnover (from previous weights)
        if i > 0:
            turnover_arr[i] = np.sum(np.abs(new_weights - current_weights)) / 2.0

        current_weights = new_weights
        bench_weights = new_bench

        # Simulate each day in the holding period
        for t in range(reb_day, min(next_reb, T - 1)):
            ret_t = daily_rets[t]  # (N,)

            # Strategy portfolio return
            valid = np.isfinite(ret_t) & (current_weights > 0)
            if valid.any():
                w = current_weights[valid] / current_weights[valid].sum()
                port_ret = float(np.dot(w, ret_t[valid]))
            else:
                port_ret = 0.0

            # Transaction cost on rebalance day
            if t == reb_day and i > 0 and not np.isnan(turnover_arr[i]):
                port_ret -= transaction_cost * turnover_arr[i]

            strat_value[t] = strat_value[max(0, t - 1)] * (1.0 + port_ret)

            # Benchmark return
            bvalid = np.isfinite(ret_t) & (bench_weights > 0)
            if bvalid.any():
                bw = bench_weights[bvalid] / bench_weights[bvalid].sum()
                bench_ret = float(np.dot(bw, ret_t[bvalid]))
            else:
                bench_ret = 0.0

            bench_value[t] = bench_value[max(0, t - 1)] * (1.0 + bench_ret)

            # Vol-targeting: scale benchmark return by target/realized leverage
            vol_tracker.append(bench_ret)
            if len(vol_tracker) > VOL_LOOKBACK:
                vol_tracker.pop(0)
            if len(vol_tracker) >= 21:
                realized_vol = float(np.std(vol_tracker)) * np.sqrt(252)
                leverage = float(np.clip(
                    TARGET_ANN_VOL / max(realized_vol, 1e-10),
                    LEVERAGE_FLOOR, LEVERAGE_CAP
                ))
                vt_ret = bench_ret * leverage
            else:
                vt_ret = bench_ret

            voltarget_value[t] = voltarget_value[max(0, t - 1)] * (1.0 + vt_ret)

    # Build equity curve DataFrame (aligned to daily_rets, length T-1)
    eq_dates = dates[1:]  # T-1 dates

    # Daily returns (percentage, length T-1, same as equity values)
    strat_rets = np.zeros(T - 1)
    bench_rets = np.zeros(T - 1)
    vt_rets = np.zeros(T - 1)
    strat_rets[0] = strat_value[0] - 1.0
    bench_rets[0] = bench_value[0] - 1.0
    vt_rets[0] = voltarget_value[0] - 1.0
    strat_rets[1:] = strat_value[1:] / strat_value[:-1] - 1.0
    bench_rets[1:] = bench_value[1:] / bench_value[:-1] - 1.0
    vt_rets[1:] = voltarget_value[1:] / voltarget_value[:-1] - 1.0

    equity_df = pd.DataFrame({
        "date": eq_dates,
        "strategy": strat_value,
        "benchmark": bench_value,
        "vol_target": voltarget_value,
        "strategy_return": strat_rets,
        "benchmark_return": bench_rets,
        "vol_target_return": vt_rets,
    }).set_index("date")

    return {
        "equity_curve": equity_df,
        "turnover": turnover_arr,
        "oos_start_idx": oos_start_idx,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Performance metrics
# ═══════════════════════════════════════════════════════════════════════════════

def annualized_sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily returns."""
    valid = returns[np.isfinite(returns)]
    if len(valid) < 20:
        return 0.0
    excess = valid - rf / 252
    std_excess = float(np.std(excess))
    if std_excess < 1e-10:
        return 0.0
    return float(np.mean(excess) / std_excess * np.sqrt(252))


def max_drawdown(equity: np.ndarray) -> Tuple[float, int, int]:
    """Maximum drawdown: peak-to-trough decline (positive value), with indices."""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-10)
    end = int(np.argmin(dd))
    if end == 0:
        return 0.0, 0, 0
    start = int(np.argmax(equity[:end + 1]))
    return float(np.abs(dd[end])), start, end


def calmar_ratio(annual_return: float, max_dd: float) -> float:
    """Calmar ratio = annual return / max drawdown."""
    if max_dd < 1e-8:
        return 0.0
    return annual_return / max_dd


def rolling_sharpe(returns: np.ndarray, window: int = 252) -> np.ndarray:
    """Rolling annualized Sharpe ratio (forward-filled for edges)."""
    n = len(returns)
    out = np.full(n, np.nan)
    for i in range(window, n + 1):
        out[i - 1] = annualized_sharpe(returns[i - window:i])
    return out


def compute_all_metrics(
    equity_df: pd.DataFrame,
    oos_start_date: pd.Timestamp = None,
) -> Dict:
    """Compute comprehensive performance metrics for all portfolios.

    If oos_start_date is provided, metrics are computed over the
    out-of-sample period only (from that date onward).
    """
    results = {}

    for col, ret_col in [("strategy", "strategy_return"),
                          ("benchmark", "benchmark_return"),
                          ("vol_target", "vol_target_return")]:
        if col not in equity_df.columns:
            continue

        df = equity_df.copy()
        if oos_start_date is not None:
            df = df[df.index >= oos_start_date]

        eq = df[col].values
        if len(eq) < 20:
            continue

        rets = df[ret_col].values if ret_col in df.columns else np.diff(eq) / eq[:-1]
        rets = np.asarray(rets, dtype=np.float64)
        rets = rets[np.isfinite(rets)]

        n_days = len(eq)
        ann_ret = float((eq[-1] / eq[0]) ** (252.0 / max(n_days, 1)) - 1.0)
        ann_vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 1 else 0.0
        sharpe = annualized_sharpe(rets)
        mdd, dd_start, dd_end = max_drawdown(eq)
        calm = calmar_ratio(ann_ret, mdd)
        total_ret = float(eq[-1] / eq[0] - 1.0)

        roll_sharpe = rolling_sharpe(rets, 252)
        rs_valid = roll_sharpe[np.isfinite(roll_sharpe)]

        results[col] = {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "max_dd_start_idx": int(dd_start),
            "max_dd_end_idx": int(dd_end),
            "calmar_ratio": calm,
            "n_days": n_days,
            "rolling_sharpe_mean": float(np.mean(rs_valid)) if len(rs_valid) > 0 else 0.0,
            "rolling_sharpe_std": float(np.std(rs_valid)) if len(rs_valid) > 0 else 0.0,
            "rolling_sharpe_min": float(np.min(rs_valid)) if len(rs_valid) > 0 else 0.0,
            "rolling_sharpe_max": float(np.max(rs_valid)) if len(rs_valid) > 0 else 0.0,
        }

    if "strategy" in results and "benchmark" in results:
        s, b = results["strategy"], results["benchmark"]
        results["comparison"] = {
            "sharpe_delta": s["sharpe_ratio"] - b["sharpe_ratio"],
            "return_delta": s["annual_return"] - b["annual_return"],
            "vol_delta": s["annual_volatility"] - b["annual_volatility"],
            "mdd_delta": s["max_drawdown"] - b["max_drawdown"],
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(metrics: Dict, full_period: str = "", oos_period: str = "") -> None:
    """Pretty-print performance summary table."""
    width = 72
    print("\n" + "=" * width)
    print("  VOLATILITY BACKTEST RESULTS".center(width))
    print("=" * width)
    if full_period:
        print(f"  Full period: {full_period}")
    if oos_period:
        print(f"  OOS period:  {oos_period}")
    print("-" * width)

    labels = [
        ("total_return",    "Total Return",     ".2%"),
        ("annual_return",   "Ann. Return",      ".2%"),
        ("annual_volatility","Ann. Volatility",  ".2%"),
        ("sharpe_ratio",    "Sharpe Ratio",     ".4f"),
        ("max_drawdown",    "Max Drawdown",     ".2%"),
        ("calmar_ratio",    "Calmar Ratio",     ".4f"),
    ]

    header = f"{'Metric':<22} {'Strategy':>12} {'Benchmark':>12} {'Vol-Target':>12}"
    print(header)
    print("-" * width)

    for key, label, fmt in labels:
        sv = metrics.get("strategy", {}).get(key, 0)
        bv = metrics.get("benchmark", {}).get(key, 0)
        vv = metrics.get("vol_target", {}).get(key, 0)
        row = f"{label:<22}"
        for v in [sv, bv, vv]:
            if fmt == ".2%":
                row += f" {v:>11.2%}"
            else:
                row += f" {v:>11.4f}" if isinstance(v, float) else f" {str(v):>11}"
        print(row)

    print("-" * width)

    # Rolling Sharpe stats
    print(f"\n{'Rolling 252d Sharpe Stats':-^{width}}")
    rs_header = f"{'':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}"
    print(rs_header)
    print("-" * 48)
    for col_name in ["strategy", "benchmark", "vol_target"]:
        if col_name in metrics:
            rs = metrics[col_name]
            print(f"  {col_name:<8} {rs['rolling_sharpe_mean']:>+10.4f} "
                  f"{rs['rolling_sharpe_std']:>10.4f} "
                  f"{rs['rolling_sharpe_min']:>+10.4f} "
                  f"{rs['rolling_sharpe_max']:>+10.4f}")

    # Comparison deltas
    comp = metrics.get("comparison", {})
    if comp:
        print(f"\n{'Strategy vs Benchmark':-^{width}}")
        for k, label in [("sharpe_delta", "Sharpe Δ"), ("return_delta", "Return Δ"),
                          ("vol_delta", "Vol Δ"), ("mdd_delta", "MDD Δ")]:
            v = comp.get(k, 0)
            print(f"  {label:<12} {v:>+.4f}")

    print("\n" + "=" * width)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Volatility Trading Backtest for Glaubenskrieg"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "..", "new_data", "data", "tencent_clean"),
        help="Directory containing stock CSV files",
    )
    parser.add_argument("--n-stocks", type=int, default=200,
                        help="Max number of stocks to load")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of stocks in portfolio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for LightGBM")
    parser.add_argument("--output",
                        default=os.path.join(PROJECT_ROOT, "results", "volatility_backtest.json"),
                        help="Output JSON path")
    parser.add_argument("--equity-csv",
                        default=os.path.join(PROJECT_ROOT, "results", "volatility_equity.csv"),
                        help="Output equity curve CSV path")
    parser.add_argument("--transaction-cost", type=float, default=0.0,
                        help="Transaction cost as fraction (e.g. 0.001 for 10bps)")
    args = parser.parse_args()

    if not HAS_LGB:
        logger.error("lightgbm is required.  Install: pip install lightgbm")
        sys.exit(1)

    np.random.seed(args.seed)
    t0 = time.time()

    # ── 1. Load data ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preparing data")
    logger.info("=" * 60)
    features, targets, liquidity, close_prices, dates, stocks, feat_names = \
        load_and_prepare_data(args.data_dir, args.n_stocks)
    T, N, D = features.shape

    # ── 2. Walk-forward training ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Walk-forward volatility prediction")
    logger.info("=" * 60)
    predictions, fold_info = walk_forward_predict(
        features, targets,
        train_days=TRAIN_DAYS, purge_days=PURGE_DAYS,
        val_days=VAL_DAYS, step_days=STEP_DAYS,
        seed=args.seed,
    )
    if not fold_info:
        logger.error("Walk-forward produced no folds — insufficient data?")
        sys.exit(1)

    # OOS start: first day of the first test window
    first_test_start = fold_info[0]["test_dates"][0] if fold_info else 0

    # ── 3. Trading simulation ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Trading simulation (weekly rebalancing)")
    logger.info("=" * 60)
    sim_result = simulate_strategy(
        predictions, close_prices, liquidity, dates, stocks,
        top_k=args.top_k, rebalance_freq=REBALANCE_FREQ,
        transaction_cost=args.transaction_cost,
        oos_start_idx=first_test_start,
    )
    equity_df = sim_result["equity_curve"]
    turnover = sim_result["turnover"]
    active_turnover = turnover[turnover > 0]
    mean_to = float(np.mean(active_turnover)) if len(active_turnover) > 0 else 0.0
    logger.info("Mean turnover: %.4f (%.1f%% per rebalance)", mean_to, mean_to * 100)

    # ── 4. Performance metrics ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Computing performance metrics")
    logger.info("=" * 60)

    # Full-period metrics
    metrics_full = compute_all_metrics(equity_df, oos_start_date=None)
    # OOS-period metrics (from first test window)
    oos_start_date = dates[first_test_start] if first_test_start < len(dates) else None
    metrics_oos = compute_all_metrics(equity_df, oos_start_date=oos_start_date)

    print_summary_table(
        metrics_oos,
        full_period=f"{dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}",
        oos_period=f"{oos_start_date.strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}"
        if oos_start_date else "",
    )

    # ── 5. Save outputs ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Saving outputs")
    logger.info("=" * 60)

    wf_summary = {}
    if fold_info:
        qs = [f["qlike"] for f in fold_info]
        ms = [f["mse"] for f in fold_info]
        ma = [f["mae"] for f in fold_info]
        wf_summary = {
            "n_folds": len(fold_info),
            "mean_qlike": float(np.mean(qs)), "std_qlike": float(np.std(qs)),
            "mean_mse": float(np.mean(ms)), "std_mse": float(np.std(ms)),
            "mean_mae": float(np.mean(ma)), "std_mae": float(np.std(ma)),
        }

    output = {
        "config": {
            "data_dir": os.path.abspath(args.data_dir),
            "n_stocks_loaded": N,
            "top_k": args.top_k,
            "seed": args.seed,
            "target_horizon": TARGET_HORIZON,
            "train_days": TRAIN_DAYS, "purge_days": PURGE_DAYS,
            "val_days": VAL_DAYS, "step_days": STEP_DAYS,
            "rebalance_freq": REBALANCE_FREQ,
            "target_ann_vol": TARGET_ANN_VOL,
            "transaction_cost": args.transaction_cost,
            "stocks": stocks,
            "feature_names": feat_names,
            "date_range": [str(dates[0]), str(dates[-1])],
            "n_dates": int(T),
            "oos_start_date": str(oos_start_date) if oos_start_date else None,
        },
        "walk_forward": wf_summary,
        "fold_details": fold_info,
        "performance_full": metrics_full,
        "performance_oos": metrics_oos,
        "mean_turnover": mean_to,
        "runtime_s": round(time.time() - t0, 2),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved → %s", args.output)

    os.makedirs(os.path.dirname(args.equity_csv) or ".", exist_ok=True)
    equity_df.to_csv(args.equity_csv)
    logger.info("Equity curve saved → %s (%d rows)", args.equity_csv, len(equity_df))

    elapsed = time.time() - t0
    logger.info("Total runtime: %.1f seconds (%.1f min)", elapsed, elapsed / 60)
    print(f"\nDone. Results → {args.output}")
    print(f"Equity  → {args.equity_csv}")


if __name__ == "__main__":
    main()
