#!/usr/bin/env python3
"""Enhanced features: wavelet denoising + FRED macro. Tests 4 configs on US stocks."""
import sys, os, json, time, logging, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("enhance")

PROJECT = "/root/Glaubenskrieg"
sys.path.insert(0, PROJECT)
from src.data.features import compute_all_features
from src.data.wavelet_denoise import wavelet_denoise
import lightgbm as lgb
import pandas_datareader.data as web
import datetime
from scipy.stats import spearmanr

# ── Config ──
DATA_DIR = "/root/data/us_stocks_full"
MIN_ROWS = 2500
WF_TRAIN, WF_PURGE, WF_VAL, WF_STEP = 1008, 126, 252, 126
SEED = 42

# ── Load data ──
def load_stocks(data_dir, min_rows):
    stocks = {}
    for fp in sorted(os.listdir(data_dir)):
        if not fp.endswith(".csv"): continue
        sym = fp.replace(".csv", "")
        df = pd.read_csv(os.path.join(data_dir, fp), index_col=0, parse_dates=True)
        if len(df) >= min_rows:
            stocks[sym] = df
    return stocks

# ── FRED macro ──
def load_fred_macro():
    series = {
        "GDP": "GDP", "UNRATE": "UNRATE", "CPIAUCSL": "CPIAUCSL",
        "DFF": "DFF", "T10Y2Y": "T10Y2Y", "VIXCLS": "VIXCLS",
        "DCOILWTICO": "DCOILWTICO",
    }
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2026, 6, 1)
    macro = {}
    for name, code in series.items():
        try:
            df = web.DataReader(code, "fred", start, end)
            df.columns = [name]
            # Resample to business days, forward fill
            df = df.resample("B").ffill()
            macro[name] = df
        except Exception as e:
            log.warning(f"FRED {name}: {e}")
    # Merge all
    if macro:
        result = macro[list(macro.keys())[0]]
        for name, df in list(macro.items())[1:]:
            result = result.join(df, how="outer")
        result = result.ffill()
        return result
    return None

# ── Feature computation ──
def compute_feature_matrix(stocks, macro_df, use_wavelet=False):
    """Compute features for all stocks, optionally with wavelet denoising."""
    common = None
    for df in stocks.values():
        common = df.index if common is None else common.intersection(df.index)
    
    symbols = sorted(stocks.keys())
    feature_cols = None
    
    # Compute per stock
    all_feats = {}
    for sym in symbols:
        df = stocks[sym].loc[common].copy()
        if use_wavelet:
            close_raw = df["close"].values.astype(np.float64)
            denoised = wavelet_denoise(close_raw, wavelet="db4", level=3)
            df["close"] = denoised
        feat = compute_all_features(df)
        all_feats[sym] = feat
        if feature_cols is None:
            feature_cols = list(feat.columns)
    
    n_days = len(common)
    n_stocks = len(symbols)
    n_feat = len(feature_cols)
    X = np.zeros((n_days, n_stocks, n_feat), dtype=np.float32)
    for j, sym in enumerate(symbols):
        X[:, j, :] = all_feats[sym].values.astype(np.float32)
    
    # Add macro features if available
    if macro_df is not None:
        macro_aligned = macro_df.reindex(common).ffill().values
        macro_aligned = np.nan_to_num(macro_aligned, 0)
        n_macro = macro_aligned.shape[1]
        macro_broadcast = np.tile(macro_aligned[:, np.newaxis, :], (1, n_stocks, 1))
        X = np.concatenate([X, macro_broadcast], axis=2)
        feature_cols = feature_cols + [f"macro_{i}" for i in range(n_macro)]
    
    # Forward returns
    rets = np.zeros((n_days, n_stocks), dtype=np.float32)
    for j, sym in enumerate(symbols):
        p = stocks[sym].loc[common, "close"].values
        rets[5:, j] = p[5:] / p[:-5] - 1.0
    
    return X, rets, symbols, common, feature_cols

# ── Metrics ──
def ic_score(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10: return 0.0
    return float(spearmanr(y_true[mask], y_pred[mask])[0])

def sharpe(returns):
    if len(returns) < 2: return 0.0
    return float(np.mean(returns) / max(np.std(returns), 1e-10)) * np.sqrt(252)

# ── Walk-Forward ──
def walk_forward_ret(X, y, n_stocks, config_name):
    n_days = X.shape[0]
    windows = []
    start = 0
    while start + WF_TRAIN + WF_PURGE + WF_VAL <= n_days:
        t0, t1 = start, start + WF_TRAIN
        t2, t3 = start + WF_TRAIN + WF_PURGE, start + WF_TRAIN + WF_PURGE + WF_VAL
        windows.append((t0, t1, t2, t3))
        start += WF_STEP
    
    folds = []
    for wi, (t0, t1, t2, t3) in enumerate(windows):
        train_idx = np.arange(t0 * n_stocks, t1 * n_stocks)
        val_idx = np.arange(t2 * n_stocks, t3 * n_stocks)
        
        X_tr = X.reshape(-1, X.shape[2])[train_idx]
        y_tr = y.reshape(-1)[train_idx]
        X_vl = X.reshape(-1, X.shape[2])[val_idx]
        y_vl = y.reshape(-1)[val_idx]
        
        mask_tr = ~np.isnan(y_tr)
        mask_vl = ~np.isnan(y_vl)
        if mask_tr.sum() < 200 or mask_vl.sum() < 100: continue
        
        model = lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                  num_leaves=31, random_state=SEED, n_jobs=-1, verbosity=-1)
        model.fit(X_tr[mask_tr], y_tr[mask_tr])
        y_pred = model.predict(X_vl[mask_vl])
        
        fold_ic = ic_score(y_vl[mask_vl], y_pred)
        
        # Strategy Sharpe
        n_v = len(y_pred)
        cutoff = max(int(n_v * 0.2), 1)
        order = np.argsort(y_pred)
        strat_ret = np.concatenate([-y_vl[mask_vl][order[:cutoff]], y_vl[mask_vl][order[-cutoff:]]])
        fold_sharpe = sharpe(strat_ret)
        
        folds.append({"w": wi, "ic": fold_ic, "sharpe": fold_sharpe})
    
    ics = [f["ic"] for f in folds]
    sharpes = [f["sharpe"] for f in folds] if folds else [0]
    return {
        "n_windows": len(folds),
        "ic_mean": float(np.mean(ics)) if ics else 0,
        "ic_std": float(np.std(ics)) if ics else 0,
        "sharpe_mean": float(np.mean(sharpes)),
        "folds": folds,
    }

# ── Main ──
def main():
    t0 = time.time()
    stocks = load_stocks(DATA_DIR, MIN_ROWS)
    log.info(f"Loaded {len(stocks)} stocks")
    
    macro = load_fred_macro()
    log.info(f"FRED macro: {macro.shape if macro is not None else 'N/A'}")
    
    configs = [
        ("raw_ohlcv", False, False),
        ("wavelet", True, False),
        ("raw_macro", False, True),
        ("wavelet_macro", True, True),
    ]
    
    results = {}
    for name, use_wav, use_mac in configs:
        log.info(f"\n{'='*60}\nConfig: {name}\n{'='*60}")
        mdf = macro if use_mac else None
        X, y, syms, common, cols = compute_feature_matrix(stocks, mdf, use_wav)
        log.info(f"Features: {X.shape}, cols={len(cols)}")
        
        ret_result = walk_forward_ret(X, y, len(syms), name)
        results[name] = {
            "ic_mean": ret_result["ic_mean"],
            "ic_std": ret_result["ic_std"],
            "sharpe_mean": ret_result["sharpe_mean"],
            "n_windows": ret_result["n_windows"],
            "n_features": X.shape[2],
        }
        log.info(f"  IC={ret_result['ic_mean']:.4f}±{ret_result['ic_std']:.4f}  Sharpe={ret_result['sharpe_mean']:.2f}")
    
    # Summary
    results["runtime_s"] = time.time() - t0
    with open("/root/results/enhanced_features.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("  ENHANCED FEATURES — RESULTS")
    print("=" * 70)
    for name in ["raw_ohlcv", "wavelet", "raw_macro", "wavelet_macro"]:
        r = results[name]
        print(f"  {name:20s}  IC={r['ic_mean']:.4f}±{r['ic_std']:.4f}  Sharpe={r['sharpe_mean']:.2f}  ({r['n_features']} feats)")
    print(f"\n  Output: /root/results/enhanced_features.json")
    print("=" * 70)

if __name__ == "__main__":
    main()
