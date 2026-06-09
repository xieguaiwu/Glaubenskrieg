#!/usr/bin/env python3
"""
Glaubenskrieg Pipeline v3 — P3 Macro Features + P5 Ensemble + Multi-Seed.

P3: FRED macro (VIX, DFF, T10Y2Y) + short-term reversal + idiosyncratic vol
P5: Ridge+LGB weighted ensemble + multi-seed averaging

Usage:
  cd /root/Glaubenskrieg
  PYTHONPATH=. python scripts/pipeline_v3.py \
    --data-dir /root/data/us_stocks_full \
    --output /root/results/pipeline_v3.json \
    --seeds 42,123,456
"""
import sys, os, json, time, argparse, warnings, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline_v3")

PROJECT = Path("/root/Glaubenskrieg")
sys.path.insert(0, str(PROJECT))
from src.data.features import compute_all_features
import lightgbm as lgb
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

# Optional CatBoost
try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
    logger.info("CatBoost not installed — skipping CB models")

# Macro via FRED
try:
    import pandas_datareader.data as web
    HAS_FRED = True
except ImportError:
    HAS_FRED = False
    logger.info("pandas_datareader not available — skipping macro features")

# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════
MIN_ROWS = 2500
WF_TRAIN, WF_PURGE, WF_VAL, WF_STEP = 1008, 126, 252, 126
HORIZONS = [5, 21]  # focus on horizons that work
TOP_K = 100

LGB_PARAMS = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03,
              "num_leaves": 31, "min_child_samples": 50, "subsample": 0.8,
              "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 0.5,
              "n_jobs": -1, "verbosity": -1}
CB_PARAMS = {"iterations": 300, "depth": 4, "learning_rate": 0.03,
             "random_seed": 42, "verbose": False, "thread_count": -1}

# ═══════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════
def sharpe(r): 
    return float(np.mean(r)/max(np.std(r),1e-10))*np.sqrt(252) if len(r)>1 else 0.0

def ic_score(y_t, y_p):
    m = ~(np.isnan(y_t)|np.isnan(y_p))
    return float(spearmanr(y_t[m], y_p[m])[0]) if m.sum()>=10 else 0.0

def dir_acc(y_t, y_p):
    m = ~(np.isnan(y_t)|np.isnan(y_p))
    return float(np.mean(np.sign(y_t[m])==np.sign(y_p[m]))) if m.sum()>=10 else 0.5

# ═══════════════════════════════════════════════════════
# P3: Macro Features
# ═══════════════════════════════════════════════════════
def load_macro_features() -> Optional[pd.DataFrame]:
    """Load FRED macro series: VIX, Fed Funds, yield curve spread."""
    if not HAS_FRED:
        return None
    series = {"VIXCLS": "VIX", "DFF": "FedFunds", "T10Y2Y": "YieldSpread"}
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2026, 6, 30)
    dfs = {}
    for code, name in series.items():
        try:
            df = web.DataReader(code, "fred", start, end)
            df.columns = [name]
            df = df.resample("B").ffill()
            dfs[name] = df
        except Exception as e:
            logger.warning(f"FRED {name}: {e}")
    if not dfs:
        return None
    result = dfs[list(dfs.keys())[0]]
    for name in list(dfs.keys())[1:]:
        result = result.join(dfs[name], how="outer")
    result = result.ffill().bfill()
    logger.info(f"Macro features: {list(result.columns)}, {len(result)} rows, {result.index[0].date()}~{result.index[-1].date()}")
    return result


# ═══════════════════════════════════════════════════════
# P3: Cross-Sectional + Reversal Features
# ═══════════════════════════════════════════════════════
def build_enhanced_features(
    X_base: np.ndarray, prices: np.ndarray, volumes: np.ndarray,
    macro_df: Optional[pd.DataFrame], common_dates: pd.DatetimeIndex
) -> np.ndarray:
    """
    P3 enhanced feature builder. X_base: (D, S, 9) from compute_all_features.
    Returns: (D, S, N_feat) augmented array.
    """
    D, S, F = X_base.shape
    blocks = [X_base]

    # ── Cross-sectional z-scores (per day, across stocks) ──
    for f in range(F):
        fd = X_base[:, :, f]
        m = np.nanmean(fd, axis=1, keepdims=True)
        s = np.maximum(np.nanstd(fd, axis=1, keepdims=True), 1e-8)
        blocks.append(np.nan_to_num((fd - m) / s, 0)[:, :, np.newaxis])

    # ── Cross-sectional percentile ranks ──
    for f in range(F):
        fd = X_base[:, :, f]
        r = np.zeros_like(fd)
        for d in range(D):
            v = fd[d]; ok = ~np.isnan(v)
            if ok.sum() > 1: r[d, ok] = pd.Series(v[ok]).rank(pct=True).values
        blocks.append(r[:, :, np.newaxis])

    # ── Pairwise interactions ──
    for a, b in [(4,5), (2,4), (6,4)]:
        blocks.append((X_base[:,:,a]*X_base[:,:,b])[:,:,np.newaxis])

    # ── Multi-period momentum ranks ──
    rets_1d = np.zeros_like(prices); rets_1d[1:] = prices[1:]/prices[:-1]-1
    for period in [5, 21, 63, 126]:
        mom = np.zeros_like(prices)
        mom[period:] = prices[period:]/prices[:-period]-1
        mr = np.zeros_like(mom)
        for d in range(period, D):
            v = mom[d]; ok = ~np.isnan(v)
            if ok.sum() > 1: mr[d, ok] = pd.Series(v[ok]).rank(pct=True).values
        blocks.append(mr[:, :, np.newaxis])

    # ── Amihud illiquidity ──
    dv = prices * np.maximum(volumes, 1.0)
    illiq = np.abs(rets_1d) / np.maximum(dv, 1e-12)
    illiq_ma = np.full_like(illiq, np.nan)
    for d in range(21, D): illiq_ma[d] = np.nanmedian(illiq[d-20:d+1], axis=0)
    illiq_r = np.zeros_like(illiq_ma)
    for d in range(21, D):
        v=illiq_ma[d]; ok=~np.isnan(v)
        if ok.sum()>1: illiq_r[d,ok]=pd.Series(v[ok]).rank(pct=True).values
    blocks.append(illiq_r[:,:,np.newaxis])

    # ── P3 NEW: Short-term reversal (Jegadeesh 1990) ──
    rev = -rets_1d  # negative of 1d return
    rev_r = np.zeros_like(rev)
    for d in range(1, D):
        v=rev[d]; ok=~np.isnan(v)
        if ok.sum()>1: rev_r[d,ok]=pd.Series(v[ok]).rank(pct=True).values
    blocks.append(rev_r[:,:,np.newaxis])

    # ── P3 NEW: Idiosyncratic volatility (simple: residual vol vs market) ──
    mkt_ret = np.nanmean(rets_1d, axis=1)
    idio_vol = np.full((D, S), np.nan)
    for j in range(S):
        for d in range(63, D):
            r = rets_1d[d-62:d+1, j]; mr = mkt_ret[d-62:d+1]
            ok = ~np.isnan(r) & ~np.isnan(mr)
            if ok.sum() > 20:
                try:
                    beta = np.polyfit(mr[ok], r[ok], 1)[0]
                    resid = r[ok] - beta * mr[ok]
                    idio_vol[d, j] = np.std(resid)
                except: pass
    idio_r = np.zeros_like(idio_vol)
    for d in range(63, D):
        v=idio_vol[d]; ok=~np.isnan(v)
        if ok.sum()>1: idio_r[d,ok]=pd.Series(v[ok]).rank(pct=True).values
    blocks.append(idio_r[:,:,np.newaxis])

    # ── P3 NEW: Macro features (broadcast across stocks) ──
    if macro_df is not None:
        macro_aligned = macro_df.reindex(common_dates).ffill().bfill().values
        macro_aligned = np.nan_to_num(macro_aligned, 0)
        n_macro = macro_aligned.shape[1]
        macro_bc = np.tile(macro_aligned[:, np.newaxis, :], (1, S, 1))
        blocks.append(macro_bc)

    X = np.concatenate(blocks, axis=2).astype(np.float32)
    return X


# ═══════════════════════════════════════════════════════
# Targets
# ═══════════════════════════════════════════════════════
def make_targets(prices: np.ndarray, horizons: List[int]) -> Dict[int, np.ndarray]:
    D, S = prices.shape
    tgt = {}
    for h in horizons:
        r = np.full((D, S), np.nan, dtype=np.float32)
        r[:-h] = prices[h:]/prices[:-h] - 1
        tgt[h] = r
    return tgt


# ═══════════════════════════════════════════════════════
# Walk-Forward Evaluation
# ═══════════════════════════════════════════════════════
def gen_windows(n: int): 
    ws=[]; start=0
    while start+WF_TRAIN+WF_PURGE+WF_VAL<=n:
        ws.append((start,start+WF_TRAIN,start+WF_TRAIN+WF_PURGE,start+WF_TRAIN+WF_PURGE+WF_VAL))
        start+=WF_STEP
    return ws

def train_predict(X_tr, y_tr, X_vl, model_type, seed):
    m_tr = ~np.isnan(y_tr)
    # Remove rows with NaN in features (critical for Ridge/sklearn)
    if np.any(np.isnan(X_tr)) or np.any(np.isinf(X_tr)):
        m_tr &= ~np.isnan(X_tr).any(axis=1) & ~np.isinf(X_tr).any(axis=1)
    if np.any(np.isnan(X_vl)) or np.any(np.isinf(X_vl)):
        X_vl = np.nan_to_num(X_vl, nan=0.0, posinf=0.0, neginf=0.0)
    if model_type == "lgb":
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(X_tr[m_tr], y_tr[m_tr])
        return m.predict(X_vl)
    elif model_type == "ridge":
        m = Ridge(alpha=1.0, random_state=seed)
        m.fit(X_tr[m_tr], y_tr[m_tr])
        return m.predict(X_vl)
    elif model_type == "cb" and HAS_CB:
        p = dict(CB_PARAMS); p["random_seed"] = seed
        m = cb.CatBoostRegressor(**p)
        m.fit(X_tr[m_tr], y_tr[m_tr], verbose=False)
        return m.predict(X_vl)
    return np.zeros(len(X_vl))

def ensemble_predict(preds_dict: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Weighted average of model predictions."""
    out = np.zeros_like(list(preds_dict.values())[0])
    w_sum = 0.0
    for name, pred in preds_dict.items():
        if name in weights:
            out += weights[name] * pred
            w_sum += weights[name]
    return out / max(w_sum, 1e-10)


# ═══════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════
def run_pipeline(data_dir: str, seeds: List[int]) -> dict:
    _t0 = time.time()
    results = {"seeds": seeds, "horizons": list(HORIZONS)}

    # ── 1. Load ──
    files = sorted(Path(data_dir).glob("*.csv"))
    stocks = {}
    for fp in files:
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if len(df) >= MIN_ROWS: stocks[fp.stem] = df
    logger.info(f"Loaded {len(stocks)} stocks")

    # ── 2. Features ──
    feats = {}
    for i, (sym, df) in enumerate(stocks.items()):
        feats[sym] = compute_all_features(df)
        if (i+1)%150==0: logger.info(f"  Features: {i+1}/{len(stocks)}")

    common = None
    for f in feats.values():
        common = f.index if common is None else common.intersection(f.index)
    symbols = sorted(feats.keys())
    S = len(symbols); D = len(common)
    base_cols = list(feats[symbols[0]].columns)

    X_base = np.zeros((D, S, len(base_cols)), dtype=np.float32)
    prices = np.zeros((D, S), dtype=np.float32)
    volumes = np.zeros((D, S), dtype=np.float32)
    for j, sym in enumerate(symbols):
        X_base[:,j,:] = feats[sym].loc[common].values.astype(np.float32)
        prices[:,j] = stocks[sym].loc[common,'close'].values.astype(np.float32)
        volumes[:,j] = stocks[sym].loc[common,'volume'].values.astype(np.float32)

    # ── 3. P3 Enhanced ──
    macro = load_macro_features()
    X = build_enhanced_features(X_base, prices, volumes, macro, common)
    n_feat = X.shape[2]
    logger.info(f"Features: {n_feat} total ({len(base_cols)} base + {n_feat-len(base_cols)} enhanced)")

    # ── 4. Targets ──
    targets = make_targets(prices, HORIZONS)

    results["data"] = {"n_stocks": S, "n_days": D, "n_features": n_feat,
                       "dates": [str(common[0].date()), str(common[-1].date())]}

    # ── 5. Walk-Forward (per seed, per horizon, per model) ──
    all_horizon = {}
    model_types = ["lgb", "ridge"]
    if HAS_CB: model_types.append("cb")

    for h in HORIZONS:
        label = f"{h}d"
        logger.info(f"\n{'='*60}\nHorizon: {label}\n{'='*60}")
        all_horizon[label] = {}

        for seed in seeds:
            logger.info(f"  Seed {seed}...")
            windows = gen_windows(D)
            Xf = X.reshape(-1, n_feat)
            yf = targets[h].reshape(-1)
            seed_folds = {m: [] for m in model_types}
            seed_ens_folds = {"ridge_lgb_avg": [], "ridge_lgb_weighted": []}

            for wi, (t0,t1,t2,t3) in enumerate(windows):
                tr_idx = np.arange(t0*S, t1*S)
                vl_idx = np.arange(t2*S, t3*S)
                Xtr, ytr = Xf[tr_idx], yf[tr_idx]
                Xvl, yvl = Xf[vl_idx], yf[vl_idx]
                mv = ~np.isnan(yvl)
                if mv.sum() < 50: continue

                preds = {}
                val_ics = {}

                for mt in model_types:
                    p = train_predict(Xtr, ytr, Xvl, mt, seed)
                    preds[mt] = p
                    val_ics[mt] = ic_score(yvl[mv], p[mv])
                    fold_ic = val_ics[mt]
                    fold_sh = sharpe(np.concatenate([-yvl[mv][np.argsort(p[mv])[:max(int(len(p[mv])*0.2),1)]],
                                                     yvl[mv][np.argsort(p[mv])[-max(int(len(p[mv])*0.2),1):]]]))
                    seed_folds[mt].append({"w": wi, "ic": float(fold_ic), "sharpe": float(fold_sh)})

                # P5: Ridge+LGB equal-weight ensemble
                if "lgb" in preds and "ridge" in preds:
                    p_avg = (preds["lgb"] + preds["ridge"]) / 2
                    ic_avg = ic_score(yvl[mv], p_avg[mv])
                    seed_ens_folds["ridge_lgb_avg"].append({"w": wi, "ic": float(ic_avg)})

                    # Weighted by val IC
                    w_sum = abs(val_ics.get("lgb",0)) + abs(val_ics.get("ridge",0))
                    if w_sum > 1e-10:
                        w_lgb = abs(val_ics["lgb"])/w_sum
                        w_ridge = abs(val_ics["ridge"])/w_sum
                        p_w = w_lgb*preds["lgb"] + w_ridge*preds["ridge"]
                        ic_w = ic_score(yvl[mv], p_w[mv])
                        seed_ens_folds["ridge_lgb_weighted"].append({"w": wi, "ic": float(ic_w)})

            # Summarize per seed
            seed_summary = {}
            for mt in model_types:
                ics = [f["ic"] for f in seed_folds[mt]]
                seed_summary[mt] = {"ic_mean": float(np.mean(ics)), "ic_std": float(np.std(ics)),
                                    "n_windows": len(ics), "folds": seed_folds[mt]}
            for ename in ["ridge_lgb_avg", "ridge_lgb_weighted"]:
                ics = [f["ic"] for f in seed_ens_folds[ename]]
                if ics:
                    seed_summary[ename] = {"ic_mean": float(np.mean(ics)), "ic_std": float(np.std(ics)),
                                           "n_windows": len(ics)}

            all_horizon[label][f"seed_{seed}"] = seed_summary
            # Print best result for this seed
            best = max(seed_summary.items(), key=lambda x: x[1]["ic_mean"])
            logger.info(f"    Best: {best[0]} IC={best[1]['ic_mean']:.4f}±{best[1]['ic_std']:.4f}")

        # ── Cross-seed summary ──
        logger.info(f"  --- Cross-seed avg ({label}) ---")
        for mt in model_types + ["ridge_lgb_avg", "ridge_lgb_weighted"]:
            s_ics = []
            for seed in seeds:
                sk = f"seed_{seed}"
                if sk in all_horizon[label] and mt in all_horizon[label][sk]:
                    s_ics.append(all_horizon[label][sk][mt]["ic_mean"])
            if s_ics:
                all_horizon[label][f"cross_seed_{mt}"] = {
                    "ic_mean": float(np.mean(s_ics)), "ic_std": float(np.std(s_ics)),
                    "seeds_used": len(s_ics)
                }
                logger.info(f"    {mt:25s}: IC={np.mean(s_ics):.4f}±{np.std(s_ics):.4f} ({len(s_ics)} seeds)")

    results["horizons"] = all_horizon
    results["runtime_s"] = round(time.time() - _t0, 1)
    return results


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/data/us_stocks_full")
    parser.add_argument("--output", default="/root/results/pipeline_v3.json")
    parser.add_argument("--seeds", default="42,123,456")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    results = run_pipeline(args.data_dir, seeds)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    # Summary
    print("\n" + "="*80)
    print("  PIPELINE v3 — P3 Macro + P5 Ensemble + Multi-Seed")
    print("="*80)
    for label, hd in results["horizons"].items():
        print(f"\n  {label}:")
        for k, v in hd.items():
            if k.startswith("cross_seed_"):
                print(f"    {k.replace('cross_seed_',''):25s} IC={v['ic_mean']:.4f}±{v['ic_std']:.4f} ({v['seeds_used']} seeds)")
    print(f"\n  Runtime: {results['runtime_s']:.0f}s  |  Output: {args.output}")
    print("="*80)

if __name__ == "__main__":
    main()
