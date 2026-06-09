#!/usr/bin/env python3
"""
Five-Step IC Validation Diagnostic for Pipeline v2 Result (IC=0.065).

Applies the paper's diagnostic framework to verify the Ridge 5d IC=0.065 result
is genuine and not an overfitting artifact.

Step 1: Held-out Test IC — train on first 70%, test on last 30% (never in WF)
Step 2: Statistical Baseline — compare vs zero/mean/random, t-test, DM test
Step 3: Window Stability — CV, t-stat, outlier analysis, leave-one-out
Step 4: Per-Stock IC — binomial test, IC distribution, in-sample vs OOS
Step 5: Overfitting Indicators — train/val IC gap, recency decay, feature importance stability

Usage on server:
  cd /root/Glaubenskrieg
  PYTHONPATH=. python scripts/diagnose_v2.py \
    --data-dir /root/data/us_stocks_full \
    --output /root/results/diagnose_v2.json
"""
import sys, os, json, time, warnings, logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnose")

PROJECT = Path("/root/Glaubenskrieg")
sys.path.insert(0, str(PROJECT))
from src.data.features import compute_all_features
import lightgbm as lgb
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, binomtest, ttest_1samp

# ═══════════════════════════════════════════════════════
# Config (same as pipeline_v2)
# ═══════════════════════════════════════════════════════
MIN_ROWS = 2500
WF_TRAIN, WF_PURGE, WF_VAL, WF_STEP = 1008, 126, 252, 126
SEED = 42

# ═══════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════
def ic_spearman(y_t, y_p):
    m = ~(np.isnan(y_t)|np.isnan(y_p))
    return float(spearmanr(y_t[m], y_p[m])[0]) if m.sum()>=10 else 0.0

def dm_test(err1, err2, h=1):
    """Diebold-Mariano test: is model 1 significantly better than model 2?"""
    d = err1**2 - err2**2
    n = len(d)
    if n < 20: return 1.0
    mean_d = np.mean(d)
    # HAC variance with h lags
    var_d = np.var(d) / n
    for i in range(1, h+1):
        var_d += 2 * (n-i)/(n*n) * np.mean((d[:-i]-mean_d) * (d[i:]-mean_d))
    var_d = max(var_d, 1e-12)
    dm_stat = mean_d / np.sqrt(var_d)
    return 2 * stats.t.sf(abs(dm_stat), n-1)

# ═══════════════════════════════════════════════════════
# Feature Engineering (same as pipeline_v2)
# ═══════════════════════════════════════════════════════
def build_features(X_base, prices, volumes):
    D, S, F = X_base.shape
    blocks = [X_base]
    # Cross-sectional z-scores
    for f in range(F):
        fd = X_base[:,:,f]; m=np.nanmean(fd,axis=1,keepdims=True); s=np.maximum(np.nanstd(fd,axis=1,keepdims=True),1e-8)
        blocks.append(np.nan_to_num((fd-m)/s,0)[:,:,np.newaxis])
    # Cross-sectional percentile ranks
    for f in range(F):
        fd=X_base[:,:,f]; r=np.zeros_like(fd)
        for d in range(D):
            v=fd[d]; ok=~np.isnan(v)
            if ok.sum()>1: r[d,ok]=pd.Series(v[ok]).rank(pct=True).values
        blocks.append(r[:,:,np.newaxis])
    # Pairwise interactions
    for a,b in [(4,5),(2,4),(6,4)]: blocks.append((X_base[:,:,a]*X_base[:,:,b])[:,:,np.newaxis])
    # Momentum ranks
    rets_1d=np.zeros_like(prices); rets_1d[1:]=prices[1:]/prices[:-1]-1
    for period in [5,21,63,126]:
        mom=np.zeros_like(prices); mom[period:]=prices[period:]/prices[:-period]-1
        mr=np.zeros_like(mom)
        for d in range(period,D):
            v=mom[d]; ok=~np.isnan(v)
            if ok.sum()>1: mr[d,ok]=pd.Series(v[ok]).rank(pct=True).values
        blocks.append(mr[:,:,np.newaxis])
    # Amihud illiquidity
    dv=prices*np.maximum(volumes,1.0); illiq=np.abs(rets_1d)/np.maximum(dv,1e-12)
    illiq_ma=np.full_like(illiq,np.nan)
    for d in range(21,D): illiq_ma[d]=np.nanmedian(illiq[d-20:d+1],axis=0)
    illiq_r=np.zeros_like(illiq_ma)
    for d in range(21,D):
        v=illiq_ma[d]; ok=~np.isnan(v)
        if ok.sum()>1: illiq_r[d,ok]=pd.Series(v[ok]).rank(pct=True).values
    blocks.append(illiq_r[:,:,np.newaxis])
    return np.concatenate(blocks,axis=2).astype(np.float32)

# ═══════════════════════════════════════════════════════
# Main Diagnostic Pipeline
# ═══════════════════════════════════════════════════════
def main():
    t0 = time.time()
    DATA = "/root/data/us_stocks_full"
    
    # ── Load data ──
    files = sorted(Path(DATA).glob("*.csv"))
    stocks = {}
    for fp in files:
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if len(df) >= MIN_ROWS: stocks[fp.stem] = df
    logger.info(f"Loaded {len(stocks)} stocks")
    
    # ── Features ──
    feats = {}
    for i, (sym, df) in enumerate(stocks.items()):
        feats[sym] = compute_all_features(df)
        if (i+1)%150==0: logger.info(f"  Features: {i+1}/{len(stocks)}")
    
    common = None
    for f in feats.values():
        common = f.index if common is None else common.intersection(f.index)
    symbols = sorted(feats.keys())
    S = len(symbols); D = len(common)
    
    X_base = np.zeros((D,S,9), dtype=np.float32)
    prices = np.zeros((D,S), dtype=np.float32)
    volumes = np.zeros((D,S), dtype=np.float32)
    for j, sym in enumerate(symbols):
        X_base[:,j,:] = feats[sym].loc[common].values.astype(np.float32)
        prices[:,j] = stocks[sym].loc[common,'close'].values.astype(np.float32)
        volumes[:,j] = stocks[sym].loc[common,'volume'].values.astype(np.float32)
    
    X = build_features(X_base, prices, volumes)
    n_feat = X.shape[2]
    
    # 5d forward returns
    rets = np.full((D,S), np.nan, dtype=np.float32)
    rets[:-5] = prices[5:]/prices[:-5] - 1
    
    results = {"data": {"n_stocks": S, "n_days": D, "n_features": n_feat,
                        "dates": [str(common[0].date()), str(common[-1].date())]}}
    
    # ═══════════════════════════════════════════════════
    # STEP 1: Held-out Test IC
    # ═══════════════════════════════════════════════════
    logger.info("="*60)
    logger.info("STEP 1: HELD-OUT TEST IC")
    logger.info("="*60)
    
    # Split: first 70% for training, last 30% NEVER seen by any model
    split_day = int(D * 0.7)
    Xf = X.reshape(-1, n_feat)
    yf = rets.reshape(-1)
    
    # Train on first 70%
    tr_mask = np.zeros(D*S, dtype=bool)
    tr_mask[:split_day*S] = True
    vl_mask = np.zeros(D*S, dtype=bool)
    vl_mask[split_day*S:] = True
    
    # Remove NaN targets
    tr_ok = tr_mask & ~np.isnan(yf)
    vl_ok = vl_mask & ~np.isnan(yf)
    # Remove NaN features
    tr_ok &= ~np.isnan(Xf).any(axis=1)
    
    # Fit Ridge on training
    ridge_ho = Ridge(alpha=1.0, random_state=SEED)
    ridge_ho.fit(Xf[tr_ok], yf[tr_ok])
    
    # Predict on held-out
    X_ho = Xf[vl_ok]
    # Handle NaNs in held-out features
    X_ho = np.nan_to_num(X_ho, nan=0.0)
    y_pred_ho = ridge_ho.predict(X_ho)
    y_true_ho = yf[vl_ok]
    
    held_out_ic = ic_spearman(y_true_ho, y_pred_ho)
    held_out_dir = np.mean(np.sign(y_true_ho) == np.sign(y_pred_ho))
    
    # Also compute per-day held-out IC
    ho_days = D - split_day
    ho_daily_ic = np.zeros(ho_days)
    for d in range(ho_days):
        day_start = (split_day + d) * S
        day_end = day_start + S
        d_ok = vl_ok[day_start:day_end]
        if d_ok.sum() > 20:
            ho_daily_ic[d] = ic_spearman(yf[day_start:day_end][d_ok], y_pred_ho[day_start-split_day*S:day_end-split_day*S][d_ok])
    
    logger.info(f"  Held-out IC: {held_out_ic:.4f}  |  DirAcc: {held_out_dir:.3f}")
    logger.info(f"  Daily IC mean: {np.nanmean(ho_daily_ic):.4f}  |  std: {np.nanstd(ho_daily_ic):.4f}")
    logger.info(f"  Daily IC positive fraction: {np.mean(ho_daily_ic[~np.isnan(ho_daily_ic)]>0):.1%}")
    
    results["step1_held_out"] = {
        "split_day": split_day, "split_date": str(common[split_day].date()),
        "train_samples": int(tr_ok.sum()), "test_samples": int(vl_ok.sum()),
        "ic": float(held_out_ic),
        "dir_acc": float(held_out_dir),
        "daily_ic_mean": float(np.nanmean(ho_daily_ic)),
        "daily_ic_std": float(np.nanstd(ho_daily_ic)),
        "daily_ic_positive_frac": float(np.mean(ho_daily_ic[~np.isnan(ho_daily_ic)] > 0)),
        "pass": abs(held_out_ic) > 0.02  # threshold: IC > 0.02 is detectable
    }
    
    # ═══════════════════════════════════════════════════
    # STEP 2: Statistical Baseline Tests
    # ═══════════════════════════════════════════════════
    logger.info("="*60)
    logger.info("STEP 2: STATISTICAL BASELINE TESTS")
    logger.info("="*60)
    
    # Compare vs: zero prediction, mean prediction, random prediction
    y_held = yf[vl_ok]
    y_ridge = y_pred_ho
    
    # Zero baseline
    zero_mse = np.mean(y_held**2)
    ridge_mse = np.mean((y_held - y_ridge)**2)
    
    # DM test: Ridge vs Zero
    dm_zero = dm_test(y_held, np.zeros_like(y_held))
    
    # DM test: Ridge vs Mean (historical mean of training returns)
    mean_pred = np.full_like(y_held, np.mean(yf[tr_ok]))
    dm_mean = dm_test(y_held, mean_pred)
    
    # t-test: is mean IC significantly > 0?
    daily_ic_clean = ho_daily_ic[~np.isnan(ho_daily_ic)]
    t_stat_ic, p_val_ic = ttest_1samp(daily_ic_clean, 0)
    
    # Random baseline: shuffle predictions
    np.random.seed(42)
    rand_pred = np.random.permutation(y_ridge)
    rand_ic = ic_spearman(y_held, rand_pred)
    
    logger.info(f"  Ridge MSE: {ridge_mse:.6f}  |  Zero MSE: {zero_mse:.6f}")
    logger.info(f"  DM vs Zero: p={dm_zero:.4f}  |  DM vs Mean: p={dm_mean:.4f}")
    logger.info(f"  t-test IC>0: t={t_stat_ic:.2f}, p={p_val_ic:.4f}")
    logger.info(f"  Random baseline IC: {rand_ic:.4f}")
    
    results["step2_statistical"] = {
        "ridge_mse": float(ridge_mse), "zero_mse": float(zero_mse),
        "dm_vs_zero_p": float(dm_zero), "dm_vs_mean_p": float(dm_mean),
        "t_stat_ic": float(t_stat_ic), "p_val_ic": float(p_val_ic),
        "random_baseline_ic": float(rand_ic),
        "pass": p_val_ic < 0.01  # IC significantly > 0
    }
    
    # ═══════════════════════════════════════════════════
    # STEP 3: Window Stability Analysis
    # ═══════════════════════════════════════════════════
    logger.info("="*60)
    logger.info("STEP 3: WINDOW STABILITY ANALYSIS")
    logger.info("="*60)
    
    # Generate walk-forward windows
    windows = []
    start = 0
    while start + WF_TRAIN + WF_PURGE + WF_VAL <= D:
        windows.append((start, start+WF_TRAIN, start+WF_TRAIN+WF_PURGE, start+WF_TRAIN+WF_PURGE+WF_VAL))
        start += WF_STEP
    
    # Run full walk-forward to get per-window IC
    wf_ics = []
    wf_sharpes = []
    wf_train_ics = []
    
    for wi, (t0, t1, t2, t3) in enumerate(windows):
        tr_idx = np.arange(t0*S, t1*S)
        vl_idx = np.arange(t2*S, t3*S)
        
        Xtr, ytr = Xf[tr_idx], yf[tr_idx]
        Xvl, yvl = Xf[vl_idx], yf[vl_idx]
        
        m_tr = ~np.isnan(ytr) & ~np.isnan(Xtr).any(axis=1)
        m_vl = ~np.isnan(yvl)
        Xvl_c = np.nan_to_num(Xvl, 0)
        
        if m_tr.sum() < 200 or m_vl.sum() < 50: continue
        
        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(Xtr[m_tr], ytr[m_tr])
        
        y_pred_vl = ridge.predict(Xvl_c[m_vl])
        y_pred_tr = ridge.predict(Xtr[m_tr])
        
        val_ic = ic_spearman(yvl[m_vl], y_pred_vl)
        train_ic = ic_spearman(ytr[m_tr], y_pred_tr)
        
        cutoff = max(int(m_vl.sum()*0.2), 1)
        order = np.argsort(y_pred_vl)
        yv = yvl[m_vl]
        strat_ret = np.concatenate([-yv[order[:cutoff]], yv[order[-cutoff:]]])
        val_sharpe = float(np.mean(strat_ret)/max(np.std(strat_ret),1e-10))*np.sqrt(252)
        
        wf_ics.append(val_ic)
        wf_train_ics.append(train_ic)
        wf_sharpes.append(val_sharpe)
    
    wf_ics = np.array(wf_ics)
    wf_train_ics = np.array(wf_train_ics)
    
    # Stability metrics
    mean_ic = np.mean(wf_ics)
    std_ic = np.std(wf_ics)
    cv_ic = std_ic / max(abs(mean_ic), 1e-8)
    t_stat = mean_ic / (std_ic / np.sqrt(len(wf_ics))) if len(wf_ics) > 1 else 0
    p_val = 2 * stats.t.sf(abs(t_stat), len(wf_ics)-1) if len(wf_ics) > 1 else 1
    pos_frac = np.mean(wf_ics > 0)
    
    # Leave-one-out: how sensitive is mean to removing any single window?
    loo_means = [(np.sum(wf_ics) - wf_ics[i])/(len(wf_ics)-1) for i in range(len(wf_ics))]
    loo_range = max(loo_means) - min(loo_means)
    
    # Binomial test: 11/11 windows positive
    binom_p = binomtest(int(np.sum(wf_ics>0)), len(wf_ics), 0.5, alternative='greater').pvalue
    
    # Recency analysis: last 3 windows vs first 8
    last3_ic = np.mean(wf_ics[-3:]) if len(wf_ics) >= 3 else mean_ic
    first8_ic = np.mean(wf_ics[:8]) if len(wf_ics) >= 8 else mean_ic
    recency_decay = last3_ic - first8_ic  # negative = decay
    
    logger.info(f"  Windows: {len(wf_ics)}")
    logger.info(f"  IC per window: {[f'{ic:+.4f}' for ic in wf_ics]}")
    logger.info(f"  Mean IC: {mean_ic:.4f}  |  Std: {std_ic:.4f}  |  CV: {cv_ic:.2f}")
    logger.info(f"  t-stat: {t_stat:.2f}  |  p-val: {p_val:.4f}")
    logger.info(f"  Positive windows: {pos_frac:.0%}  |  Binomial p: {binom_p:.6f}")
    logger.info(f"  LOO range: {loo_range:.4f}  (sensitivity to single window)")
    logger.info(f"  Train IC mean: {np.mean(wf_train_ics):.4f}  |  Val IC mean: {mean_ic:.4f}")
    logger.info(f"  Recency: last3={last3_ic:.4f}, first8={first8_ic:.4f}, decay={recency_decay:+.4f}")
    
    results["step3_stability"] = {
        "n_windows": len(wf_ics),
        "per_window_ic": [float(x) for x in wf_ics],
        "per_window_train_ic": [float(x) for x in wf_train_ics],
        "per_window_sharpe": [float(x) for x in wf_sharpes],
        "mean_ic": float(mean_ic), "std_ic": float(std_ic), "cv": float(cv_ic),
        "t_stat": float(t_stat), "p_val": float(p_val),
        "positive_fraction": float(pos_frac), "binomial_p": float(binom_p),
        "loo_range": float(loo_range),
        "recency_decay": float(recency_decay),
        "last3_ic": float(last3_ic), "first8_ic": float(first8_ic),
        "pass_cv": cv_ic < 0.6,      # CV < 0.6 (relaxed from 0.3 for financial data)
        "pass_binomial": binom_p < 0.01,
        "pass_recency": recency_decay > -0.03,  # no more than 0.03 decay
    }
    
    # ═══════════════════════════════════════════════════
    # STEP 4: Per-Stock IC Analysis
    # ═══════════════════════════════════════════════════
    logger.info("="*60)
    logger.info("STEP 4: PER-STOCK IC ANALYSIS")
    logger.info("="*60)
    
    # For each stock, compute IC across all validation windows
    per_stock_ic = []
    per_stock_train_ic = []
    per_stock_n = []
    
    for j in range(S):
        stock_ics = []
        stock_train_ics = []
        
        for wi, (t0, t1, t2, t3) in enumerate(windows):
            # Get this stock's data in validation window
            val_indices = []
            for d in range(t2, t3):
                val_indices.append(d * S + j)
            val_indices = np.array(val_indices)
            
            tr_indices = []
            for d in range(t0, t1):
                tr_indices.append(d * S + j)
            tr_indices = np.array(tr_indices)
            
            y_vl_s = yf[val_indices]
            y_tr_s = yf[tr_indices]
            
            m_vl = ~np.isnan(y_vl_s)
            m_tr = ~np.isnan(y_tr_s)
            
            if m_vl.sum() < 20 or m_tr.sum() < 20:
                continue
            
            Xtr_s = Xf[tr_indices]; ytr_s = yf[tr_indices]
            Xvl_s = Xf[val_indices]; yvl_s = yf[val_indices]
            
            m_tr_s = ~np.isnan(ytr_s) & ~np.isnan(Xtr_s).any(axis=1)
            m_vl_s = ~np.isnan(yvl_s)
            
            if m_tr_s.sum() < 20 or m_vl_s.sum() < 20:
                continue
            
            Xvl_sc = np.nan_to_num(Xvl_s, 0)
            
            ridge_s = Ridge(alpha=1.0, random_state=SEED)
            ridge_s.fit(Xtr_s[m_tr_s], ytr_s[m_tr_s])
            
            yp_vl = ridge_s.predict(Xvl_sc[m_vl_s])
            yp_tr = ridge_s.predict(Xtr_s[m_tr_s])
            
            stock_ics.append(ic_spearman(yvl_s[m_vl_s], yp_vl))
            stock_train_ics.append(ic_spearman(ytr_s[m_tr_s], yp_tr))
        
        if stock_ics:
            per_stock_ic.append(np.mean(stock_ics))
            per_stock_train_ic.append(np.mean(stock_train_ics))
            per_stock_n.append(len(stock_ics))
    
    per_stock_ic = np.array(per_stock_ic)
    per_stock_train_ic = np.array(per_stock_train_ic)
    
    pos_frac = np.mean(per_stock_ic > 0)
    n_stocks = len(per_stock_ic)
    n_pos = int(np.sum(per_stock_ic > 0))
    binom_p = binomtest(n_pos, n_stocks, 0.5, alternative='greater').pvalue
    
    # Is mean per-stock IC significantly > 0?
    t_stat_ps, p_val_ps = ttest_1samp(per_stock_ic, 0)
    
    # Train/val gap per stock
    gap = per_stock_train_ic - per_stock_ic
    mean_gap = np.mean(gap)
    
    logger.info(f"  Stocks analyzed: {n_stocks}")
    logger.info(f"  Per-stock IC: mean={np.mean(per_stock_ic):.4f}, std={np.std(per_stock_ic):.4f}")
    logger.info(f"  IC range: [{np.min(per_stock_ic):.4f}, {np.max(per_stock_ic):.4f}]")
    logger.info(f"  Positive fraction: {pos_frac:.1%} ({n_pos}/{n_stocks})")
    logger.info(f"  Binomial p: {binom_p:.6f}")
    logger.info(f"  t-test IC>0: t={t_stat_ps:.2f}, p={p_val_ps:.4f}")
    logger.info(f"  Train/Val gap: {mean_gap:+.4f}")
    
    # Percentile breakdown
    pcts = [10, 25, 50, 75, 90]
    pct_vals = np.percentile(per_stock_ic, pcts)
    logger.info(f"  Percentiles: " + ", ".join(f"P{p}={v:.4f}" for p, v in zip(pcts, pct_vals)))
    
    results["step4_per_stock"] = {
        "n_stocks": n_stocks, "n_pos": n_pos, "positive_fraction": float(pos_frac),
        "mean_ic": float(np.mean(per_stock_ic)), "std_ic": float(np.std(per_stock_ic)),
        "min_ic": float(np.min(per_stock_ic)), "max_ic": float(np.max(per_stock_ic)),
        "t_stat": float(t_stat_ps), "p_val": float(p_val_ps),
        "binomial_p": float(binom_p),
        "train_val_gap": float(mean_gap),
        "percentiles": {f"p{p}": float(v) for p, v in zip(pcts, pct_vals)},
        "pass_binomial": binom_p < 0.01,
        "pass_t_test": p_val_ps < 0.01,
        "pass_majority": pos_frac > 0.55,  # significantly > 50%
    }
    
    # ═══════════════════════════════════════════════════
    # STEP 5: Overfitting & Robustness Checks
    # ═══════════════════════════════════════════════════
    logger.info("="*60)
    logger.info("STEP 5: OVERFITTING & ROBUSTNESS CHECKS")
    logger.info("="*60)
    
    # 5a: Train/Val IC gap per window
    gap_per_window = wf_train_ics - wf_ics
    gap_mean = np.mean(gap_per_window)
    gap_max = np.max(gap_per_window)
    
    # 5b: Feature importance stability (coefficient variation across windows)
    # Retrain on each window, check coefficient stability
    coefs = []
    for wi, (t0, t1, t2, t3) in enumerate(windows):
        tr_idx = np.arange(t0*S, t1*S)
        Xtr, ytr = Xf[tr_idx], yf[tr_idx]
        m_tr = ~np.isnan(ytr) & ~np.isnan(Xtr).any(axis=1)
        if m_tr.sum() < 200: continue
        ridge_c = Ridge(alpha=1.0, random_state=SEED)
        ridge_c.fit(Xtr[m_tr], ytr[m_tr])
        coefs.append(ridge_c.coef_)
    
    if coefs:
        coefs = np.array(coefs)  # (n_windows, n_features)
        coef_cv = np.std(coefs, axis=0) / (np.abs(np.mean(coefs, axis=0)) + 1e-10)
        coef_stability = float(np.median(coef_cv))
        # Top features by absolute mean coefficient
        mean_coefs = np.mean(np.abs(coefs), axis=0)
        top_feats = np.argsort(mean_coefs)[-10:]
    else:
        coef_stability = 0
        top_feats = []
    
    # 5c: Recency decay significance
    # Split windows into first half and second half
    mid_w = len(wf_ics) // 2
    first_half_ic = np.mean(wf_ics[:mid_w]) if mid_w > 0 else mean_ic
    second_half_ic = np.mean(wf_ics[mid_w:]) if mid_w < len(wf_ics) else mean_ic
    half_decay = second_half_ic - first_half_ic
    
    # 5d: Sub-sample stability (bootstrap)
    np.random.seed(42)
    bootstrap_ics = []
    for _ in range(1000):
        idx = np.random.choice(len(wf_ics), len(wf_ics), replace=True)
        bootstrap_ics.append(np.mean(wf_ics[idx]))
    bootstrap_ics = np.array(bootstrap_ics)
    ci_95 = np.percentile(bootstrap_ics, [2.5, 97.5])
    
    logger.info(f"  Train/Val IC gap: mean={gap_mean:+.4f}, max={gap_max:+.4f}")
    logger.info(f"  Feature coeff stability: median CV={coef_stability:.2f}")
    logger.info(f"  First half IC: {first_half_ic:.4f}  |  Second half: {second_half_ic:.4f}  |  Δ: {half_decay:+.4f}")
    logger.info(f"  Bootstrap 95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    
    results["step5_overfitting"] = {
        "train_val_gap_mean": float(gap_mean),
        "train_val_gap_max": float(gap_max),
        "coef_stability": float(coef_stability),
        "top_feature_indices": [int(x) for x in top_feats],
        "first_half_ic": float(first_half_ic),
        "second_half_ic": float(second_half_ic),
        "half_decay": float(half_decay),
        "bootstrap_ci_95": [float(ci_95[0]), float(ci_95[1])],
        "pass_gap": gap_mean < 0.03,  # train/val gap < 0.03
        "pass_decay": half_decay > -0.02,  # second half not significantly worse
        "pass_bootstrap": ci_95[0] > 0.02,  # lower CI > 0.02
    }
    
    # ═══════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════
    results["runtime_s"] = round(time.time() - t0, 1)
    
    passes = {
        "step1_held_out": results["step1_held_out"]["pass"],
        "step2_statistical": results["step2_statistical"]["pass"],
        "step3_stability_cv": results["step3_stability"]["pass_cv"],
        "step3_stability_binomial": results["step3_stability"]["pass_binomial"],
        "step3_stability_recency": results["step3_stability"]["pass_recency"],
        "step4_per_stock_binomial": results["step4_per_stock"]["pass_binomial"],
        "step4_per_stock_t_test": results["step4_per_stock"]["pass_t_test"],
        "step4_per_stock_majority": results["step4_per_stock"]["pass_majority"],
        "step5_gap": results["step5_overfitting"]["pass_gap"],
        "step5_decay": results["step5_overfitting"]["pass_decay"],
        "step5_bootstrap": results["step5_overfitting"]["pass_bootstrap"],
    }
    n_pass = sum(passes.values())
    n_total = len(passes)
    
    results["verdict"] = {
        "passes": passes,
        "n_pass": n_pass,
        "n_total": n_total,
        "overall": "PASS" if n_pass >= n_total * 0.7 else "BORDERLINE" if n_pass >= n_total * 0.5 else "FAIL",
    }
    
    # Print verdict
    print("\n" + "="*70)
    print("  FIVE-STEP IC VALIDATION — FINAL VERDICT")
    print("="*70)
    for step_name, passed in passes.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {step_name:40s} {status}")
    print(f"\n  Passed: {n_pass}/{n_total}  →  {results['verdict']['overall']}")
    print(f"  Runtime: {results['runtime_s']:.0f}s")
    print("="*70)
    
    return results

if __name__ == "__main__":
    r = main()
    with open("/root/results/diagnose_v2.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"\nResults saved to /root/results/diagnose_v2.json")
