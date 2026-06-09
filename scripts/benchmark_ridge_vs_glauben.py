#!/usr/bin/env python3
"""
Benchmark: Ridge vs Minimalist Glaubenskrieg on identical 35-feature 5d data.

Tests whether a ~500-param Mamba SSM can match or exceed Ridge (451 params) 
when given the same enhanced features and prediction horizon.
"""
import sys, os, json, time, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

PROJECT = Path("/root/Glaubenskrieg")
sys.path.insert(0, str(PROJECT))
from src.data.features import compute_all_features
from src.model.multiasset_ctm import MultiAssetCTM

# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════
MIN_ROWS = 2500
FORWARD_H = 5
SEQ_LEN = 63
WF_TRAIN, WF_PURGE, WF_VAL, WF_STEP = 1008, 126, 252, 126

# Glaubenskrieg minimalist config
GLAUBEN_CONFIG = dict(
    n_assets=50, input_dim=35, model_dim=4, state_dim=2, n_layers=1,
    output_dim=1, embedding_dim=2, use_cross_attention=False,
    dropout=0.05, conv_kernel=3,
)

# ═══════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════
def ic_score(y_t, y_p):
    m = ~(np.isnan(y_t)|np.isnan(y_p))
    return float(spearmanr(y_t[m], y_p[m])[0]) if m.sum()>=10 else 0.0

def sharpe(r):
    return float(np.mean(r)/max(np.std(r),1e-10))*np.sqrt(252) if len(r)>1 else 0.0

# ═══════════════════════════════════════════════════════
# Feature Engineering (identical to pipeline_v2)
# ═══════════════════════════════════════════════════════
def build_features(X_base, prices, volumes):
    D, S, F = X_base.shape
    blocks = [X_base]
    for f in range(F):
        fd=X_base[:,:,f]; m=np.nanmean(fd,axis=1,keepdims=True); s=np.maximum(np.nanstd(fd,axis=1,keepdims=True),1e-8)
        blocks.append(np.nan_to_num((fd-m)/s,0)[:,:,np.newaxis])
    for f in range(F):
        fd=X_base[:,:,f]; r=np.zeros_like(fd)
        for d in range(D):
            v=fd[d]; ok=~np.isnan(v)
            if ok.sum()>1: r[d,ok]=pd.Series(v[ok]).rank(pct=True).values
        blocks.append(r[:,:,np.newaxis])
    for a,b in [(4,5),(2,4),(6,4)]: blocks.append((X_base[:,:,a]*X_base[:,:,b])[:,:,np.newaxis])
    rets_1d=np.zeros_like(prices); rets_1d[1:]=prices[1:]/prices[:-1]-1
    for period in [5,21,63,126]:
        mom=np.zeros_like(prices); mom[period:]=prices[period:]/prices[:-period]-1
        mr=np.zeros_like(mom)
        for d in range(period,D):
            v=mom[d]; ok=~np.isnan(v)
            if ok.sum()>1: mr[d,ok]=pd.Series(v[ok]).rank(pct=True).values
        blocks.append(mr[:,:,np.newaxis])
    dv=prices*np.maximum(volumes,1.0); illiq=np.abs(rets_1d)/np.maximum(dv,1e-12)
    illiq_ma=np.full_like(illiq,np.nan)
    for d in range(21,D): illiq_ma[d]=np.nanmedian(illiq[d-20:d+1],axis=0)
    illiq_r=np.zeros_like(illiq_ma)
    for d in range(21,D):
        v=illiq_ma[d]; ok=~np.isnan(v)
        if ok.sum()>1: illiq_r[d,ok]=pd.Series(v[ok]).rank(pct=True).values
    blocks.append(illiq_r[:,:,np.newaxis])
    return np.concatenate(blocks,axis=2).astype(np.float32)

def load_data(data_dir):
    files = sorted(Path(data_dir).glob("*.csv"))
    stocks = {}
    for fp in files:
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if len(df) >= MIN_ROWS: stocks[fp.stem] = df
    return stocks

# ═══════════════════════════════════════════════════════
# Ridge walk-forward (mirrors pipeline_v2)
# ═══════════════════════════════════════════════════════
def run_ridge(X, rets, S, windows, seed=42):
    D = X.shape[0]; nf = X.shape[2]
    Xf = X.reshape(-1, nf); yf = rets.reshape(-1)
    results = []
    for wi, (t0,t1,t2,t3) in enumerate(windows):
        tr_idx = np.arange(t0*S, t1*S); vl_idx = np.arange(t2*S, t3*S)
        Xtr, ytr = Xf[tr_idx], yf[tr_idx]; Xvl, yvl = Xf[vl_idx], yf[vl_idx]
        m_tr = ~np.isnan(ytr) & ~np.isnan(Xtr).any(axis=1)
        m_vl = ~np.isnan(yvl)
        Xvl_c = np.nan_to_num(Xvl, 0)
        if m_tr.sum()<200 or m_vl.sum()<50: continue
        m = Ridge(alpha=1.0, random_state=seed)
        m.fit(Xtr[m_tr], ytr[m_tr])
        yp = m.predict(Xvl_c[m_vl])
        ic = ic_score(yvl[m_vl], yp)
        cutoff=max(int(m_vl.sum()*0.2),1); order=np.argsort(yp); yv=yvl[m_vl]
        sh = sharpe(np.concatenate([-yv[order[:cutoff]], yv[order[-cutoff:]]]))
        results.append({"window":wi,"ic":ic,"sharpe":sh,"model":"ridge"})
    return results

# ═══════════════════════════════════════════════════════
# Glaubenskrieg walk-forward (sequence-based training)
# ═══════════════════════════════════════════════════════
def build_sequences(X, rets, S, t0, t1, t2, t3):
    """
    Build (B, N, T, D) sequences for MultiAssetCTM.
    X: (n_days, n_stocks, n_features)
    Uses sliding windows of SEQ_LEN days.
    """
    D = X.shape[0]; nf = X.shape[2]
    
    # Use top N assets by volume (first N after sorting)
    N = GLAUBEN_CONFIG["n_assets"]
    N = min(N, S)
    
    seqs_X = []; seqs_y = []
    # Training sequences
    for d in range(t0+SEQ_LEN-1, t1):
        seq = X[d-SEQ_LEN+1:d+1, :N, :]  # (T, N, D)
        # Transpose to (N, T, D) → add batch dim later
        seqs_X.append(seq.transpose(1,0,2))  # (N, T, D)
        seqs_y.append(rets[d, :N])  # (N,) target is forward return at day d
    
    # Validation sequences  
    val_X = []; val_y = []
    for d in range(t2+SEQ_LEN-1, t3):
        seq = X[d-SEQ_LEN+1:d+1, :N, :]
        val_X.append(seq.transpose(1,0,2))
        val_y.append(rets[d, :N])
    
    if not seqs_X: return None
    
    X_train = np.stack(seqs_X)  # (B, N, T, D)
    y_train = np.stack(seqs_y)  # (B, N)
    X_val = np.stack(val_X) if val_X else None
    y_val = np.stack(val_y) if val_y else None
    
    # Remove NaN targets and ensure no NaN in features
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    mask_tr = ~np.isnan(y_train).any(axis=1)
    X_train = X_train[mask_tr]; y_train = y_train[mask_tr]
    
    if X_val is not None:
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        mask_vl = ~np.isnan(y_val).any(axis=1)
        X_val = X_val[mask_vl]; y_val = y_val[mask_vl]
    
    return X_train, y_train, X_val, y_val, N

def run_glauben(X, rets, S, windows, device="cuda"):
    results = []
    
    for wi, (t0,t1,t2,t3) in enumerate(windows):
        data = build_sequences(X, rets, S, t0, t1, t2, t3)
        if data is None: continue
        Xtr, ytr, Xvl, yvl, N = data
        if len(Xtr) < 20 or Xvl is None or len(Xvl) < 10: continue
        
        # Build model
        cfg = dict(GLAUBEN_CONFIG); cfg["n_assets"] = N
        model = MultiAssetCTM(**cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        Xtr_t = torch.from_numpy(Xtr).float().to(device)
        ytr_t = torch.from_numpy(ytr).float().to(device)
        Xvl_t = torch.from_numpy(Xvl).float().to(device)
        yvl_t = torch.from_numpy(yvl).float().to(device)
        
        # Train
        B = 16
        best_val_loss = float('inf'); patience_counter = 0
        for epoch in range(30):
            model.train()
            perm = torch.randperm(len(Xtr_t))
            for i in range(0, len(Xtr_t), B):
                idx = perm[i:i+B]
                xb, yb = Xtr_t[idx], ytr_t[idx]
                out = model(xb)  # (B, T, N*4)
                # Extract regression output: shape (B, T, N) from first N channels
                Bs, Tlen = out.shape[0], out.shape[1]
                pred = out[:, -1, :N]  # last timestep, regression channels
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Validate
            model.eval()
            with torch.no_grad():
                vout = model(Xvl_t)
                vpred = vout[:, -1, :N]
                vloss = criterion(vpred, yvl_t).item()
            
            if vloss < best_val_loss:
                best_val_loss = vloss; patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5: break
        
        # Predict on validation
        model.eval()
        with torch.no_grad():
            vout = model(Xvl_t)
            vpred = vout[:, -1, :N].cpu().numpy()
        yv = yvl_t.cpu().numpy()
        
        # Flatten predictions and targets
        vp_f = vpred.reshape(-1); yv_f = yv.reshape(-1)
        m = ~np.isnan(yv_f)
        ic = ic_score(yv_f[m], vp_f[m]) if m.sum()>=10 else 0.0
        cutoff=max(int(m.sum()*0.2),1); order=np.argsort(vp_f[m]); yf2=yv_f[m]
        sh = sharpe(np.concatenate([-yf2[order[:cutoff]], yf2[order[-cutoff:]]]))
        
        n_params = sum(p.numel() for p in model.parameters())
        results.append({"window":wi,"ic":ic,"sharpe":sh,"model":"glauben","params":n_params,
                        "val_loss":best_val_loss})
        logger.info(f"  glauben w{wi:02d}: IC={ic:+.4f} Sharpe={sh:+.2f} params={n_params} loss={best_val_loss:.4f}")
        
        del model, optimizer, Xtr_t, ytr_t, Xvl_t, yvl_t
        torch.cuda.empty_cache()
    
    return results

# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # ── Load ──
    stocks = load_data("/root/data/us_stocks_full")
    logger.info(f"Loaded {len(stocks)} stocks")
    
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
    # Replace NaN/Inf in features (momentum/illiq have NaN for early days)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    rets = np.full((D,S), np.nan, dtype=np.float32)
    rets[:-FORWARD_H] = prices[FORWARD_H:]/prices[:-FORWARD_H] - 1
    
    # ── Windows ──
    windows = []; start = 0
    while start+WF_TRAIN+WF_PURGE+WF_VAL <= D:
        windows.append((start,start+WF_TRAIN,start+WF_TRAIN+WF_PURGE,start+WF_TRAIN+WF_PURGE+WF_VAL))
        start += WF_STEP
    logger.info(f"Walk-forward: {len(windows)} windows")
    
    # ── Run Ridge (CPU, fast) ──
    logger.info("="*60)
    logger.info("RIDGE BASELINE")
    logger.info("="*60)
    ridge_results = run_ridge(X, rets, S, windows)
    for r in ridge_results:
        logger.info(f"  ridge w{r['window']:02d}: IC={r['ic']:+.4f} Sharpe={r['sharpe']:+.2f}")
    
    # ── Run Glaubenskrieg (GPU) ──
    logger.info("="*60)
    logger.info("GLAUBENSKRIEG (minimalist)")
    logger.info("="*60)
    glauben_results = run_glauben(X, rets, S, windows, device)
    
    # ── Summary ──
    r_ics = [r["ic"] for r in ridge_results]
    g_ics = [g["ic"] for g in glauben_results]
    
    results = {
        "data": {"n_stocks": S, "n_days": D, "n_features": X.shape[2]},
        "ridge": {"n_windows": len(r_ics), "ic_mean": float(np.mean(r_ics)), 
                  "ic_std": float(np.std(r_ics)), "folds": ridge_results},
        "glauben": {"n_windows": len(g_ics), "ic_mean": float(np.mean(g_ics)),
                    "ic_std": float(np.std(g_ics)), "folds": glauben_results,
                    "config": GLAUBEN_CONFIG},
        "runtime_s": round(time.time()-t0, 1),
    }
    
    with open("/root/results/benchmark_ridge_glauben.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("  RIDGE vs GLAUBENSKRIEG — COMPARISON")
    print("="*70)
    print(f"  Ridge:      IC={np.mean(r_ics):.4f}±{np.std(r_ics):.4f}  ({len(r_ics)} windows)")
    print(f"  Glaubenskrieg: IC={np.mean(g_ics):.4f}±{np.std(g_ics):.4f}  ({len(g_ics)} windows)")
    print(f"  Δ:          {np.mean(g_ics)-np.mean(r_ics):+.4f}")
    print(f"  Runtime: {results['runtime_s']:.0f}s")
    print("="*70)

if __name__ == "__main__":
    main()
