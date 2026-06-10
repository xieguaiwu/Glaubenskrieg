#!/usr/bin/env python3
"""
Glaubenskrieg v2 — Noise-Eliminated SSM Architecture.

Three variants compared against Ridge baseline:

Variant A: Frozen SSM (HiPPO-initialized, NO training of SSM params)
  → SSM acts as deterministic temporal smoother on raw price sequences
  → Only input/output projection layers are trained
  → Cannot overfit — SSM is a fixed filter

Variant B: Residual SSM (SSM learns only the delta from linear baseline)
  → ŷ = Linear(35 features) + λ · SSM(raw prices)
  → λ initialized to 0.01, L1 regularized on SSM output
  → If SSM is useless, λ → 0 naturally

Variant C: Raw-price SSM (SSM on raw OHLCV only, then combine with features)
  → SSM processes raw OHLCV (T=63) → temporal embedding
  → Linear processes 35 features → CS embedding  
  → Combine via small MLP

All use the SAME enhanced features + 5d target as proven Ridge baseline.
"""
import sys, os, json, time, warnings, logging, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("glauben_v2")

PROJECT = Path("/root/Glaubenskrieg")
sys.path.insert(0, str(PROJECT))
from src.data.features import compute_all_features
from src.model.ctm_model import CTMStockModel

# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════
MIN_ROWS = 2500; FORWARD_H = 5; SEQ_LEN = 63
WF_TRAIN, WF_PURGE, WF_VAL, WF_STEP = 1008, 126, 252, 126
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════
def ic_score(y_t, y_p):
    m = ~(np.isnan(y_t)|np.isnan(y_p))
    return float(spearmanr(y_t[m], y_p[m])[0]) if m.sum()>=10 else 0.0
def sharpe(r):
    return float(np.mean(r)/max(np.std(r),1e-10))*np.sqrt(252) if len(r)>1 else 0.0

# ═══════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════
def build_features(X_base, prices, volumes):
    D, S, F = X_base.shape; blocks = [X_base]
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

# ═══════════════════════════════════════════════════════
# Sequence Builder
# ═══════════════════════════════════════════════════════
def build_seq(data, rets, t0, t1, t2, t3, N):
    """Build (B, N, T, D) sequences. data: (n_days, N, n_feat)"""
    seqs_X, seqs_y = [], []
    for d in range(t0+SEQ_LEN-1, t1):
        seq = data[d-SEQ_LEN+1:d+1]  # (T, N, D)
        seqs_X.append(seq.transpose(1,0,2)); seqs_y.append(rets[d])
    val_X, val_y = [], []
    for d in range(t2+SEQ_LEN-1, t3):
        seq = data[d-SEQ_LEN+1:d+1]
        val_X.append(seq.transpose(1,0,2)); val_y.append(rets[d])
    if not seqs_X: return None
    Xtr=np.nan_to_num(np.stack(seqs_X),0); ytr=np.stack(seqs_y)
    Xvl=np.nan_to_num(np.stack(val_X),0); yvl=np.stack(val_y) if val_X else None
    m_tr=~np.isnan(ytr).any(axis=1); Xtr=Xtr[m_tr]; ytr=ytr[m_tr]
    if Xvl is not None: m_vl=~np.isnan(yvl).any(axis=1); Xvl=Xvl[m_vl]; yvl=yvl[m_vl]
    return Xtr, ytr, Xvl, yvl

# ═══════════════════════════════════════════════════════
# Variant A: Frozen SSM (deterministic temporal smoother)
# ═══════════════════════════════════════════════════════
class FrozenSSMPredictor(nn.Module):
    """
    SSM with HiPPO-initialized, FROZEN parameters. Only projection layers train.
    The frozen SSM acts as a structured temporal filter (like a multi-scale EMA).
    """
    def __init__(self, input_dim=9, d_model=8, state_dim=4, n_layers=1):
        super().__init__()
        self.ssm = CTMStockModel(input_dim=input_dim, model_dim=d_model, state_dim=state_dim,
                                  n_layers=n_layers, output_dim=0)  # encode-only
        # Freeze ALL SSM parameters
        for p in self.ssm.parameters(): p.requires_grad = False
        self.proj = nn.Linear(d_model, 1)
    
    def forward(self, x):
        """x: (B, N, T, D_in) — raw OHLCV"""
        B, N, T, D = x.shape
        xf = x.reshape(B*N, T, D)
        h = self.ssm.encode(xf)                      # (B*N, T, d_model)
        h_last = h[:, -1, :]                          # (B*N, d_model)
        out = self.proj(h_last)                       # (B*N, 1)
        return out.reshape(B, N)

# ═══════════════════════════════════════════════════════
# Variant B: Residual SSM (SSM learns delta from linear)
# ═══════════════════════════════════════════════════════
class ResidualSSMPredictor(nn.Module):
    """
    ŷ = Linear(35 features) + λ · SSM(raw prices)
    λ = sigmoid(trainable_scalar) → initializes near 0
    """
    def __init__(self, feat_dim=35, raw_dim=9, d_model=8, state_dim=4):
        super().__init__()
        self.linear = nn.Linear(feat_dim, 1)          # Ridge equivalent
        self.ssm = CTMStockModel(input_dim=raw_dim, model_dim=d_model, state_dim=state_dim,
                                  n_layers=1, output_dim=0)
        self.ssm_proj = nn.Linear(d_model, 1)
        self.lambda_raw = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ≈ 0.047
    
    def forward(self, x_feat, x_raw):
        """x_feat: (total_samples, feat_dim); x_raw: (B, N, T, raw_dim)"""
        # x_feat is already flattened: (B*N, feat_dim)
        B_raw, N_raw = x_raw.shape[0], x_raw.shape[1]
        y_linear = self.linear(x_feat)  # (B*N, 1)
        # SSM path
        Bn, Nn, T, D = x_raw.shape
        xf = x_raw.reshape(Bn*Nn, T, D)
        h = self.ssm.encode(xf); h_last = h[:,-1,:]
        y_ssm = self.ssm_proj(h_last)  # (B*N, 1)
        lam = torch.sigmoid(self.lambda_raw)
        return y_linear + lam * y_ssm, lam

# ═══════════════════════════════════════════════════════
# Variant C: Decoupled paths
# ═══════════════════════════════════════════════════════
class DecoupledPredictor(nn.Module):
    """SSM on raw prices + Linear on features → small MLP combine."""
    def __init__(self, feat_dim=35, raw_dim=9, d_model=8, state_dim=4):
        super().__init__()
        self.linear = nn.Linear(feat_dim, 8)
        self.ssm = CTMStockModel(input_dim=raw_dim, model_dim=d_model, state_dim=state_dim,
                                  n_layers=1, output_dim=0)
        self.ssm_proj = nn.Linear(d_model, 8)
        self.combine = nn.Sequential(nn.Linear(16, 4), nn.ReLU(), nn.Linear(4, 1))
    
    def forward(self, x_feat, x_raw):
        """x_feat: (total_samples, feat_dim); x_raw: (B, N, T, raw_dim)"""
        y_lin = F.relu(self.linear(x_feat))  # (B*N, 8)
        Bn, Nn, T, D = x_raw.shape
        xf = x_raw.reshape(Bn*Nn, T, D)
        h = self.ssm.encode(xf); h_last = h[:,-1,:]
        y_ssm = F.relu(self.ssm_proj(h_last))  # (B*N, 8)
        return self.combine(torch.cat([y_lin, y_ssm], dim=-1))  # (B*N, 1)

# ═══════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════
def train_variant(model, variant_name, X_feat, X_raw, rets, windows, N=50):
    results = []
    for wi, (t0,t1,t2,t3) in enumerate(windows):
        # Build feature and raw-price sequences
        data_f = build_seq(X_feat[:,:N,:], rets[:,:N], t0,t1,t2,t3, N)
        data_r = build_seq(X_raw[:,:N,:], rets[:,:N], t0,t1,t2,t3, N)
        if data_f is None or data_r is None: continue
        Xtr_f, ytr, Xvl_f, yvl = data_f
        Xtr_r, _, Xvl_r, _ = data_r
        if len(Xtr_f) < 20 or Xvl_f is None or len(Xvl_f) < 10: continue
        
        model_cp = copy.deepcopy(model).to(DEVICE)
        opt = torch.optim.Adam(model_cp.parameters(), lr=1e-3)
        
        Xtr_ft = torch.from_numpy(Xtr_f).float().to(DEVICE)
        Xtr_rt = torch.from_numpy(Xtr_r).float().to(DEVICE)
        ytr_t = torch.from_numpy(ytr).float().to(DEVICE)
        Xvl_ft = torch.from_numpy(Xvl_f).float().to(DEVICE)
        Xvl_rt = torch.from_numpy(Xvl_r).float().to(DEVICE)
        yvl_t = torch.from_numpy(yvl).float().to(DEVICE)
        
        B = 16; best_loss = float('inf'); pat = 0
        for ep in range(30):
            model_cp.train(); perm = torch.randperm(len(Xtr_ft))
            for i in range(0, len(Xtr_ft), B):
                idx = perm[i:i+B]
                if variant_name == "frozen":
                    pred = model_cp(Xtr_rt[idx])
                elif variant_name == "residual":
                    pred, lam = model_cp(Xtr_ft[idx][:,:,-1,:].reshape(len(idx)*N, -1), Xtr_rt[idx])
                else:  # decoupled
                    pred = model_cp(Xtr_ft[idx][:,:,-1,:].reshape(len(idx)*N, -1), Xtr_rt[idx])
                target_flat = ytr_t[idx].reshape(-1)
                pred_flat = pred.reshape(-1) if pred.dim() > 1 else pred
                loss = F.mse_loss(pred_flat, target_flat)
                # L1 on SSM output for residual variant
                if variant_name == "residual":
                    loss = loss + 0.001 * lam
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model_cp.parameters(), 1.0)
                opt.step()
            
            model_cp.eval()
            with torch.no_grad():
                if variant_name == "frozen": vp = model_cp(Xvl_rt)
                elif variant_name == "residual": vp, _ = model_cp(Xvl_ft[:,:,-1,:].reshape(len(Xvl_ft)*N, -1), Xvl_rt)
                else: vp = model_cp(Xvl_ft[:,:,-1,:].reshape(len(Xvl_ft)*N, -1), Xvl_rt)
                vp_flat = vp.reshape(-1); vt_flat = yvl_t.reshape(-1)
                vl = F.mse_loss(vp_flat, vt_flat).item()
            if vl < best_loss: best_loss=vl; pat=0
            else: pat+=1
            if pat>=5: break
        
        model_cp.eval()
        with torch.no_grad():
            if variant_name == "frozen": vp = model_cp(Xvl_rt).cpu().numpy()
            elif variant_name == "residual": vp, lam_final = model_cp(Xvl_ft[:,:,-1,:].reshape(len(Xvl_ft)*N, -1), Xvl_rt); vp=vp.cpu().numpy()
            else: vp = model_cp(Xvl_ft[:,:,-1,:].reshape(len(Xvl_ft)*N, -1), Xvl_rt).cpu().numpy()
        yv = yvl_t.cpu().numpy().reshape(-1)
        
        vpf=vp.reshape(-1); yvf=yv.reshape(-1); m=~np.isnan(yvf)
        ic = ic_score(yvf[m], vpf[m]) if m.sum()>=10 else 0.0
        cutoff=max(int(m.sum()*0.2),1); order=np.argsort(vpf[m]); y2=yvf[m]
        sh = sharpe(np.concatenate([-y2[order[:cutoff]], y2[order[-cutoff:]]]))
        
        n_p = sum(p.numel() for p in model_cp.parameters() if p.requires_grad)
        r = {"window":wi,"ic":ic,"sharpe":sh,"variant":variant_name,"trainable_params":n_p}
        if variant_name == "residual": r["lambda"]=float(lam_final)
        results.append(r)
        logger.info(f"  {variant_name:10s} w{wi:02d}: IC={ic:+.4f} Sharpe={sh:+.2f} params={n_p}")
        del model_cp; torch.cuda.empty_cache()
    return results

# ═══════════════════════════════════════════════════════
# Ridge baseline (identical to pipeline_v2)
# ═══════════════════════════════════════════════════════
def run_ridge(X, rets, S, windows):
    D=X.shape[0]; nf=X.shape[2]; Xf=X.reshape(-1,nf); yf=rets.reshape(-1); r=[]
    for wi,(t0,t1,t2,t3) in enumerate(windows):
        tr=np.arange(t0*S,t1*S); vl=np.arange(t2*S,t3*S)
        Xtr,ytr=Xf[tr],yf[tr]; Xvl,yvl=Xf[vl],yf[vl]
        mt=~np.isnan(ytr)&~np.isnan(Xtr).any(1); mv=~np.isnan(yvl); Xc=np.nan_to_num(Xvl,0)
        if mt.sum()<200 or mv.sum()<50: continue
        m=Ridge(alpha=1.0); m.fit(Xtr[mt],ytr[mt]); yp=m.predict(Xc[mv])
        ic=ic_score(yvl[mv],yp); co=max(int(mv.sum()*0.2),1); o=np.argsort(yp); yv=yvl[mv]
        sh=sharpe(np.concatenate([-yv[o[:co]],yv[o[-co:]]])); r.append({"window":wi,"ic":ic,"sharpe":sh,"variant":"ridge"})
    return r

# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
def main():
    t0=time.time(); logger.info(f"Device: {DEVICE}")
    # Load
    files=sorted(Path("/root/data/us_stocks_full").glob("*.csv")); stocks={}
    for fp in files:
        df=pd.read_csv(fp,index_col=0,parse_dates=True)
        if len(df)>=MIN_ROWS: stocks[fp.stem]=df
    logger.info(f"Loaded {len(stocks)} stocks")
    # Features
    feats={}
    for i,(sym,df) in enumerate(stocks.items()):
        feats[sym]=compute_all_features(df)
        if (i+1)%150==0: logger.info(f"  Features: {i+1}/{len(stocks)}")
    common=None
    for f in feats.values(): common=f.index if common is None else common.intersection(f.index)
    syms=sorted(feats.keys()); S=len(syms); D=len(common)
    X_base=np.zeros((D,S,9),np.float32); prices=np.zeros((D,S),np.float32)
    vols=np.zeros((D,S),np.float32)
    for j,sym in enumerate(syms):
        X_base[:,j,:]=feats[sym].loc[common].values.astype(np.float32)
        prices[:,j]=stocks[sym].loc[common,'close'].values.astype(np.float32)
        vols[:,j]=stocks[sym].loc[common,'volume'].values.astype(np.float32)
    
    X_enhanced=build_features(X_base,prices,vols)
    X_enhanced=np.nan_to_num(X_enhanced,0)
    rets=np.full((D,S),np.nan,np.float32); rets[:-FORWARD_H]=prices[FORWARD_H:]/prices[:-FORWARD_H]-1
    
    # Raw OHLCV for SSM (just the 9 base features, not enhanced)
    X_raw=np.nan_to_num(X_base,0)
    
    windows=[]; start=0
    while start+WF_TRAIN+WF_PURGE+WF_VAL<=D:
        windows.append((start,start+WF_TRAIN,start+WF_TRAIN+WF_PURGE,start+WF_TRAIN+WF_PURGE+WF_VAL))
        start+=WF_STEP
    logger.info(f"Walk-forward: {len(windows)} windows")
    
    # Ridge baseline
    logger.info("="*50+"\nRIDGE\n"+"="*50)
    ridge_res=run_ridge(X_enhanced,rets,S,windows)
    for r in ridge_res: logger.info(f"  ridge w{r['window']:02d}: IC={r['ic']:+.4f}")
    
    # Variant A: Frozen SSM
    logger.info("="*50+"\nVARIANT A: FROZEN SSM\n"+"="*50)
    model_a=FrozenSSMPredictor(input_dim=9,d_model=8,state_dim=4)
    res_a=train_variant(model_a,"frozen",X_enhanced[...,:9],X_raw,rets,windows)
    
    # Variant B: Residual SSM
    logger.info("="*50+"\nVARIANT B: RESIDUAL SSM\n"+"="*50)
    model_b=ResidualSSMPredictor(feat_dim=X_enhanced.shape[2],raw_dim=9,d_model=8,state_dim=4)
    res_b=train_variant(model_b,"residual",X_enhanced,X_raw,rets,windows)
    
    # Variant C: Decoupled
    logger.info("="*50+"\nVARIANT C: DECOUPLED\n"+"="*50)
    model_c=DecoupledPredictor(feat_dim=X_enhanced.shape[2],raw_dim=9,d_model=8,state_dim=4)
    res_c=train_variant(model_c,"decoupled",X_enhanced,X_raw,rets,windows)
    
    # Summary
    all_res={"ridge":ridge_res,"frozen":res_a,"residual":res_b,"decoupled":res_c}
    for name,res in all_res.items():
        ics=[r["ic"] for r in res]
        logger.info(f"  {name:10s}: IC={np.mean(ics):.4f}±{np.std(ics):.4f} ({len(ics)}w)")
    
    with open("/root/results/glauben_v2_noise_free.json","w") as f:
        json.dump({"runtime_s":round(time.time()-t0,1),"variants":{k:[{kk:vv for kk,vv in r.items()} for r in v] for k,v in all_res.items()}},f,indent=2,default=str)
    
    print("\n"+"="*70)
    print("  GLAUBENSKRIEG v2 — NOISE ELIMINATION RESULTS")
    print("="*70)
    for name,res in all_res.items():
        ics=[r["ic"] for r in res]
        print(f"  {name:12s}: IC={np.mean(ics):+.4f}±{np.std(ics):.4f}")
    print("="*70)

if __name__=="__main__": main()
