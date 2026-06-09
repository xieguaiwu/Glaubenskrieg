#!/usr/bin/env python3
"""
Glaubenskrieg Production Training Pipeline — Trains and SAVES models for deployment.

Key additions over pipeline_v2.py:
  - Saves trained Ridge model per walk-forward window
  - Saves feature engineering metadata (column names, normalization state)
  - Exports the LAST window's model as the production model
  - Saves all artifacts to a dated model directory

Usage:
  cd /root/Glaubenskrieg
  PYTHONPATH=. python scripts/train_production.py \
    --data-dir /root/data/us_stocks_full \
    --model-dir /root/models/production_$(date +%Y%m%d)
"""
import sys, os, json, time, argparse, warnings, logging, joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_prod")

PROJECT = Path("/root/Glaubenskrieg")
sys.path.insert(0, str(PROJECT))
from src.data.features import compute_all_features, compute_forward_returns, compute_sma, compute_rsi, compute_bollinger_bands, compute_volume_ratio, compute_realized_volatility
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ═══════════════════════════════════════════════════════
# Config (same as validated pipeline_v2)
# ═══════════════════════════════════════════════════════
MIN_ROWS = 2500
FORWARD_HORIZON = 5
WF_TRAIN, WF_PURGE, WF_VAL, WF_STEP = 1008, 126, 252, 126
LGB_PARAMS = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03,
              "num_leaves": 31, "min_child_samples": 50, "subsample": 0.8,
              "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 0.5,
              "n_jobs": -1, "verbosity": -1, "random_state": 42}

# ═══════════════════════════════════════════════════════
# Feature Engineering State (saved with model)
# ═══════════════════════════════════════════════════════
class FeaturePipeline:
    """
    Encapsulates the full feature engineering pipeline with saved state.
    On training: fit() computes normalization parameters from training data.
    On inference: transform() applies saved parameters to new data.
    """
    def __init__(self):
        self.feature_names: List[str] = []
        self.base_feature_names: List[str] = []
        # Per-feature cross-sectional normalization parameters
        self.cs_means: Optional[np.ndarray] = None   # (n_features,) — mean per feature
        self.cs_stds: Optional[np.ndarray] = None    # (n_features,) — std per feature
        # For momentum rank, we need trailing price history
        self.momentum_periods = [5, 21, 63, 126]
        # Amihud illiquidity lookback
        self.illiq_lookback = 21
        # Feature order must be EXACTLY preserved for the Ridge coefficient vector
        
    def fit(self, X_base: np.ndarray, n_stocks: int):
        """
        Compute normalization parameters from training data.
        X_base: (n_days, n_stocks, n_features) — base OHLCV features.
        """
        D, S, F = X_base.shape
        self.base_feature_names = [f"base_{i}" for i in range(F)]
        
        # Cross-sectional means/stds (per feature, computed across all days+stocks)
        X_flat = X_base.reshape(-1, F)
        self.cs_means = np.nanmean(X_flat, axis=0)
        self.cs_stds = np.nanstd(X_flat, axis=0)
        self.cs_stds = np.maximum(self.cs_stds, 1e-8)
        
        # Build full feature name list
        names = list(self.base_feature_names)
        for f in range(F): names.append(f"cs_z_{f}")
        for f in range(F): names.append(f"cs_rank_{f}")
        names.extend(["inter_rsi_volratio", "inter_sma5_rsi", "inter_bb_rsi"])
        for p in self.momentum_periods: names.append(f"mom_rank_{p}d")
        names.append("amihud_illiq")
        self.feature_names = names
        logger.info(f"FeaturePipeline fitted: {len(names)} features")
        
    def transform(self, X_base: np.ndarray, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Apply saved feature engineering to new data.
        X_base: (n_days, n_stocks, n_features)
        prices: (n_days, n_stocks)
        volumes: (n_days, n_stocks)
        Returns: (n_days, n_stocks, n_total_features)
        """
        D, S, F = X_base.shape
        blocks = [X_base]
        
        # Cross-sectional z-scores (using SAVED means/stds)
        if self.cs_means is not None:
            for f in range(F):
                fd = X_base[:, :, f]
                z = np.nan_to_num((fd - self.cs_means[f]) / self.cs_stds[f], 0)
                blocks.append(z[:, :, np.newaxis])
        else:
            for f in range(F):
                fd = X_base[:, :, f]
                m = np.nanmean(fd, axis=1, keepdims=True)
                s = np.maximum(np.nanstd(fd, axis=1, keepdims=True), 1e-8)
                blocks.append(np.nan_to_num((fd - m) / s, 0)[:, :, np.newaxis])
        
        # Cross-sectional percentile ranks
        for f in range(F):
            fd = X_base[:, :, f]; r = np.zeros_like(fd)
            for d in range(D):
                v = fd[d]; ok = ~np.isnan(v)
                if ok.sum() > 1: r[d, ok] = pd.Series(v[ok]).rank(pct=True).values
            blocks.append(r[:, :, np.newaxis])
        
        # Pairwise interactions
        for a, b in [(4,5), (2,4), (6,4)]:
            blocks.append((X_base[:,:,a] * X_base[:,:,b])[:,:,np.newaxis])
        
        # Momentum ranks
        rets_1d = np.zeros_like(prices); rets_1d[1:] = prices[1:]/prices[:-1]-1
        for period in self.momentum_periods:
            mom = np.zeros_like(prices)
            mom[period:] = prices[period:]/prices[:-period]-1
            mr = np.zeros_like(mom)
            for d in range(period, D):
                v = mom[d]; ok = ~np.isnan(v)
                if ok.sum() > 1: mr[d, ok] = pd.Series(v[ok]).rank(pct=True).values
            blocks.append(mr[:,:,np.newaxis])
        
        # Amihud illiquidity
        dv = prices * np.maximum(volumes, 1.0)
        illiq = np.abs(rets_1d) / np.maximum(dv, 1e-12)
        illiq_ma = np.full_like(illiq, np.nan)
        for d in range(self.illiq_lookback, D):
            illiq_ma[d] = np.nanmedian(illiq[d-self.illiq_lookback+1:d+1], axis=0)
        illiq_r = np.zeros_like(illiq_ma)
        for d in range(self.illiq_lookback, D):
            v = illiq_ma[d]; ok = ~np.isnan(v)
            if ok.sum() > 1: illiq_r[d,ok] = pd.Series(v[ok]).rank(pct=True).values
        blocks.append(illiq_r[:,:,np.newaxis])
        
        return np.concatenate(blocks, axis=2).astype(np.float32)
    
    def save(self, path: str):
        state = {"feature_names": self.feature_names,
                 "base_feature_names": self.base_feature_names,
                 "cs_means": self.cs_means.tolist() if self.cs_means is not None else None,
                 "cs_stds": self.cs_stds.tolist() if self.cs_stds is not None else None,
                 "momentum_periods": self.momentum_periods,
                 "illiq_lookback": self.illiq_lookback}
        with open(path, "w") as f: json.dump(state, f, indent=2)
        logger.info(f"FeaturePipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        fp = cls()
        with open(path) as f: state = json.load(f)
        fp.feature_names = state["feature_names"]
        fp.base_feature_names = state["base_feature_names"]
        fp.cs_means = np.array(state["cs_means"]) if state["cs_means"] else None
        fp.cs_stds = np.array(state["cs_stds"]) if state["cs_stds"] else None
        fp.momentum_periods = state["momentum_periods"]
        fp.illiq_lookback = state["illiq_lookback"]
        return fp


# ═══════════════════════════════════════════════════════
# Main Training Pipeline
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/data/us_stocks_full")
    parser.add_argument("--model-dir", default="/root/models/production")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    t_start = time.time()
    results = {}
    
    # ── 1. Load Data ──
    logger.info("="*60)
    logger.info("STEP 1: Load & Filter Data")
    logger.info("="*60)
    files = sorted(Path(args.data_dir).glob("*.csv"))
    stocks = {}
    for fp in files:
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if len(df) >= MIN_ROWS: stocks[fp.stem] = df
    logger.info(f"Loaded {len(stocks)} stocks")
    
    # ── 2. Base Features ──
    logger.info("STEP 2: Compute Base OHLCV Features")
    feats = {}
    for i, (sym, df) in enumerate(stocks.items()):
        feats[sym] = compute_all_features(df)
        if (i+1) % 150 == 0: logger.info(f"  {i+1}/{len(stocks)}")
    
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
    
    logger.info(f"Data: {D} days × {S} stocks × {len(base_cols)} features")
    logger.info(f"Dates: {common[0].date()} ~ {common[-1].date()}")
    
    # ── 3. Fit Feature Pipeline ──
    logger.info("STEP 3: Fit Feature Pipeline (save normalization state)")
    feature_pipeline = FeaturePipeline()
    feature_pipeline.fit(X_base, S)
    X = feature_pipeline.transform(X_base, prices, volumes)
    n_feat = X.shape[2]
    logger.info(f"Total features: {n_feat}")
    
    # Save feature pipeline
    feature_pipeline.save(str(model_dir / "feature_pipeline.json"))
    
    # ── 4. Forward Returns ──
    rets = np.full((D, S), np.nan, dtype=np.float32)
    rets[:-FORWARD_HORIZON] = prices[FORWARD_HORIZON:] / prices[:-FORWARD_HORIZON] - 1
    
    # ── 5. Walk-Forward Training ──
    logger.info("STEP 4: Walk-Forward Training (saving models)")
    
    windows = []
    start = 0
    while start + WF_TRAIN + WF_PURGE + WF_VAL <= D:
        windows.append((start, start+WF_TRAIN, start+WF_TRAIN+WF_PURGE, start+WF_TRAIN+WF_PURGE+WF_VAL))
        start += WF_STEP
    logger.info(f"Walk-forward: {len(windows)} windows")
    
    X_flat = X.reshape(-1, n_feat)
    y_flat = rets.reshape(-1)
    
    wf_models = []
    for wi, (t0, t1, t2, t3) in enumerate(windows):
        tr_idx = np.arange(t0*S, t1*S)
        vl_idx = np.arange(t2*S, t3*S)
        
        Xtr, ytr = X_flat[tr_idx], y_flat[tr_idx]
        Xvl, yvl = X_flat[vl_idx], y_flat[vl_idx]
        
        m_tr = ~np.isnan(ytr) & ~np.isnan(Xtr).any(axis=1)
        m_vl = ~np.isnan(yvl)
        Xvl_c = np.nan_to_num(Xvl, 0)
        
        if m_tr.sum() < 200: continue
        
        # Train Ridge
        model = Ridge(alpha=1.0, random_state=args.seed)
        model.fit(Xtr[m_tr], ytr[m_tr])
        y_pred = model.predict(Xvl_c[m_vl])
        
        val_ic = float(spearmanr(yvl[m_vl], y_pred)[0]) if m_vl.sum() >= 10 else 0.0
        
        # Compute strategy Sharpe
        cutoff = max(int(m_vl.sum() * 0.2), 1)
        order = np.argsort(y_pred)
        yv = yvl[m_vl]
        strat_ret = np.concatenate([-yv[order[:cutoff]], yv[order[-cutoff:]]])
        val_sharpe = float(np.mean(strat_ret) / max(np.std(strat_ret), 1e-10)) * np.sqrt(252)
        
        # Save model for this window
        win_model_path = model_dir / f"ridge_w{wi:02d}.joblib"
        joblib.dump(model, win_model_path)
        
        meta = {"window": wi, "train_start": str(common[t0].date()), "train_end": str(common[t1].date()),
                "val_start": str(common[t2].date()), "val_end": str(common[t3].date()),
                "val_ic": val_ic, "val_sharpe": val_sharpe, "coef_norm": float(np.linalg.norm(model.coef_))}
        wf_models.append(meta)
        
        logger.info(f"  w{wi:02d}: IC={val_ic:+.4f}  Sharpe={val_sharpe:+.2f}  |  saved to {win_model_path.name}")
        
        # Also train and save LGB if available
        if HAS_LGB:
            lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
            lgb_model.fit(Xtr[m_tr], ytr[m_tr])
            lgb_path = model_dir / f"lgb_w{wi:02d}.joblib"
            joblib.dump(lgb_model, lgb_path)
    
    # ── 6. Export Production Model (last window) ──
    logger.info("STEP 5: Export Production Model")
    
    prod_ridge = model_dir / "production_ridge.joblib"
    prod_lgb = model_dir / "production_lgb.joblib"
    prod_meta = model_dir / "production_metadata.json"
    
    # Copy last window's model as production
    last_w = len(wf_models) - 1
    import shutil
    shutil.copy(model_dir / f"ridge_w{last_w:02d}.joblib", prod_ridge)
    if HAS_LGB:
        shutil.copy(model_dir / f"lgb_w{last_w:02d}.joblib", prod_lgb)
    
    # Production metadata
    prod_meta_data = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_stocks": S,
        "n_features": n_feat,
        "feature_names": feature_pipeline.feature_names,
        "symbols": symbols,
        "forward_horizon_days": FORWARD_HORIZON,
        "model_type": "Ridge(alpha=1.0)",
        "last_window": last_w,
        "last_train_end": str(common[windows[last_w][1]].date()),
        "last_val_end": str(common[windows[last_w][3]].date()),
        "walk_forward_windows": wf_models,
        "feature_pipeline": "feature_pipeline.json",
    }
    with open(prod_meta, "w") as f:
        json.dump(prod_meta_data, f, indent=2, default=str)
    
    # ── 7. Summary ──
    elapsed = time.time() - t_start
    all_ics = [w["val_ic"] for w in wf_models]
    
    print("\n" + "="*70)
    print("  PRODUCTION MODEL TRAINED")
    print("="*70)
    print(f"  Data: {S} stocks × {D} days")
    print(f"  Features: {n_feat}")
    print(f"  WF Windows: {len(wf_models)}")
    print(f"  IC mean: {np.mean(all_ics):.4f} ± {np.std(all_ics):.4f}")
    print(f"  All windows positive: {all(ic>0 for ic in all_ics)}")
    print(f"")
    print(f"  Production model:  {prod_ridge}")
    print(f"  Feature pipeline:  {model_dir / 'feature_pipeline.json'}")
    print(f"  Metadata:          {prod_meta}")
    print(f"  Runtime: {elapsed:.0f}s")
    print("="*70)

if __name__ == "__main__":
    main()
