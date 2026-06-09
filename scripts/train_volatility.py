#!/usr/bin/env python3
"""Standalone volatility prediction using LightGBM on feature-engineered data.

Predicts realized_vol_21 (rolling 21-day std of returns) from all other technical
indicators. Same walk-forward / tabular format as train_fe_lgb.py.
Metrics: MSE, MAE, QLIKE (Patton 2011).

Usage:
    PYTHONPATH=. python scripts/train_volatility.py --data-dir /path/to/csvs --output results/vol.json
"""

import argparse, json, logging, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.features import compute_all_features
from src.data.dataset import create_sequences

try:
    import lightgbm as lgb; HAS_LGB = True
except ImportError:
    HAS_LGB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET = "realized_vol_21"


# ── Data ──────────────────────────────────────────────────────────

def load_data(data_dir, n_assets=50, seq_len=63, train_frac=0.7, val_frac=0.15):
    """Load CSVs → compute_all_features → (B,N,T,D) data, (B,T,N) targets."""
    csv_files = sorted(Path(data_dir).glob("*.csv"))[:n_assets]
    logger.info("Loading %d assets", len(csv_files))

    feature_dfs = []
    for fp in csv_files:
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if df.empty or "close" not in df.columns:
            continue
        feats = compute_all_features(df).dropna()
        if len(feats) >= seq_len + 10:
            feature_dfs.append(feats)

    N = len(feature_dfs)
    logger.info("%d assets after filter", N)

    common_idx = feature_dfs[0].index
    for fdf in feature_dfs[1:]:
        common_idx = common_idx.intersection(fdf.index)
    logger.info("Common dates: %d", len(common_idx))

    feat_cols = [c for c in feature_dfs[0].columns if c != TARGET]

    all_data, all_target = [], []
    for fdf in feature_dfs:
        a = fdf.loc[common_idx]
        all_data.append(create_sequences(a[feat_cols].values.astype(np.float32), seq_len))
        all_target.append(create_sequences(a[TARGET].values.astype(np.float32), seq_len))

    data_seq = np.stack(all_data, axis=1).astype(np.float32)          # (B,N,T,D)
    targ_seq = np.stack(all_target, axis=1).astype(np.float32).transpose(0, 2, 1)  # (B,T,N)

    B = len(data_seq)
    n_tr = int(B * train_frac); n_vl = int(B * val_frac)
    logger.info("Batches: %d total | train=%d val=%d test=%d", B, n_tr, n_vl, B - n_tr - n_vl)
    return (data_seq[:n_tr], targ_seq[:n_tr],
            data_seq[n_tr:n_tr+n_vl], targ_seq[n_tr:n_tr+n_vl],
            data_seq[n_tr+n_vl:], targ_seq[n_tr+n_vl:])


def to_tabular(data_b, targ_b):
    """(B,N,T,D)→(B*N,D) at last timestep, (B,T,N)→(B*N,)."""
    B, N, T, D = data_b.shape
    return (data_b[:, :, -1, :].reshape(B * N, D).astype(np.float32),
            targ_b[:, -1, :].reshape(B * N).astype(np.float32))


# ── Model ─────────────────────────────────────────────────────────

def train_lgb(X_tr, y_tr, X_val, y_val, seed=42):
    m = lgb.LGBMRegressor(
        objective="regression", metric="l1", boosting_type="gbdt",
        num_leaves=31, max_depth=6, learning_rate=0.05, n_estimators=500,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        min_child_samples=20, verbosity=-1, random_state=seed, n_jobs=-1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="l1",
          callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
    return m


# ── Metrics ───────────────────────────────────────────────────────

def qlike(y_true, y_pred):
    """QLIKE = log(h) + σ²/h  where h=y_pred², σ²=y_true².  Lower is better."""
    h = np.maximum(y_pred ** 2, 1e-8)
    return float(np.mean(np.log(h) + y_true ** 2 / h))


def vol_metrics(y_true, y_pred):
    v = np.isfinite(y_pred) & np.isfinite(y_true)
    if v.sum() < 10:
        return {"mse": 0.0, "mae": 0.0, "qlike": 0.0, "n": int(v.sum())}
    yt, yp = y_true[v], y_pred[v]
    return {"mse": float(np.mean((yt - yp) ** 2)),
            "mae": float(np.mean(np.abs(yt - yp))),
            "qlike": qlike(yt, yp), "n": int(v.sum())}


# ── Walk-forward ──────────────────────────────────────────────────

def walk_forward(data_b, targ_b, train_b=400, val_b=80, step_b=63, purge_b=126, seed=42):
    B = len(data_b)
    results, all_p, all_t = [], [], []
    pos = 0
    while pos + train_b + purge_b + val_b <= B:
        te = pos + train_b; vs = te + purge_b; ve = vs + val_b
        wX, wy = to_tabular(data_b[pos:te], targ_b[pos:te])
        vX, vy = to_tabular(data_b[vs:ve], targ_b[vs:ve])
        if len(wX) < 100 or len(vX) < 50:
            pos += step_b; continue
        m = train_lgb(wX, wy, vX, vy, seed)
        pred = m.predict(vX)
        met = vol_metrics(vy, pred)
        results.append({"window": len(results), **met})
        all_p.extend(pred); all_t.extend(vy)
        logger.info("  w%d MSE=%.6f MAE=%.6f QLIKE=%.4f",
                    len(results)-1, met["mse"], met["mae"], met["qlike"])
        pos += step_b
    if not results:
        return {"n_windows": 0, "error": "No windows fit"}
    ap, at = np.array(all_p), np.array(all_t)
    om = vol_metrics(at, ap)
    mses, maes, qls = [r["mse"] for r in results], [r["mae"] for r in results], [r["qlike"] for r in results]
    return {"n_windows": len(results),
            "mean_mse": float(np.mean(mses)), "std_mse": float(np.std(mses)),
            "mean_mae": float(np.mean(maes)), "std_mae": float(np.std(maes)),
            "mean_qlike": float(np.mean(qls)), "std_qlike": float(np.std(qls)),
            "overall_mse": om["mse"], "overall_mae": om["mae"], "overall_qlike": om["qlike"],
            "per_window": results}


# ── Main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-assets", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if not HAS_LGB:
        logger.error("pip install lightgbm"); sys.exit(1)
    np.random.seed(args.seed)
    t0 = time.time()

    tr_d, tr_t, vl_d, vl_t, te_d, te_t = load_data(args.data_dir, args.n_assets)

    logger.info("=== Walk-Forward ===")
    wf = walk_forward(tr_d, tr_t, seed=args.seed)

    logger.info("=== Test ===")
    trX, try_ = to_tabular(tr_d, tr_t)
    vlX, vly = to_tabular(vl_d, vl_t)
    teX, tey = to_tabular(te_d, te_t)
    fm = train_lgb(trX, try_, vlX, vly, args.seed)
    te_pred = fm.predict(teX)
    te_met = vol_metrics(tey, te_pred)
    logger.info("Test: MSE=%.6f MAE=%.6f QLIKE=%.4f",
                te_met["mse"], te_met["mae"], te_met["qlike"])

    out = {"config": {"data_dir": args.data_dir, "n_assets": args.n_assets,
                      "seed": args.seed, "target": TARGET, "model": "LightGBM/LGBMRegressor"},
           "walk_forward": wf, "test_metrics": te_met, "runtime_s": round(time.time() - t0, 2)}

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2, default=str)

    print(f"\n{'='*60}\nVolatility LGB  seed={args.seed}\n{'='*60}")
    if wf.get("n_windows", 0) > 0:
        print(f"  WF ({wf['n_windows']}w): MSE={wf['mean_mse']:.6f}±{wf['std_mse']:.6f}  "
              f"MAE={wf['mean_mae']:.6f}±{wf['std_mae']:.6f}  QLIKE={wf['mean_qlike']:.4f}±{wf['std_qlike']:.4f}")
    print(f"  Test: MSE={te_met['mse']:.6f}  MAE={te_met['mae']:.6f}  QLIKE={te_met['qlike']:.4f}")


if __name__ == "__main__":
    main()
