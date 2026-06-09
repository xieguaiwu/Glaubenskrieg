#!/usr/bin/env python3
"""
Glaubenskrieg Production Inference — Load trained model and generate signals.

Usage:
  python scripts/infer_production.py \
    --model-dir /root/models/production \
    --data-dir /root/data/us_stocks_full \
    --output signals.csv

Or for single-stock prediction from latest market data:
  python scripts/infer_production.py \
    --model-dir /root/models/production \
    --prices AAPL,150.23,148.90,149.50,... (comma-separated close prices, oldest first)
"""
import sys, os, json, time, argparse, warnings, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("infer")

PROJECT = Path("/root/Glaubenskrieg")
sys.path.insert(0, str(PROJECT))
from src.data.features import compute_all_features
from scripts.train_production import FeaturePipeline

# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════
MIN_PRICE_HISTORY = 130  # Need at least 126 days for momentum + 5 for forward return
FORWARD_HORIZON = 5

# ═══════════════════════════════════════════════════════
# Inference Engine
# ═══════════════════════════════════════════════════════
class GlaubenskriegPredictor:
    """
    Production predictor: loads model + feature pipeline, generates signals.
    
    Usage:
        predictor = GlaubenskriegPredictor("/root/models/production")
        signals = predictor.predict_from_files("/root/data/us_stocks_full")
        # signals: DataFrame with columns [symbol, signal, predicted_return, confidence]
    """
    
    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        
        # Load feature pipeline
        fp_path = model_dir / "feature_pipeline.json"
        if not fp_path.exists():
            raise FileNotFoundError(f"Feature pipeline not found: {fp_path}")
        self.feature_pipeline = FeaturePipeline.load(str(fp_path))
        logger.info(f"Loaded feature pipeline: {len(self.feature_pipeline.feature_names)} features")
        
        # Load Ridge model (primary)
        ridge_path = model_dir / "production_ridge.joblib"
        if not ridge_path.exists():
            raise FileNotFoundError(f"Ridge model not found: {ridge_path}")
        self.ridge_model = joblib.load(ridge_path)
        logger.info(f"Loaded Ridge model: {self.ridge_model.coef_.shape[0]} coefficients")
        
        # Load LGB model (optional ensemble)
        lgb_path = model_dir / "production_lgb.joblib"
        self.lgb_model = joblib.load(lgb_path) if lgb_path.exists() else None
        if self.lgb_model:
            logger.info("Loaded LGB model (ensemble mode)")
        
        # Load metadata
        meta_path = model_dir / "production_metadata.json"
        self.metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
        
        self.n_features = len(self.feature_pipeline.feature_names)
        self.symbols = self.metadata.get("symbols", [])
    
    def predict_from_files(self, data_dir: str) -> pd.DataFrame:
        """
        Generate signals for all stocks in a data directory.
        
        Returns DataFrame with columns: symbol, signal, ridge_return, lgb_return, ensemble_return
        """
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob("*.csv"))
        
        signals = []
        for fp in files:
            sym = fp.stem
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if len(df) < MIN_PRICE_HISTORY:
                continue
            
            try:
                result = self._predict_single(df, sym)
                if result is not None:
                    signals.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict {sym}: {e}")
        
        df_out = pd.DataFrame(signals)
        if not df_out.empty:
            # Cross-sectional z-score of the ensemble signal (for ranking)
            df_out["cs_zscore"] = (df_out["ensemble_return"] - df_out["ensemble_return"].mean()) / max(df_out["ensemble_return"].std(), 1e-8)
            df_out = df_out.sort_values("ensemble_return", ascending=False)
        return df_out
    
    def predict_batch(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate signals for a batch of stocks passed as DataFrames."""
        signals = []
        for sym, df in stock_data.items():
            result = self._predict_single(df, sym)
            if result is not None:
                signals.append(result)
        return pd.DataFrame(signals).sort_values("ensemble_return", ascending=False)
    
    def _predict_single(self, df: pd.DataFrame, symbol: str) -> Optional[dict]:
        """
        Predict forward return for a single stock.
        
        Steps:
        1. Compute base OHLCV features from price history
        2. Apply saved feature engineering pipeline
        3. Ridge prediction → expected 5d forward return
        4. (Optional) LGB prediction → ensemble
        """
        if len(df) < MIN_PRICE_HISTORY:
            return None
        
        # 1. Base features (uses the FULL history to compute indicators)
        base_feats = compute_all_features(df)
        
        # We need the LATEST day's features in the right shape
        # compute_all_features returns (n_days, 9) — we take all days
        n_days = len(base_feats)
        base_arr = base_feats.values.astype(np.float32)  # (n_days, 9)
        
        # Reshape to (1, n_stocks=1, n_days, 9) for feature pipeline
        X_base = base_arr[np.newaxis, np.newaxis, :, :]  # (1, 1, n_days, 9)
        
        prices = df["close"].values.astype(np.float32)[np.newaxis, np.newaxis, :]  # (1, 1, n_days)
        volumes = df["volume"].values.astype(np.float32)[np.newaxis, np.newaxis, :] if "volume" in df.columns else np.ones_like(prices)
        
        # 2. Enhanced features
        X_enhanced = self.feature_pipeline.transform(
            X_base[0], prices[0], volumes[0]
        )  # (n_days, 1, n_total_features)
        
        # Take the LAST day's feature vector
        x_today = X_enhanced[-1, 0, :]  # (n_total_features,)
        
        if self.n_features != len(x_today):
            logger.warning(f"Feature mismatch: model expects {self.n_features}, got {len(x_today)}")
            return None
        
        # Handle NaN (can happen if not enough history for some indicators)
        if np.any(np.isnan(x_today)) or np.any(np.isinf(x_today)):
            x_today = np.nan_to_num(x_today, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Predict
        ridge_pred = float(self.ridge_model.predict(x_today.reshape(1, -1))[0])
        lgb_pred = float(self.lgb_model.predict(x_today.reshape(1, -1))[0]) if self.lgb_model else ridge_pred
        
        ensemble_pred = (ridge_pred + lgb_pred) / 2 if self.lgb_model else ridge_pred
        
        # 4. Confidence: based on feature validity (more valid features = higher confidence)
        valid_frac = 1.0 - np.mean(np.isnan(x_today))
        
        return {
            "symbol": symbol,
            "date": str(df.index[-1].date()),
            "ridge_return": round(ridge_pred, 6),
            "lgb_return": round(lgb_pred, 6) if self.lgb_model else None,
            "ensemble_return": round(ensemble_pred, 6),
            "confidence": round(valid_frac, 3),
            "n_days_history": n_days,
        }
    
    def get_top_picks(self, data_dir: str, top_n: int = 20) -> pd.DataFrame:
        """Convenience: get top-N long picks."""
        df = self.predict_from_files(data_dir)
        if df.empty: return df
        return df.head(top_n)
    
    def get_portfolio_weights(self, data_dir: str, top_n: int = 100, 
                               weighting: str = "equal") -> Dict[str, float]:
        """
        Generate portfolio weights.
        
        weighting: "equal" → equal weight top N
                   "inverse_vol" → inverse volatility weight top N
                   "signal" → proportional to ensemble signal
        """
        df = self.predict_from_files(data_dir)
        if df.empty: return {}
        
        top = df.head(top_n)
        if weighting == "equal":
            w = 1.0 / len(top)
            return {r["symbol"]: w for _, r in top.iterrows()}
        elif weighting == "signal":
            signals = top["ensemble_return"].values
            signals = signals - signals.min() + 0.01
            w = signals / signals.sum()
            return {r["symbol"]: float(w[i]) for i, (_, r) in enumerate(top.iterrows())}
        else:
            return {r["symbol"]: 1.0/len(top) for _, r in top.iterrows()}


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Glaubenskrieg Production Inference")
    parser.add_argument("--model-dir", default="/root/models/production",
                        help="Directory containing production model files")
    parser.add_argument("--data-dir", default=None,
                        help="Directory of stock CSV files to predict")
    parser.add_argument("--output", default="signals.csv",
                        help="Output CSV for predictions")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top picks to display")
    args = parser.parse_args()
    
    predictor = GlaubenskriegPredictor(args.model_dir)
    
    if args.data_dir:
        df = predictor.predict_from_files(args.data_dir)
        df.to_csv(args.output, index=False)
        
        print(f"\n{'='*60}")
        print(f"  Glaubenskrieg Signals — {len(df)} stocks")
        print(f"{'='*60}")
        print(f"\n  Top {args.top_n} Long Picks:")
        print(f"  {'Symbol':<10} {'Ensemble':>10} {'Ridge':>10} {'CS Z':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for _, row in df.head(args.top_n).iterrows():
            print(f"  {row['symbol']:<10} {row['ensemble_return']:>10.4f} {row['ridge_return']:>10.4f} {row.get('cs_zscore',0):>8.2f}")
        
        print(f"\n  Bottom {min(args.top_n, 5)} Short Picks:")
        for _, row in df.tail(min(args.top_n, 5)).iterrows():
            print(f"  {row['symbol']:<10} {row['ensemble_return']:>10.4f}")
        
        print(f"\n  Full signals: {args.output}")
        print(f"{'='*60}")
    else:
        print("Usage: provide --data-dir for batch prediction")
        print(f"Model loaded: {len(predictor.symbols)} known symbols, {predictor.n_features} features")

if __name__ == "__main__":
    main()
