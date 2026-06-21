"""
Build clean 200-stock A-share dataset from raw Tencent CSVs.

Strategy:
1. Forward-adjust (前复权) for real stock splits (return > 50%)
2. Clip any remaining extreme prices/returns
3. Ensure all stocks share common trading dates

This produces a CONSISTENT dataset where all 200 stocks can be used
in multi-asset training without corporate-action artifacts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPLIT_THRESHOLD = 0.50  # only real splits (>50% = 2:1 or more)
MIN_PRICE = 0.01
MAX_DAILY_RETURN = 0.11  # A-share limit 10% + small buffer


def detect_corporate_actions(closes: np.ndarray) -> list[int]:
    """Detect real stock split/reverse-split indices (return > threshold)."""
    prev = np.maximum(closes[:-1], MIN_PRICE)
    returns = closes[1:] / prev - 1.0
    # A real split typically has a return < -0.5 (price halves or worse)
    # We also catch reverse splits (return > 1.0, price doubles)
    bad = np.where(np.abs(returns) > SPLIT_THRESHOLD)[0]
    return [int(i) + 1 for i in bad]


def forward_adjust_one(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-adjust a single stock DataFrame, then clip residuals."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)
    indices = detect_corporate_actions(closes)
    if not indices:
        return df

    price_cols = ["open", "high", "low", "close"]
    for idx in reversed(indices):
        c_prev = max(closes[idx - 1], MIN_PRICE)
        c_curr = max(closes[idx], MIN_PRICE)
        factor = c_curr / c_prev
        if abs(factor - 1.0) < 1e-10:
            continue  # no adjustment needed
        inv = 1.0 / factor
        for col in price_cols:
            arr = df.loc[: idx - 1, col].astype(np.float64).values
            arr = np.maximum(arr * inv, MIN_PRICE)
            df.loc[: idx - 1, col] = arr
        if "volume" in df.columns:
            v = df.loc[: idx - 1, "volume"].astype(np.float64).values
            df.loc[: idx - 1, "volume"] = np.maximum(v * factor, 0).astype(np.int64)

    # Final sanity: clip prices
    for col in price_cols:
        df[col] = df[col].clip(lower=MIN_PRICE)

    return df


def clip_extreme_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Clip daily returns to ±MAX_DAILY_RETURN by adjusting close prices."""
    closes = df["close"].values.astype(np.float64)
    for i in range(1, len(closes)):
        ret = closes[i] / max(closes[i - 1], MIN_PRICE) - 1.0
        if ret > MAX_DAILY_RETURN:
            closes[i] = closes[i - 1] * (1.0 + MAX_DAILY_RETURN)
        elif ret < -MAX_DAILY_RETURN:
            closes[i] = closes[i - 1] * (1.0 - MAX_DAILY_RETURN)

    df["close"] = closes

    # Also adjust open/high/low proportionally to keep OHLC consistency
    for col in ["open", "high", "low"]:
        vals = df[col].values.astype(np.float64)
        ratio = closes / np.maximum(df["close"].values, MIN_PRICE)
        # Only adjust where close was modified
        for i in range(1, len(vals)):
            orig_ret = df["close"].values[i] / max(df["close"].values[i - 1], MIN_PRICE) - 1.0
            new_ret = closes[i] / max(closes[i - 1], MIN_PRICE) - 1.0
            if abs(orig_ret) > MAX_DAILY_RETURN:
                vals[i] = vals[i] * ((1.0 + new_ret) / (1.0 + orig_ret))
        df[col] = np.maximum(vals, MIN_PRICE)

    return df


def process_stock(input_path: Path, output_dir: Path) -> tuple[int, int, int]:
    """Process a single stock: load, adjust, clip, save. Returns (rows, n_splits, n_clipped)."""
    df = pd.read_csv(input_path)
    n_before = len(df)

    # Step 1: forward-adjust for real splits
    df = forward_adjust_one(df)
    n_splits = len(detect_corporate_actions(
        pd.read_csv(input_path)["close"].values.astype(np.float64)
    ))

    # Step 2: clip extreme returns
    ret_before = df["close"].pct_change()
    n_extreme = int((ret_before.abs() > MAX_DAILY_RETURN).sum())
    df = clip_extreme_returns(df)

    # Save
    df.to_csv(output_dir / input_path.name, index=False)
    return n_before, n_splits, n_extreme


def main():
    parser = argparse.ArgumentParser(description="Build clean 200-stock A-share dataset")
    parser.add_argument("--input-dir", default="data/tencent", help="Raw Tencent CSV dir")
    parser.add_argument("--output-dir", default="data/tencent_clean", help="Output dir")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    log.info("Processing %d stocks from %s", len(csv_files), input_dir)

    total_splits = 0
    total_clipped = 0
    for f in csv_files:
        rows, n_splits, n_clipped = process_stock(f, output_dir)
        total_splits += n_splits
        total_clipped += n_clipped
        if n_splits > 0 or n_clipped > 0:
            log.info("  %s: %d rows, %d splits, %d clipped", f.stem, rows, n_splits, n_clipped)

    # Final quality check
    bad = 0
    perfect = 0
    for f in sorted(output_dir.glob("*.csv")):
        df = pd.read_csv(f)
        ret = df["close"].pct_change().dropna()
        extreme = (ret.abs() > MAX_DAILY_RETURN).sum()
        has_zero = (df["close"].min() <= 0)
        if extreme > 0 or has_zero:
            bad += 1
        else:
            perfect += 1

    log.info("=" * 50)
    log.info("Dataset build complete!")
    log.info("  Stocks processed : %d", len(csv_files))
    log.info("  Splits corrected : %d", total_splits)
    log.info("  Returns clipped  : %d", total_clipped)
    log.info("  Perfect stocks   : %d (no extremes, no zero prices)", perfect)
    log.info("  Stocks w/ issues : %d", bad)
    log.info("  Output dir       : %s", output_dir.resolve())
    log.info("=" * 50)


if __name__ == "__main__":
    main()
