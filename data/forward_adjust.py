"""
Forward-adjust (前复权) OHLCV data from Tencent Finance API.

Tencent raw data contains unadjusted prices — stock splits and rights
issues create extreme daily returns (>20%) that violate A-share ±10% limits.

This script detects corporate actions via abnormal close-price jumps
and forward-adjusts all historical prices to be continuous with today's.

Usage:
    python forward_adjust.py --input-dir data/tencent --output-dir data/tencent_adjusted
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# A-shares have ±10% daily price limit; use 15% threshold for safety
SPLIT_THRESHOLD = 0.15


def detect_splits(closes: np.ndarray, threshold: float = SPLIT_THRESHOLD) -> list[int]:
    """Detect stock split / reverse-split indices.

    Returns list of indices (1-indexed in the array) where a corporate
    action caused close price to change beyond ``threshold``.
    """
    returns = closes[1:] / closes[:-1] - 1.0
    return [i + 1 for i, r in enumerate(returns) if abs(r) > threshold]


def forward_adjust(df: pd.DataFrame, threshold: float = SPLIT_THRESHOLD) -> pd.DataFrame:
    """Forward-adjust (前复权) OHLCV data in-place.

    Detects splits by abnormal close jumps and adjusts all prices
    BEFORE each event so the series is continuous.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)

    indices = detect_splits(closes, threshold=threshold)
    if not indices:
        return df  # no adjustment needed

    price_cols = ["open", "high", "low", "close"]
    volume_col = "volume"

    for idx in reversed(indices):
        factor = closes[idx] / closes[idx - 1]  # e.g. 0.5 for 2:1 split
        # Adjust all rows before idx
        for col in price_cols:
            df.loc[: idx - 1, col] = (df.loc[: idx - 1, col].astype(np.float64) * (1.0 / factor)).astype(np.float64)
        if volume_col in df.columns:
            df.loc[: idx - 1, volume_col] = (df.loc[: idx - 1, volume_col].astype(np.float64) * factor).astype(np.float64)

    return df


def verify_adjustment(df: pd.DataFrame, label: str = "") -> list[str]:
    """Verify no extreme returns remain after adjustment."""
    ret = df["close"].pct_change().dropna()
    extreme = ret[ret.abs() > SPLIT_THRESHOLD]
    issues: list[str] = []
    if not extreme.empty:
        for idx in extreme.index:
            issues.append(f"  {ret.loc[idx]:+.4f} at {df.loc[idx, 'date']}")
    return issues


def process_stock(
    input_path: Path, output_dir: Path, threshold: float = SPLIT_THRESHOLD
) -> tuple[int, int]:
    """Forward-adjust a single stock CSV. Returns (rows, n_splits)."""
    df = pd.read_csv(input_path)
    n_before = len(df)

    adjusted = forward_adjust(df, threshold=threshold)
    issues = verify_adjustment(adjusted)

    output_path = output_dir / input_path.name
    adjusted.to_csv(output_path, index=False)

    n_splits = len(detect_splits(df["close"].values, threshold=threshold))
    return n_before, n_splits, issues


def main():
    parser = argparse.ArgumentParser(description="Forward-adjust Tencent OHLCV data")
    parser.add_argument("--input-dir", default="data/tencent", help="Input CSV directory")
    parser.add_argument("--output-dir", default="data/tencent_adjusted", help="Output CSV directory")
    parser.add_argument("--threshold", type=float, default=SPLIT_THRESHOLD, help="Split detection threshold")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    log.info("Processing %d CSV files from %s", len(csv_files), input_dir)

    total_rows = 0
    total_splits = 0
    total_issues = 0
    stock_issues: list[tuple[str, list[str]]] = []

    for f in csv_files:
        symbol = f.stem
        rows, n_splits, issues = process_stock(f, output_dir, threshold=args.threshold)
        total_rows += rows
        total_splits += n_splits
        if issues:
            total_issues += len(issues)
            stock_issues.append((symbol, issues))
            if len(stock_issues) <= 5:
                for iss in issues:
                    log.warning("  %s: %s", symbol, iss)

        if n_splits > 0:
            log.info("  %s: %d rows, %d split(s) corrected", symbol, rows, n_splits)

    log.info("")
    log.info("=" * 50)
    log.info("Forward adjustment complete!")
    log.info("  Stocks processed : %d", len(csv_files))
    log.info("  Total rows      : %d", total_rows)
    log.info("  Splits corrected: %d", total_splits)
    log.info("  Residual issues : %d", total_issues)
    log.info("  Output dir      : %s", output_dir.resolve())
    log.info("=" * 50)

    if stock_issues:
        log.warning("Stocks with residual extreme returns (>%.0f%%):", args.threshold * 100)
        for sym, issues in stock_issues:
            log.warning("  %s: %d issues", sym, len(issues))
            for iss in issues[:3]:
                log.warning("    %s", iss)
            if len(issues) > 3:
                log.warning("    ... and %d more", len(issues) - 3)


if __name__ == "__main__":
    main()
