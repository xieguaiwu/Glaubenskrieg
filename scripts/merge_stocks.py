#!/usr/bin/env python3
"""Regenerate merged CSV from individual stock files."""
import pandas as pd
from pathlib import Path

def merge(data_dir="data/us_stocks_full", output="data/us_stocks_all.csv"):
    dfs = []
    for fp in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        df["symbol"] = fp.stem
        dfs.append(df.reset_index())
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output, index=False)
    print(f"Merged {len(dfs)} stocks → {output} ({merged.shape[0]} rows)")

if __name__ == "__main__":
    merge()
