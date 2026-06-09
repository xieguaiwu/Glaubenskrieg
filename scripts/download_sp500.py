#!/usr/bin/env python3
"""
Download S&P 500 historical daily OHLCV data (2015-01-01 to 2026-06-05).

- Fetches constituent list from Wikipedia
- Downloads in batches of 50 via yfinance with proxy
- Saves individual CSVs + combined sp500_all.csv
- Reports progress and summary statistics

Usage:
    python scripts/download_sp500.py
    python scripts/download_sp500.py --start 2015-01-01 --end 2026-06-05
"""

import argparse
import csv
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-06-05"
BATCH_SIZE = 50
REQUEST_DELAY = 1.0  # seconds between batches
MAX_RETRIES = 3
RETRY_DELAY = 5.0  # seconds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "new_data" / "data" / "sp500"
ALL_CSV = PROJECT_ROOT / "new_data" / "data" / "sp500_all.csv"

# Proxy: set HTTP_PROXY/HTTPS_PROXY env vars if behind a firewall
PROXY_URL = os.environ.get("HTTP_PROXY", "")
if PROXY_URL:
    os.environ.setdefault("HTTP_PROXY", PROXY_URL)
    os.environ.setdefault("HTTPS_PROXY", PROXY_URL)
    os.environ.setdefault("http_proxy", PROXY_URL)
    os.environ.setdefault("https_proxy", PROXY_URL)


# ---------------------------------------------------------------------------
# Fetch S&P 500 constituents from Wikipedia
# ---------------------------------------------------------------------------
def fetch_sp500_tickers() -> List[str]:
    """
    Scrape the current S&P 500 constituent list from Wikipedia.
    Returns cleaned ticker symbols (e.g. BRK.B → BRK-B).
    Uses direct connection (no proxy) with a browser User-Agent,
    since Wikipedia blocks datacenter/proxy IPs.
    """
    print("[1/4] Fetching S&P 500 constituent list from Wikipedia...")

    session = requests.Session()
    session.proxies = {}  # No proxy for Wikipedia
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(WIKI_URL, timeout=30)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Retry {attempt+1}/{MAX_RETRIES} after error: {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Failed to fetch Wikipedia page: {e}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the constituent table: the wikitable whose header row contains "Symbol"
    table = None
    for t in soup.find_all("table", class_="wikitable"):
        header_row = t.find("tr")
        if header_row:
            th_texts = [th.get_text(strip=True) for th in header_row.find_all("th")]
            if th_texts and th_texts[0] == "Symbol":
                table = t
                break

    if table is None:
        raise RuntimeError("Could not find S&P 500 constituent table on Wikipedia page")

    tickers = []
    for row in table.find_all("tr")[1:]:  # skip header
        cols = row.find_all("td")
        if not cols:
            continue
        ticker = cols[0].get_text(strip=True)
        if ticker:
            # Clean: Wikipedia uses '.' in tickers, yfinance uses '-'
            ticker = ticker.replace(".", "-")
            tickers.append(ticker)

    print(f"  Found {len(tickers)} tickers")
    return tickers


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------
def download_batch(
    tickers: List[str],
    start: str,
    end: str,
    batch_idx: int,
    total_batches: int,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data for a batch of tickers using yfinance.
    Returns dict of ticker → DataFrame.
    Uses auto_adjust=True so prices are adjusted for splits/dividends.
    """
    label = f"[Batch {batch_idx+1}/{total_batches}]"
    print(f"  {label} Downloading {len(tickers)} tickers...", end=" ", flush=True)
    
    for attempt in range(MAX_RETRIES):
        try:
            data: pd.DataFrame = yf.download(
                tickers=" ".join(tickers),
                start=start,
                end=end,
                auto_adjust=True,
                threads=True,
                progress=False,
                # Proxy is picked up from env vars
            )
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"(retry {attempt+1})", end=" ", flush=True)
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n  {label} FAILED after {MAX_RETRIES} attempts: {e}")
                return {}
    
    if data is None or data.empty:
        print("no data returned")
        return {}

    # yfinance 1.3.0 always returns MultiIndex columns: (Close, AAPL), (High, AAPL), ...
    results: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            df = data.xs(ticker, axis=1, level=1, drop_level=True).copy()
        except KeyError:
            # Ticker not in returned data (delisted, etc.)
            continue

        if df.empty:
            continue

        # Standardize column names: lowercase, strip 'adj '
        col_map = {}
        for c in df.columns:
            clean = str(c).lower().replace("adj ", "").replace(" ", "_")
            col_map[c] = clean
        df.rename(columns=col_map, inplace=True)

        # Keep only standard OHLCV columns
        expected = ["open", "high", "low", "close", "volume"]
        present = [c for c in expected if c in df.columns]
        if present:
            results[ticker] = df[present]
    
    n_got = len(results)
    n_fail = len(tickers) - n_got
    status = f"{n_got} ok"
    if n_fail > 0:
        status += f", {n_fail} missing"
    print(status)
    
    return results


# ---------------------------------------------------------------------------
# Save individual CSV
# ---------------------------------------------------------------------------
def save_stock_csv(ticker: str, df: pd.DataFrame, output_dir: Path) -> str:
    """Save a single ticker's data as CSV. Returns the file path."""
    # Ensure date column is index and properly formatted
    df = df.copy()
    if df.index.name is None or df.index.name != "date":
        df.index.name = "date"
    # Keep only trading days with valid data
    df = df.dropna(how="all")
    
    filepath = output_dir / f"{ticker}.csv"
    df.to_csv(filepath, float_format="%.6f")
    return str(filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download S&P 500 historical OHLCV data")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--all-csv", default=str(ALL_CSV))
    parser.add_argument("--tickers", nargs="*", help="Specific tickers (skip Wikipedia fetch)")
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    batch_size = args.batch_size
    output_dir = Path(args.output_dir)
    all_csv = Path(args.all_csv)
    
    # Validate dates
    for d in [start, end]:
        datetime.strptime(d, "%Y-%m-%d")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    all_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # -----------------------------------------------------------------------
    # Step 1: Fetch ticker list
    # -----------------------------------------------------------------------
    if args.tickers:
        tickers = [t.replace(".", "-") for t in args.tickers]
        print(f"[1/4] Using {len(tickers)} user-specified tickers")
    else:
        tickers = fetch_sp500_tickers()
    
    if not tickers:
        print("ERROR: No tickers to download")
        sys.exit(1)
    
    # -----------------------------------------------------------------------
    # Step 2: Batch download
    # -----------------------------------------------------------------------
    print(f"\n[2/4] Downloading {len(tickers)} stocks from {start} to {end}")
    print(f"      Batch size: {batch_size}, Proxy: {PROXY_URL}")
    
    all_data: Dict[str, pd.DataFrame] = {}
    failed_tickers: List[str] = []
    
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    total_batches = len(batches)
    
    for bidx, batch in enumerate(batches):
        results = download_batch(batch, start, end, bidx, total_batches)
        
        for ticker, df in results.items():
            if df is not None and not df.empty:
                all_data[ticker] = df
            else:
                failed_tickers.append(ticker)
        
        # Record tickers that returned nothing
        for t in batch:
            if t not in results:
                failed_tickers.append(t)
        
        # Delay between batches to avoid rate limiting
        if bidx < total_batches - 1:
            time.sleep(REQUEST_DELAY)
    
    # -----------------------------------------------------------------------
    # Step 3: Save individual CSVs
    # -----------------------------------------------------------------------
    print(f"\n[3/4] Saving {len(all_data)} individual CSVs to {output_dir}")
    
    for ticker, df in all_data.items():
        save_stock_csv(ticker, df, output_dir)
    
    print(f"      Saved {len(all_data)} files")
    
    # -----------------------------------------------------------------------
    # Step 4: Save combined CSV (read from saved files for consistency)
    # -----------------------------------------------------------------------
    print(f"[4/4] Building combined CSV → {all_csv}")
    
    csv_files = sorted(output_dir.glob("*.csv"))
    combined_rows = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            continue
        if df.empty:
            continue
        ticker = f.stem
        df["symbol"] = ticker
        df = df.reset_index()
        # Standardize: first column should be 'date'
        if df.columns[0] != "date":
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        std_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in std_cols if c in df.columns]]
        df = df.dropna(subset=["date"])
        if not df.empty:
            combined_rows.append(df)
    
    if combined_rows:
        combined = pd.concat(combined_rows, axis=0, ignore_index=True)
        combined.to_csv(all_csv, index=False, float_format="%.6f")
        print(f"      Saved {len(combined):,} rows across {len(combined_rows)} symbols")
    else:
        print("      WARNING: No data to combine")
    
    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Start date:        {start}")
    print(f"  End date:          {end}")
    print(f"  Tickers requested: {len(tickers)}")
    print(f"  Tickers succeeded: {len(all_data)}")
    print(f"  Tickers failed:    {len(failed_tickers)}")
    
    if all_data:
        total_rows = sum(len(df) for df in all_data.values())
        date_ranges = [(df.index.min(), df.index.max()) for df in all_data.values()]
        all_mins = [dr[0] for dr in date_ranges if dr[0] is not None]
        all_maxs = [dr[1] for dr in date_ranges if dr[1] is not None]
        print(f"  Total rows:        {total_rows:,}")
        if all_mins and all_maxs:
            print(f"  Overall date range: {min(all_mins).date()} to {max(all_maxs).date()}")
        print(f"  Output directory:  {output_dir}")
        print(f"  Combined CSV:      {all_csv}")
    
    if failed_tickers:
        print(f"\n  Failed tickers ({len(failed_tickers)}):")
        for ft in failed_tickers[:20]:
            print(f"    - {ft}")
        if len(failed_tickers) > 20:
            print(f"    ... and {len(failed_tickers) - 20} more")
    
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
