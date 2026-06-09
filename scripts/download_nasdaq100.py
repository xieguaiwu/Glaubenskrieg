#!/usr/bin/env python3
"""
Download NASDAQ-100 historical daily OHLCV data from yfinance.

Sources:
  - NASDAQ-100 constituent list from Wikipedia: https://en.wikipedia.org/wiki/Nasdaq-100
  - OHLCV data via yfinance (Yahoo Finance)

Output:
  - Individual CSVs: ../new_data/data/nasdaq100/{TICKER}.csv
  - Combined CSV:    ../new_data/data/nasdaq100_all.csv
"""

import argparse
import logging
import os
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
PROXY_URL = "http://127.0.0.1:7897"
PROXIES = {"http": PROXY_URL, "https": PROXY_URL}

# Ensure proxy env vars are set for yfinance (v1.3.0 uses HTTPS_PROXY/HTTP_PROXY)
os.environ.setdefault("HTTP_PROXY", PROXY_URL)
os.environ.setdefault("HTTPS_PROXY", PROXY_URL)

WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
WIKI_TABLE_INDEX = 5
START_DATE = "2015-01-01"
END_DATE = "2026-06-05"
BATCH_SIZE = 50
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Fetch NASDAQ-100 constituents from Wikipedia
# ──────────────────────────────────────────────
def fetch_constituents() -> list[str]:
    """
    Scrape NASDAQ-100 ticker list from Wikipedia.
    Returns a list of uppercased, cleaned ticker symbols.
    """
    logger.info("Fetching NASDAQ-100 constituents from Wikipedia...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(
            WIKI_URL, headers=headers, proxies=PROXIES, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to fetch Wikipedia page: %s", e)
        sys.exit(1)

    html_io = StringIO(resp.text)
    tables = pd.read_html(html_io)

    if WIKI_TABLE_INDEX >= len(tables):
        logger.error(
            "Table index %d out of range (found %d tables)", WIKI_TABLE_INDEX, len(tables)
        )
        sys.exit(1)

    df = tables[WIKI_TABLE_INDEX]
    logger.info("Wikipedia table columns: %s", list(df.columns))

    ticker_col = None
    for col in df.columns:
        if col.lower() in ("ticker", "symbol"):
            ticker_col = col
            break
    if ticker_col is None:
        # Try multi-level columns
        for col in df.columns:
            if isinstance(col, tuple) and col[0].lower() in ("ticker", "symbol"):
                ticker_col = col
                break
    if ticker_col is None:
        logger.error("Could not find ticker/symbol column in table. Columns: %s", list(df.columns))
        sys.exit(1)

    tickers_raw = df[ticker_col].dropna().astype(str).str.strip().tolist()
    # Clean: remove footnotes, keep only valid ticker characters
    import re

    tickers: list[str] = []
    seen: set[str] = set()
    for t in tickers_raw:
        cleaned = re.sub(r"\[.*?\]", "", t).strip().upper()
        # Must be 1-8 uppercase letters (optionally with . for BRK.B etc.)
        match = re.match(r"^[A-Z]{1,8}$", cleaned)
        if not match:
            # Allow BRK.B type (with dot)
            match = re.match(r"^[A-Z]{1,8}\.[A-Z]{1,5}$", cleaned)
        if match and cleaned not in seen:
            tickers.append(cleaned)
            seen.add(cleaned)

    logger.info("Extracted %d tickers: %s...", len(tickers), tickers[:5])
    return tickers


# ──────────────────────────────────────────────
# Download via yfinance
# ──────────────────────────────────────────────
def download_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download a batch of tickers using yfinance (proxy via env vars)."""
    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=True,
                timeout=REQUEST_TIMEOUT,
            )
            return data
        except Exception as e:
            logger.warning(
                "Batch attempt %d/%d failed for %d tickers: %s",
                attempt + 1,
                MAX_RETRIES,
                len(tickers),
                e,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


def download_single(ticker: str, start: str, end: str) -> "pd.DataFrame | None":
    """Download a single ticker individually (for retry)."""
    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                timeout=REQUEST_TIMEOUT,
            )
            if data is not None and not data.empty:
                return data
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def download_all(
    tickers: list[str],
    start: str,
    end: str,
    batch_size: int = 50,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for all tickers in batches.
    Returns dict: {ticker: DataFrame}.
    """
    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logger.info(
            "Downloading batch %d/%d (%d tickers: %s...)",
            i // batch_size + 1,
            (len(tickers) + batch_size - 1) // batch_size,
            len(batch),
            batch[:3],
        )
        try:
            df = download_batch(batch, start, end)
        except Exception as e:
            logger.error("Batch failed after %d retries: %s", MAX_RETRIES, e)
            failed.extend(batch)
            continue

        # yfinance 1.3.0 always returns MultiIndex columns: (Price, Ticker)
        if df is None or df.empty:
            failed.extend(batch)
            continue

        cols = df.columns
        if not isinstance(cols, pd.MultiIndex):
            # Unexpected format — try treating as single ticker
            logger.warning("Unexpected column format (not MultiIndex), trying plain columns")
            tdf = df.copy()
            tdf.columns = [str(c).lower() for c in tdf.columns]
            tdf.index.name = "date"
            tdf = tdf.dropna(how="all")
            if not tdf.empty and len(batch) == 1:
                results[batch[0]] = tdf
            else:
                failed.extend(batch)
            continue

        # Extract ticker-level data from MultiIndex
        available_tickers = set(cols.get_level_values(1))

        for ticker in batch:
            if ticker not in available_tickers:
                failed.append(ticker)
                continue
            try:
                ticker_df = df.xs(ticker, level=1, axis=1).copy()
            except KeyError:
                failed.append(ticker)
                continue

            # Flatten column names: "Close" -> "close", etc.
            ticker_df.columns = [str(c).lower() for c in ticker_df.columns]
            ticker_df = ticker_df.dropna(how="all")
            if ticker_df.empty:
                failed.append(ticker)
            else:
                ticker_df.index.name = "date"
                results[ticker] = ticker_df

        # Small pause between batches to avoid rate limiting
        if i + batch_size < len(tickers):
            time.sleep(0.5)

    # Retry failed tickers individually (Yahoo may rate-limit large batches)
    if failed:
        logger.info(
            "Retrying %d failed tickers individually with delay...", len(failed)
        )
        retry_failed: list[str] = []
        for ticker in failed:
            logger.info("  Retrying %s...", ticker)
            try:
                df = download_single(ticker, start, end)
            except Exception:
                df = None
            if df is not None and not df.empty:
                cols = df.columns
                if isinstance(cols, pd.MultiIndex):
                    try:
                        ticker_df = df.xs(ticker, level=1, axis=1).copy()
                    except KeyError:
                        ticker_df = df.copy()
                        ticker_df.columns = [str(c).lower() for c in ticker_df.columns]
                else:
                    ticker_df = df.copy()
                    ticker_df.columns = [str(c).lower() for c in ticker_df.columns]
                ticker_df.index.name = "date"
                ticker_df = ticker_df.dropna(how="all")
                if not ticker_df.empty:
                    results[ticker] = ticker_df
                    logger.info("    -> %d rows downloaded", len(ticker_df))
                else:
                    retry_failed.append(ticker)
            else:
                retry_failed.append(ticker)
            # Delay between individual requests
            time.sleep(0.3)
        failed = retry_failed

    if failed:
        logger.warning("Failed to download data for %d tickers: %s", len(failed), failed)

    return results


# ──────────────────────────────────────────────
# Save outputs
# ──────────────────────────────────────────────
def save_results(
    results: dict[str, pd.DataFrame],
    output_dir: str,
    combined_path: str,
) -> tuple[int, int, str, str]:
    """Save individual CSVs and combined CSV. Returns stats."""
    os.makedirs(output_dir, exist_ok=True)

    total_rows = 0
    stocks_saved = 0
    all_frames: list[pd.DataFrame] = []

    for ticker, df in sorted(results.items()):
        if df is None or df.empty:
            continue
        # Save individual
        out_path = os.path.join(output_dir, f"{ticker}.csv")
        df.to_csv(out_path, float_format="%.4f")
        stocks_saved += 1
        total_rows += len(df)

        # Prepare for combined
        df_copy = df.copy()
        df_copy["symbol"] = ticker
        all_frames.append(df_copy)

    # Save combined
    if all_frames:
        combined = pd.concat(all_frames, axis=0)
        combined.index.name = "date"
        combined.to_csv(combined_path, float_format="%.4f")
        # Compute date range
        date_min = str(combined.index.min().date())
        date_max = str(combined.index.max().date())
    else:
        date_min = date_max = "N/A"
        combined = pd.DataFrame()

    return stocks_saved, total_rows, date_min, date_max


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download NASDAQ-100 OHLCV data from yfinance"
    )
    parser.add_argument(
        "--start", default=START_DATE, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default=END_DATE, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for individual CSVs (default: ../new_data/data/nasdaq100)",
    )
    parser.add_argument(
        "--combined",
        default=None,
        help="Combined CSV path (default: ../new_data/data/nasdaq100_all.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for yfinance downloads",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download, just show constituents",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    default_output_dir = str(project_root / "new_data" / "data" / "nasdaq100")
    default_combined = str(project_root / "new_data" / "data" / "nasdaq100_all.csv")

    output_dir = args.output_dir or default_output_dir
    combined_path = args.combined or default_combined

    logger.info("=" * 60)
    logger.info("NASDAQ-100 OHLCV Downloader")
    logger.info("=" * 60)
    logger.info("Date range: %s → %s", args.start, args.end)
    logger.info("Output dir: %s", output_dir)
    logger.info("Combined:   %s", combined_path)

    # 1. Fetch constituents
    tickers = fetch_constituents()
    logger.info("Constituents: %d tickers", len(tickers))

    if args.no_download:
        for t in tickers:
            print(t)
        return

    # 2. Download
    start_time = time.monotonic()
    results = download_all(tickers, args.start, args.end, batch_size=args.batch_size)
    elapsed = time.monotonic() - start_time
    logger.info("Download completed in %.1f seconds", elapsed)

    # 3. Save
    stocks_saved, total_rows, date_min, date_max = save_results(
        results, output_dir, combined_path
    )

    # 4. Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Tickers requested:   %d", len(tickers))
    logger.info("Stocks downloaded:   %d", stocks_saved)
    logger.info("Stocks failed:       %d", len(tickers) - stocks_saved)
    logger.info("Total rows:          %d", total_rows)
    logger.info("Date range:          %s → %s", date_min, date_max)
    logger.info("Individual files:    %s/*.csv", output_dir)
    logger.info("Combined file:       %s", combined_path)

    # Print failure list if any
    if len(results) < len(tickers):
        failed = set(tickers) - set(results.keys())
        logger.warning("Failed tickers: %s", sorted(failed))

    return 0


if __name__ == "__main__":
    sys.exit(main())
