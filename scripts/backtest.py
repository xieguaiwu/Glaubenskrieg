"""Simple backtesting script for CTM + GBDT ensemble signals.

Simulates a long-short portfolio strategy from prediction signals.
Generates performance metrics and an HTML report.

Usage:
    python scripts/backtest.py --predictions results/predictions.csv
    python scripts/backtest.py --infer  # run inference then backtest
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Metric helpers ───────────────────────────────────────────────

def _compute_metrics(signal: pd.Series, top_k: int, bottom_k: int) -> Dict[str, float]:
    """Compute portfolio metrics for a given signal series using rank-based long-short."""
    n = len(signal)
    tk = min(top_k, n // 2)
    bk = min(bottom_k, n // 2)
    rank = signal.rank(ascending=False)
    pos = pd.Series(0, index=signal.index)
    pos[rank <= tk] = 1
    pos[rank > (n - bk)] = -1
    daily_ret = pos * signal
    cum = (1 + daily_ret).cumprod()
    total_ret = float(cum.iloc[-1] - 1)
    sr = float(np.sqrt(252) * daily_ret.mean() / (daily_ret.std(ddof=1) + 1e-10))
    peak = cum.expanding().max()
    mdd = float(((cum - peak) / peak).min())
    wr = float((daily_ret > 0).mean())
    return {"total_return": total_ret, "sharpe": sr, "max_drawdown": mdd, "win_rate": wr}


def _build_comparison_table(metrics: Dict[str, Dict[str, float]]) -> str:
    """Build an HTML table comparing metrics across models."""
    cols = sorted(metrics.keys())
    header = "<tr><th>Metric</th>" + "".join(f"<th>{c.upper()}</th>" for c in cols) + "</tr>"
    rows = ""
    for metric_name in ["total_return", "sharpe", "max_drawdown", "win_rate"]:
        rows += f"<tr><td>{metric_name}</td>"
        for c in cols:
            val = metrics[c].get(metric_name, 0)
            if metric_name == "total_return":
                rows += f"<td class=\"{'positive' if val > 0 else 'negative'}\">{val:.2%}</td>"
            elif metric_name == "sharpe":
                rows += f"<td class=\"{'positive' if val > 0 else 'negative'}\">{val:.2f}</td>"
            elif metric_name == "max_drawdown":
                rows += f"<td class=\"{'positive' if val > -0.2 else 'negative'}\">{val:.2%}</td>"
            else:
                rows += f"<td>{val:.1%}</td>"
        rows += "</tr>"
    return f"<table>{header}{rows}</table>"


def _compute_ic_series(df: pd.DataFrame) -> pd.Series | None:
    """Compute Information Coefficient (Spearman ρ) per time step.

    Returns a Series of per-timestamp IC values, or None if insufficient data.
    """
    if "fused_signal" not in df.columns:
        return None
    pred = df["fused_signal"]
    # Use gbdt_prediction as independent benchmark if available, else actual_return
    if "gbdt_prediction" in df.columns:
        bench = df["gbdt_prediction"]
    elif "actual_return" in df.columns:
        bench = df["actual_return"]
    else:
        return None
    # Compute IC per timestamp (cross-sectional if multi-asset)
    if isinstance(df.index, pd.DatetimeIndex):
        ic_vals = {}
        for ts, grp in df.groupby(df.index):
            if len(grp) < 3:
                continue
            corr, _ = spearmanr(grp["fused_signal"], bench.loc[grp.index])
            ic_vals[ts] = corr
        return pd.Series(ic_vals) if ic_vals else None
    else:
        # Single-asset: rolling IC
        window = min(252, len(df))
        return pred.rolling(window).apply(
            lambda x: spearmanr(x, bench.loc[x.index])[0] if len(x) > 2 else np.nan
        )


def _compute_hit_rate_by_decile(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute hit rate (fraction) for each decile of signal strength.

    Hit = direction of fused_signal matches direction of gbdt_prediction (or
    actual_return if available).  Returns a DataFrame with decile stats, or
    None if no reference column exists.
    """
    if "fused_signal" not in df.columns:
        return None
    signal = df["fused_signal"]
    if "gbdt_prediction" in df.columns:
        ref = df["gbdt_prediction"]
    elif "actual_return" in df.columns:
        ref = df["actual_return"]
    else:
        return None
    decile_labels, bins = pd.qcut(signal, 10, labels=range(1, 11), retbins=True)
    result = df.copy()
    result["decile"] = decile_labels
    result["hit"] = (np.sign(result["fused_signal"]) == np.sign(ref)).astype(int)
    decile_stats = result.groupby("decile", observed=False).agg(
        hit_rate=("hit", "mean"),
        count=("hit", "count"),
        mean_signal=("fused_signal", "mean"),
    ).reset_index()
    decile_stats["hit_rate_pct"] = decile_stats["hit_rate"] * 100
    return decile_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest CTM + GBDT ensemble signals")
    parser.add_argument("--predictions", default=None, help="CSV from infer.py")
    parser.add_argument("--infer", action="store_true", help="Run infer.py first")
    parser.add_argument("--infer-args", default="", help="Args to pass to infer.py")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K assets to long")
    parser.add_argument("--bottom-k", type=int, default=10, help="Bottom-K assets to short")
    parser.add_argument("--output", default="results/backtest_report.html", help="Output HTML report path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # ── 1. Load predictions ──
    if args.infer:
        import subprocess
        infer_cmd = f"python scripts/infer.py {args.infer_args}"
        logging.info("Running inference: %s", infer_cmd)
        subprocess.run(infer_cmd, shell=True, check=True)
        pred_path = None
        # Find the output from infer_args
        for part in args.infer_args.split():
            if part.startswith("--output"):
                try:
                    idx = args.infer_args.split().index(part)
                    pred_path = args.infer_args.split()[idx + 1]
                except (ValueError, IndexError):
                    pass
        if pred_path is None:
            pred_path = "results/predictions.csv"
        predictions_path = pred_path
    else:
        predictions_path = args.predictions

    if predictions_path is None or not os.path.exists(predictions_path):
        logging.error("No predictions file found at %s", predictions_path)
        sys.exit(1)

    df = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
    logging.info("Loaded %d prediction timesteps", len(df))

    if "fused_signal" not in df.columns:
        logging.error("Predictions file must contain 'fused_signal' column")
        sys.exit(1)

    # ── 2. Simulate long-short strategy ──
    # Rank assets by signal each day
    df["rank"] = df["fused_signal"].rank(ascending=False)

    # Long top-K, short bottom-K (if enough assets)
    n_assets = len(df)
    top_k = min(args.top_k, n_assets // 2)
    bottom_k = min(args.bottom_k, n_assets // 2)

    df["position"] = 0
    df.loc[df["rank"] <= top_k, "position"] = 1      # long
    df.loc[df["rank"] > (n_assets - bottom_k), "position"] = -1  # short

    # Daily returns (long-short equally weighted)
    df["daily_return"] = df["position"] * df["fused_signal"]

    # ── 3. Performance metrics ──
    cumulative = (1 + df["daily_return"]).cumprod()
    total_return = cumulative.iloc[-1] - 1
    sharpe = np.sqrt(252) * df["daily_return"].mean() / (df["daily_return"].std(ddof=1) + 1e-10)

    # Max drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (df["daily_return"] > 0).mean()

    # ── 3b. GBDT vs CTM comparison ──
    comparison_html = ""
    if "gbdt_prediction" in df.columns and "ctm_prediction" in df.columns:
        signals = {
            "ctm": df["ctm_prediction"],
            "gbdt": df["gbdt_prediction"],
            "ensemble": df["fused_signal"],
        }
        comparison_metrics = {k: _compute_metrics(v, top_k, bottom_k) for k, v in signals.items()}
        comparison_html = f"""<h2>Model Comparison</h2>
{_build_comparison_table(comparison_metrics)}"""
        logging.info(
            "CTM-only sharpe=%.3f, GBDT-only sharpe=%.3f, Ensemble sharpe=%.3f",
            comparison_metrics.get("ctm", {}).get("sharpe", 0),
            comparison_metrics.get("gbdt", {}).get("sharpe", 0),
            comparison_metrics.get("ensemble", {}).get("sharpe", 0),
        )

    # ── 3c. IC time series ──
    ic_html = ""
    ic_data = _compute_ic_series(df)
    if ic_data is not None and len(ic_data) > 0:
        mean_ic = float(ic_data.mean())
        std_ic = float(ic_data.std(ddof=1))
        ic_html = f"""<h2>Information Coefficient (IC) Time Series</h2>
<div class="metric"><div class="value {('positive' if mean_ic > 0 else 'negative')}">{mean_ic:.4f}</div><div class="label">Mean IC</div></div>
<div class="metric"><div class="value">{std_ic:.4f}</div><div class="label">IC Std</div></div>
<div class="metric"><div class="value">{float(ic_data.iloc[-1]):.4f}</div><div class="label">Last IC</div></div>
<div class="metric"><div class="value">{float((ic_data > 0).mean()):.1%}</div><div class="label">IC > 0 Rate</div></div>"""
        logging.info("IC: mean=%.4f, std=%.4f", mean_ic, std_ic)

    # ── 3d. Hit rate by decile ──
    decile_html = ""
    decile_table = _compute_hit_rate_by_decile(df)
    if decile_table is not None:
        decile_rows = ""
        for _, row in decile_table.iterrows():
            decile_rows += (
                f"<tr><td>{int(row['decile'])}</td>"
                f"<td>{row['hit_rate_pct']:.1f}%</td>"
                f"<td>{int(row['count'])}</td>"
                f"<td>{row['mean_signal']:.4f}</td></tr>"
            )
        decile_html = f"""<h2>Hit Rate by Signal Decile</h2>
<table>
<tr><th>Decile</th><th>Hit Rate</th><th>Samples</th><th>Mean Signal</th></tr>
{decile_rows}
</table>"""

    # ── 4. Generate HTML report ──
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # Simple HTML report (inline styles for portability)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Backtest Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #1a1a2e; }}
  .metric {{ display: inline-block; margin: 10px; padding: 15px 25px; background: #f0f2f5; border-radius: 8px; text-align: center; }}
  .metric .value {{ font-size: 24px; font-weight: bold; color: #16213e; }}
  .metric .label {{ font-size: 12px; color: #666; }}
  .positive {{ color: #00a86b; }}
  .negative {{ color: #d32f2f; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #16213e; color: white; }}
  tr:hover {{ background: #f5f5f5; }}
</style></head><body>
<h1>Backtest Report</h1>
<p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>Performance Summary</h2>
<div class="metric"><div class="value {'positive' if total_return > 0 else 'negative'}">{total_return:.2%}</div><div class="label">Total Return</div></div>
<div class="metric"><div class="value {'positive' if sharpe > 0 else 'negative'}">{sharpe:.2f}</div><div class="label">Sharpe Ratio</div></div>
<div class="metric"><div class="value {'positive' if max_drawdown > -0.2 else 'negative'}">{max_drawdown:.2%}</div><div class="label">Max Drawdown</div></div>
<div class="metric"><div class="value">{win_rate:.1%}</div><div class="label">Win Rate</div></div>

<h2>Strategy Parameters</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Top-K (long)</td><td>{top_k}</td></tr>
<tr><td>Bottom-K (short)</td><td>{bottom_k}</td></tr>
<tr><td>Signals</td><td>{len(df)}</td></tr>
</table>

{comparison_html}
{ic_html}
{decile_html}

<h2>Top 10 Predictions (last day)</h2>
<table>
<tr><th>Rank</th><th>Signal</th><th>Position</th></tr>
"""
    # Last day's top predictions
    last_day = df.sort_values("fused_signal", ascending=False).head(10)
    for i, (_, row) in enumerate(last_day.iterrows(), 1):
        pos = "Long" if row["position"] > 0 else ("Short" if row["position"] < 0 else "None")
        html += f"<tr><td>{i}</td><td>{row['fused_signal']:.4f}</td><td>{pos}</td></tr>\n"

    html += """</table>
</body></html>"""

    with open(args.output, "w") as f:
        f.write(html)
    logging.info("Report saved to %s", args.output)

    # Print summary
    print("\n=== BACKTEST RESULTS ===")
    print(f"Total Return:    {total_return:>8.2%}")
    print(f"Sharpe Ratio:    {sharpe:>8.2f}")
    print(f"Max Drawdown:    {max_drawdown:>8.2%}")
    print(f"Win Rate:        {win_rate:>8.1%}")
    print(f"Report:          {args.output}")


if __name__ == "__main__":
    main()
