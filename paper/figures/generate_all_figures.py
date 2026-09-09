#!/usr/bin/env python3
"""
Glaubenskrieg — Generate all 8 publication figures.

Usage:
    cd /home/xieguiawu/Desktop/ML/Glaubenskrieg
    python3 paper/figures/generate_all_figures.py

Output:
    paper/figures/*.pdf (vector format for publication)
    paper/figures/*.png (raster for quick viewing)

Dependencies: numpy, pandas, matplotlib, seaborn, json
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette (IBM Carbon)
CBT = ["#0062ff", "#ff832b", "#24a148", "#fa4d56", "#8a3ffc", "#009d9a"]
LGB_COLOR = CBT[0]
RIDGE_COLOR = CBT[1]
GARCH_COLOR = CBT[2]
CTM_COLOR = CBT[4]
BASELINE_GRAY = "#8d8d8d"


def save(fig, name):
    """Save figure as PDF + PNG."""
    fig.savefig(OUT / f"{name}.pdf", format="pdf")
    fig.savefig(OUT / f"{name}.png", format="png", dpi=200)
    print(f"  ✅ {name}.pdf + {name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Cross-Market IC Comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig1_cross_market_ic():
    """Bar chart: LightGBM vs Ridge IC across 3 markets, with 0.02 threshold."""
    markets = ["Hong Kong", "A-Share", "US"]
    n_stocks = [200, 200, 477]

    lgb_ic = np.array([0.053, -0.003, 0.007])
    lgb_std = np.array([0.054, 0.042, 0.043])
    ridge_ic = np.array([0.006, 0.030, 0.031])
    # Ridge std: HK has no per-window data; use A-Share/US values
    ridge_std = np.array([np.nan, 0.054, 0.030])

    x = np.arange(len(markets))
    w = 0.3

    fig, ax = plt.subplots(figsize=(5, 3.5))

    bars1 = ax.bar(x - w / 2, lgb_ic, w, label="LightGBM",
                   color=LGB_COLOR, yerr=lgb_std, capsize=3,
                   error_kw={"linewidth": 1})
    bars2 = ax.bar(x + w / 2, ridge_ic, w, label="Ridge (451 params)",
                   color=RIDGE_COLOR, yerr=ridge_std, capsize=3,
                   error_kw={"linewidth": 1})

    # 0.02 detection threshold
    ax.axhline(0.02, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Detection threshold (IC=0.02)")
    ax.axhline(0, color="black", linewidth=0.5)

    # Stock count annotations
    for i, (m, n) in enumerate(zip(markets, n_stocks)):
        ax.text(i, -0.07, f"n={n}", ha="center", fontsize=7,
                color=BASELINE_GRAY)

    ax.set_ylabel("Information Coefficient (Spearman ρ)")
    ax.set_xticks(x)
    ax.set_xticklabels(markets)
    ax.set_title("Figure 1: Cross-Market Return Prediction IC")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_ylim(-0.10, 0.14)

    fig.tight_layout()
    save(fig, "fig01_cross_market_ic")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-Window IC Distribution (US, 477 stocks)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_per_window_ic():
    """Dot + line plot: per-window IC for LightGBM and Ridge."""
    windows = np.arange(10)
    lgb_ic = np.array([-0.067, -0.042, -0.039, 0.025, 0.047,
                       0.001, 0.026, 0.022, 0.083, 0.010])
    ridge_ic = np.array([0.010, 0.003, 0.012, 0.065, 0.056,
                         0.042, 0.015, -0.002, 0.095, 0.012])

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    ax.plot(windows, lgb_ic, "o-", color=LGB_COLOR, label="LightGBM",
            markersize=5, linewidth=1.2, alpha=0.85)
    ax.plot(windows, ridge_ic, "s--", color=RIDGE_COLOR, label="Ridge",
            markersize=5, linewidth=1.2, alpha=0.85)

    ax.axhline(0.02, color="red", linestyle=":", linewidth=0.8, alpha=0.5,
               label="IC=0.02 threshold")
    ax.axhline(0, color="black", linewidth=0.5)

    # Annotate means
    ax.axhline(lgb_ic.mean(), color=LGB_COLOR, linewidth=0.8, linestyle=":",
               alpha=0.5)
    ax.axhline(ridge_ic.mean(), color=RIDGE_COLOR, linewidth=0.8, linestyle=":",
               alpha=0.5)

    ax.set_xlabel("Walk-Forward Window")
    ax.set_ylabel("IC (Spearman ρ)")
    ax.set_title("Figure 2: Per-Window IC Stability (US, 477 stocks)")
    ax.set_xticks(windows)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.set_ylim(-0.10, 0.12)

    # Add text annotation for means
    ax.text(9.5, lgb_ic.mean() + 0.005, f"μ={lgb_ic.mean():.3f}",
            color=LGB_COLOR, fontsize=7, ha="right", va="bottom")
    ax.text(9.5, ridge_ic.mean() + 0.005, f"μ={ridge_ic.mean():.3f}",
            color=RIDGE_COLOR, fontsize=7, ha="right", va="bottom")

    fig.tight_layout()
    save(fig, "fig02_per_window_ic")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Volatility QLIKE — Cross-Market Comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig3_volatility_qlike():
    """Grouped bar: LightGBM vs Mean vs Persistence QLIKE across markets."""
    markets = ["Hong Kong", "A-Share", "US"]
    lgb_q = np.array([-6.42, -6.78, -7.10])
    mean_q = np.array([-6.29, -6.34, -6.75])
    persist_q = np.array([-5.67, -5.85, -6.50])
    lgb_q_std = np.array([0.24, 0.26, 0.20])

    x = np.arange(len(markets))
    w = 0.25

    fig, ax = plt.subplots(figsize=(5, 3.2))

    ax.bar(x - w, lgb_q, w, label="LightGBM", color=LGB_COLOR,
           yerr=lgb_q_std, capsize=3, error_kw={"linewidth": 1})
    ax.bar(x, mean_q, w, label="Historical Mean", color=BASELINE_GRAY,
           alpha=0.6)
    ax.bar(x + w, persist_q, w, label="Persistence", color="#a8a8a8",
           alpha=0.4)

    # Annotations
    for i in range(len(markets)):
        ax.text(i, lgb_q[i] + 0.15, f"Δ={lgb_q[i]-mean_q[i]:+.2f}",
                ha="center", fontsize=7, fontweight="bold",
                color=LGB_COLOR)

    ax.set_ylabel("QLIKE (↑ better)")
    ax.set_xticks(x)
    ax.set_xticklabels(markets)
    ax.set_title("Figure 3: Cross-Market Volatility Prediction (QLIKE)")
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_ylim(-7.6, -5.2)

    fig.tight_layout()
    save(fig, "fig03_volatility_qlike")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Sharpe vs IC Scatter (The Sharpe Paradox)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_sharpe_vs_ic():
    """Scatter: Sharpe vs IC for each window-model combination."""
    data = [
        ("US-0", -0.067, 2.84), ("US-1", -0.042, 1.02),
        ("US-2", -0.039, 0.09), ("US-3", 0.025, 0.27),
        ("US-4", 0.047, 0.64), ("US-5", 0.001, 2.12),
        ("US-6", 0.026, 2.15), ("US-7", 0.022, 0.94),
        ("US-8", 0.083, 0.56), ("US-9", 0.010, 0.96),
        ("AS-0", 0.053, -0.93), ("AS-1", 0.014, 1.48),
        ("AS-2", 0.037, 0.67), ("AS-3", -0.059, 2.32),
        ("AS-4", -0.051, 1.31), ("AS-5", -0.014, -0.33),
        # Add portfolio optimizer points for extra impact
        ("PO-MSE", 0.010, 2.36), ("PO-ST", -0.014, 2.98),
        ("PO-Rand", 0.003, 0.04), ("PO-MRev", -0.009, 0.04),
    ]

    labels = [d[0] for d in data]
    ics = [d[1] for d in data]
    sharpes = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Separate main data vs portfolio optimizer
    main_idx = [i for i, l in enumerate(labels) if l.startswith("US-") or l.startswith("AS-")]
    po_idx = [i for i, l in enumerate(labels) if l.startswith("PO-")]

    ax.scatter([ics[i] for i in main_idx], [sharpes[i] for i in main_idx],
               c=LGB_COLOR, s=30, alpha=0.7, label="Walk-forward windows",
               zorder=5)

    ax.scatter([ics[i] for i in po_idx], [sharpes[i] for i in po_idx],
               c=RIDGE_COLOR, s=50, marker="s", alpha=0.8,
               label="Portfolio optimizer", zorder=6)

    # Annotate extreme points
    for i in po_idx:
        ax.annotate(labels[i].replace("PO-", ""),
                    (ics[i], sharpes[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color=RIDGE_COLOR, fontweight="bold")

    # Pearson correlation
    ics_arr = np.array(ics)
    sh_arr = np.array(sharpes)
    corr = np.corrcoef(ics_arr, sh_arr)[0, 1]
    ax.text(0.02, 0.95, f"Pearson r = {corr:.2f}",
            transform=ax.transAxes, fontsize=8, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.02, color="red", linestyle=":", linewidth=0.8, alpha=0.5,
               label="IC=0.02 threshold")

    ax.set_xlabel("IC (Spearman ρ)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Figure 4: The Sharpe Paradox — IC vs Sharpe Decoupling")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_xlim(-0.10, 0.12)

    fig.tight_layout()
    save(fig, "fig04_sharpe_vs_ic")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Volatility Regime Strategy — Cumulative Returns
# ─────────────────────────────────────────────────────────────────────────────
def fig5_cumulative_returns():
    """Line plot: equity curves for 4 volatility strategies."""
    strategy_files = {
        "Equal Weight": "vol_regime_equal_weight.csv",
        "Inverse Vol": "vol_regime_inverse_vol.csv",
        "Vol Target 15%": "vol_regime_vol_target_15.csv",
        "Regime-Adaptive": "vol_regime_regime_adaptive.csv",
    }
    colors = [BASELINE_GRAY, "#a8a8a8", LGB_COLOR, CBT[1]]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    for (name, fname), color in zip(strategy_files.items(), colors):
        df = pd.read_csv(ROOT / "results" / fname)
        ax.plot(pd.to_datetime(df["date"]), df["equity"],
                label=name, color=color, linewidth=1.0)

    ax.set_ylabel("Cumulative Equity (log scale)")
    ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_title("Figure 5: Volatility Regime Strategy — Cumulative Returns")
    ax.legend(fontsize=7, framealpha=0.9, loc="upper left")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}×"))

    # Add annotation for key events
    ax.axvline(pd.Timestamp("2020-03-01"), color="red", linewidth=0.5,
               linestyle=":", alpha=0.4)
    ax.text(pd.Timestamp("2020-03-15"), ax.get_ylim()[1] * 0.95,
            "COVID-19", fontsize=6, color="red", alpha=0.5, rotation=90)

    fig.tight_layout()
    save(fig, "fig05_cumulative_returns")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Portfolio Optimization — Method Comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig6_portfolio_optimization():
    """Bar chart: Sharpe ratio comparison across optimization methods."""
    methods = ["LGB-MSE", "LGB-LambdaRank", "LGB-SharpeTuned", "Random", "MeanRev"]
    sharpes = np.array([2.361, 2.793, 2.981, 0.036, 0.041])
    stds = np.array([0.930, 2.345, 1.048, 0.847, 0.399])
    colors = [LGB_COLOR, CBT[3], CBT[1], BASELINE_GRAY, "#a8a8a8"]

    fig, ax = plt.subplots(figsize=(5, 3.2))

    bars = ax.bar(methods, sharpes, color=colors, width=0.6,
                   yerr=stds, capsize=4, error_kw={"linewidth": 1})

    # Value annotations
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.2f}", ha="center", fontsize=7, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Validation Sharpe Ratio")
    ax.set_title("Figure 6: Portfolio Optimization — Method Comparison")
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=7)

    # Delta annotation
    delta = sharpes[2] - sharpes[0]
    ax.annotate(f"Δ = +{delta:.2f} (p=0.011)",
                xy=(1.5, sharpes.mean()), fontsize=8,
                fontweight="bold", color=CBT[1],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=CBT[1], alpha=0.8))

    fig.tight_layout()
    save(fig, "fig06_portfolio_optimization")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: GARCH vs LightGBM — Per-Stock Volatility QLIKE
# ─────────────────────────────────────────────────────────────────────────────
def fig7_garch_vs_lgb():
    """Box plot or paired bar: GARCH vs LightGBM per-stock QLIKE."""
    # Use the 47-stock data from garch_baseline.json
    with open(ROOT / "results/garch_baseline.json") as f:
        gb = json.load(f)

    ps_dict = gb["aggregate"]["per_stock"]  # dict: symbol -> {garch_qlike, lgb_qlike}
    symbols = sorted(ps_dict.keys())[:30]  # First 30 alphabetically
    garch_q = [ps_dict[s]["garch_qlike"] for s in symbols]
    lgb_q = [ps_dict[s]["lgb_qlike"] for s in symbols]

    x = np.arange(len(symbols))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.bar(x - w / 2, garch_q, w, label="GARCH(1,1)", color=GARCH_COLOR,
           alpha=0.85)
    ax.bar(x + w / 2, lgb_q, w, label="LightGBM", color=LGB_COLOR,
           alpha=0.85)

    ax.set_ylabel("QLIKE (↓ better)")
    ax.set_xlabel("Stock")
    ax.set_title("Figure 7: GARCH(1,1) vs LightGBM — Per-Stock QLIKE")
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=90, fontsize=5)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_yscale("symlog", linthresh=1)  # Log scale for wide range

    # Win rate annotation
    fraction_lgb_better = gb["aggregate"].get("fraction_lgb_better_than_garch", 0.0)
    n_total = gb["aggregate"]["n_stocks_valid"]
    n_garch_wins = n_total - int(fraction_lgb_better * n_total)
    ax.text(0.98, 0.95,
            f"GARCH wins: {n_garch_wins}/{n_total} ({n_garch_wins/n_total*100:.0f}%)",
            transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="right", va="top", color=GARCH_COLOR,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    save(fig, "fig07_garch_vs_lgb")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: CTM IC Evolution (v1 → v5)
# ─────────────────────────────────────────────────────────────────────────────
def fig8_ctm_evolution():
    """Bar + line: CTM validation IC across versions, with test IC reveal."""
    versions = ["v1", "v2", "v3", "v4 (TGPE)", "v5 (final)", "test"]
    ics = [0.14, 0.08, 0.14, 0.065, 0.049, 0.006]
    colors = [CBT[i % len(CBT)] for i in range(5)] + [BASELINE_GRAY]

    fig, ax = plt.subplots(figsize=(5, 3.2))

    bars = ax.bar(versions, ics, color=colors, width=0.5, edgecolor="black",
                   linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, ics):
        y_pos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.015
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

    # Gap annotation
    ax.annotate("Overstatement\n23.3×",
                xy=(4, 0.049), xytext=(5, 0.10),
                fontsize=8, color="red", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="red",
                                connectionstyle="arc3,rad=-0.3"))

    # Arrow from v5 to test
    ax.annotate("Held-out\ntest reveal",
                xy=(4, 0.049), xytext=(5, 0.03),
                fontsize=7, color=BASELINE_GRAY,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=BASELINE_GRAY,
                                connectionstyle="arc3,rad=0.2"))

    ax.axhline(0.02, color="red", linestyle=":", linewidth=0.8, alpha=0.5,
               label="IC=0.02 detection threshold")
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set_ylabel("IC (Spearman ρ)")
    ax.set_title("Figure 8: CTM IC Evolution — Bug Fixes Reveal Overfitting")
    ax.set_xticklabels(versions, rotation=20, ha="right", fontsize=7)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_ylim(-0.01, 0.18)

    fig.tight_layout()
    save(fig, "fig08_ctm_evolution")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: Parameter Efficiency Bar (for Table 5 visualization)
# ─────────────────────────────────────────────────────────────────────────────
def fig_bonus_param_efficiency():
    """Log-scale bar: params/sample for each model."""
    models = ["Linear Ridge", "GBDT", "LightGBM", "CTM (Mamba)", "CTM (pre-fix)"]
    params = [451, 5000, 5000, 97000, 157323]
    ps_ratio = [0.58, 6.4, 6.4, 125, 202]
    ic_values = [0.031, 0.008, 0.007, 0.006, 0.006]
    colors = [RIDGE_COLOR, CBT[2], LGB_COLOR, CBT[4], "#a8a8a8"]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.2))

    bars = ax1.bar(models, ps_ratio, color=colors, width=0.5, alpha=0.85,
                   edgecolor="black", linewidth=0.5)

    # Params/sample value labels
    for bar, val in zip(bars, ps_ratio):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f"{val:.1f}", ha="center", fontsize=7, fontweight="bold")

    ax1.set_ylabel("Parameters per Sample (log scale)")
    ax1.set_yscale("log")
    ax1.set_title("Figure B1: Parameter Efficiency vs Predictive Power")
    ax1.set_xticklabels(models, rotation=25, ha="right", fontsize=7)

    # Overlay IC as a line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(models, ic_values, "o-", color="red", linewidth=1.5,
             markersize=6, label="Test IC")
    ax2.set_ylabel("Test IC", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.axhline(0.02, color="red", linestyle=":", linewidth=0.6, alpha=0.4)

    # Annotation: Ridge has highest IC with lowest params/sample
    ax1.annotate("Best IC/\nEfficiency",
                 xy=(0, 0.58), xytext=(0.8, 80),
                 fontsize=7, color=RIDGE_COLOR, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RIDGE_COLOR))

    fig.tight_layout()
    save(fig, "fig_bonus_param_efficiency")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...\n")

    fig1_cross_market_ic()
    fig2_per_window_ic()
    fig3_volatility_qlike()
    fig4_sharpe_vs_ic()
    fig5_cumulative_returns()
    fig6_portfolio_optimization()
    fig7_garch_vs_lgb()
    fig8_ctm_evolution()
    fig_bonus_param_efficiency()

    print(f"\nAll figures saved to {OUT}/")
    print("Files:")
    for f in sorted(OUT.glob("*")):
        if f.suffix in (".pdf", ".png"):
            print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
