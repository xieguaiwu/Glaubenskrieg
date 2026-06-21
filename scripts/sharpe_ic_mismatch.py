#!/usr/bin/env python3
"""
Sharpe-IC Mismatch Analysis: compute the percentage of experimental units
where |IC| < THRESHOLD but Sharpe > 0 (indicating deceptive performance).

Reads all available experimental result JSONs, extracts IC and Sharpe
at the most granular level (per fold/window/seed), and computes
the mismatch fraction.

Usage:
  cd ~/Desktop/ML/Glaubenskrieg
  python scripts/sharpe_ic_mismatch.py
"""
import json, os, glob, sys
from collections import defaultdict
from typing import List, Tuple, Dict

BASE = os.path.expanduser("~/Desktop/ML/Glaubenskrieg/results")

IC_THRESHOLD = 0.02
SHARPE_THRESHOLD = 0.0  # Sharpe > 0

# ──────────────────────────────────────────────
# Data structures for unit collection
# ──────────────────────────────────────────────
Unit = Dict  # {source, model, horizon, seed, window, ic, sharpe, ...}
all_units: List[Unit] = []


def add_unit(source: str, model: str, horizon: str = "",
             seed: str = "", window: int = -1,
             ic: float = None, sharpe: float = None, **extra):
    """Add one experimental unit."""
    if ic is None or sharpe is None:
        return
    if not isinstance(ic, (int, float)) or not isinstance(sharpe, (int, float)):
        return
    if abs(ic) > 1.0 or abs(sharpe) > 100:
        return  # obviously invalid
    all_units.append({
        "source": source, "model": model, "horizon": horizon,
        "seed": str(seed), "window": window,
        "ic": float(ic), "sharpe": float(sharpe),
        **{k: v for k, v in extra.items() if v is not None}
    })


# ──────────────────────────────────────────────
# Parser: pipeline_v2.json
# ──────────────────────────────────────────────
def parse_pipeline_v2():
    path = os.path.join(BASE, "pipeline_v2.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    horizons = d.get("horizons", {})
    for hname, hdata in horizons.items():
        for mname, mdata in hdata.items():
            folds = mdata.get("folds", [])
            for f in folds:
                add_unit("pipeline_v2", mname, hname,
                         window=f.get("window", -1),
                         ic=f.get("ic"), sharpe=f.get("sharpe"))


# ──────────────────────────────────────────────
# Parser: pipeline_v3.json
# ──────────────────────────────────────────────
def parse_pipeline_v3():
    path = os.path.join(BASE, "pipeline_v3.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    horizons = d.get("horizons", {})
    for hname, hdata in horizons.items():
        for seed_key, seed_data in hdata.items():
            if not isinstance(seed_data, dict):
                continue
            for mname, mdata in seed_data.items():
                if not isinstance(mdata, dict):
                    continue
                folds = mdata.get("folds", [])
                if folds:
                    for f in folds:
                        add_unit("pipeline_v3", mname, hname,
                                 seed=seed_key,
                                 window=f.get("window", -1),
                                 ic=f.get("ic"), sharpe=f.get("sharpe"))
                else:
                    # cross-seed aggregates: no fold-level data, skip
                    pass


# ──────────────────────────────────────────────
# Parser: benchmark_ridge_glauben.json
# ──────────────────────────────────────────────
def parse_benchmark():
    path = os.path.join(BASE, "benchmark_ridge_glauben.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    for mname in ["ridge", "glauben"]:
        mdata = d.get(mname, {})
        for f in mdata.get("folds", []):
            add_unit("benchmark", mname, "5d",
                     window=f.get("window", -1),
                     ic=f.get("ic"), sharpe=f.get("sharpe"))


# ──────────────────────────────────────────────
# Parser: glauben_v2_noise_free.json
# ──────────────────────────────────────────────
def parse_noise_free():
    path = os.path.join(BASE, "glauben_v2_noise_free.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    variants = d.get("variants", {})
    for vname, vdata in variants.items():
        for f in vdata:
            add_unit("noise_free", vname, "5d",
                     window=f.get("window", -1),
                     ic=f.get("ic"), sharpe=f.get("sharpe"))


# ──────────────────────────────────────────────
# Parser: us_full_baseline.json
# ──────────────────────────────────────────────
def parse_us_full_baseline():
    path = os.path.join(BASE, "us_full_baseline.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    results = d.get("results", {})
    for market, mdata in results.items():
        if not isinstance(mdata, dict):
            continue
        # Check for folds/windows
        for subkey in ["windows", "folds", "per_window"]:
            folds = mdata.get(subkey, [])
            if isinstance(folds, list) and folds:
                for f in folds:
                    if isinstance(f, dict):
                        add_unit("us_baseline", market, "",
                                 window=f.get("window", -1),
                                 ic=f.get("ic"), sharpe=f.get("sharpe"))
                break
        # Also check per_model
        per_model = mdata.get("per_model", mdata.get("models", {}))
        if isinstance(per_model, dict):
            for mk, mv in per_model.items():
                if isinstance(mv, dict):
                    folds = mv.get("folds", mv.get("windows", []))
                    for f in folds:
                        if isinstance(f, dict):
                            add_unit("us_baseline", mk, "",
                                     window=f.get("window", -1),
                                     ic=f.get("ic"), sharpe=f.get("sharpe"))


# ──────────────────────────────────────────────
# Parser: newdata_baselines.json
# ──────────────────────────────────────────────
def parse_newdata_baselines():
    path = os.path.join(BASE, "newdata_baselines.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    results = d.get("results", {})
    for market, mdata in results.items():
        if not isinstance(mdata, dict):
            continue
        for subkey in ["windows", "folds", "per_window"]:
            folds = mdata.get(subkey, [])
            if isinstance(folds, list) and folds:
                for f in folds:
                    if isinstance(f, dict):
                        add_unit("newdata", f"lgb_{market}", "",
                                 window=f.get("window", -1),
                                 ic=f.get("ic"), sharpe=f.get("sharpe"))
                break


# ──────────────────────────────────────────────
# Parser: remote/v5_final (CTM, Ensemble, GBDT)
# ──────────────────────────────────────────────
def parse_v5_final():
    v5dir = os.path.join(BASE, "remote/v5_final")
    if not os.path.isdir(v5dir):
        return
    for fpath in sorted(glob.glob(os.path.join(v5dir, "*.json"))):
        fname = os.path.basename(fpath).replace(".json", "")
        d = json.load(open(fpath))

        # CTM / Ensemble files: 'results' key
        results = d.get("results", [])
        if results and isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    ic = r.get("ctm_ic", r.get("val_ic", r.get("ic")))
                    sh = r.get("best_sharpe", r.get("sharpe"))
                    add_unit("v5_final", fname, "1d",
                             window=r.get("window", -1),
                             ic=ic, sharpe=sh)
            continue

        # GBDT files: 'windows' key
        windows = d.get("windows", [])
        if windows and isinstance(windows, list):
            for w in windows:
                if isinstance(w, dict):
                    add_unit("v5_final", fname, "1d",
                             window=w.get("window", -1),
                             ic=w.get("ic"), sharpe=w.get("sharpe"))
            continue


# ──────────────────────────────────────────────
# Parser: remote/fe_lgb_final
# ──────────────────────────────────────────────
def parse_fe_lgb():
    fldir = os.path.join(BASE, "remote/fe_lgb_final")
    if not os.path.isdir(fldir):
        return
    for fpath in sorted(glob.glob(os.path.join(fldir, "*.json"))):
        fname = os.path.basename(fpath).replace(".json", "")
        d = json.load(open(fpath))
        # These files have per_window or windows
        for key in ["per_window", "windows", "folds"]:
            folds = d.get(key, [])
            if isinstance(folds, list) and folds:
                for f in folds:
                    if isinstance(f, dict):
                        ic = f.get("ic", f.get("val_ic", f.get("test_ic")))
                        sh = f.get("sharpe", f.get("test_sharpe"))
                        add_unit("fe_lgb", fname, "1d",
                                 window=f.get("window", -1),
                                 ic=ic, sharpe=sh)
                break
        # Also check test_metrics
        tm = d.get("test_metrics", {})
        vm = d.get("val_metrics", {})
        if tm or vm:
            ic = tm.get("test_ic", vm.get("val_ic"))
            sh = tm.get("test_sharpe", vm.get("val_sharpe"))
            if ic is not None and sh is not None:
                add_unit("fe_lgb", fname, "1d", ic=ic, sharpe=sh)


# ──────────────────────────────────────────────
# Parser: archive server1/server2 (old v1/v2/v3)
# ──────────────────────────────────────────────
def parse_archives():
    for arch_dir in ["archive_20260605/server1", "archive_20260605/server2"]:
        ad = os.path.join(BASE, arch_dir)
        if not os.path.isdir(ad):
            continue
        for fpath in sorted(glob.glob(os.path.join(ad, "*.json"))):
            fname = os.path.basename(fpath).replace(".json", "")
            d = json.load(open(fpath))
            # Try results key
            results = d.get("results", [])
            if isinstance(results, list) and results:
                for r in results:
                    if isinstance(r, dict):
                        ic = r.get("ctm_ic", r.get("val_ic", r.get("ic", None)))
                        sh = r.get("best_sharpe", r.get("sharpe", None))
                        add_unit(f"archive/{arch_dir.split('/')[-1]}", fname,
                                 window=r.get("window", -1),
                                 ic=ic, sharpe=sh)
            # Try windows key
            windows = d.get("windows", [])
            if isinstance(windows, list) and windows:
                for w in windows:
                    if isinstance(w, dict):
                        add_unit(f"archive/{arch_dir.split('/')[-1]}", fname,
                                 window=w.get("window", -1),
                                 ic=w.get("ic"), sharpe=w.get("sharpe"))


# ──────────────────────────────────────────────
# Parser: remote/server1_v3, server2_v3 (n20/n50)
# ──────────────────────────────────────────────
def parse_remote_v3():
    for rdir in ["remote/server1_v3", "remote/server2_v3"]:
        ad = os.path.join(BASE, rdir)
        if not os.path.isdir(ad):
            continue
        for fpath in sorted(glob.glob(os.path.join(ad, "*.json"))):
            fname = os.path.basename(fpath).replace(".json", "")
            d = json.load(open(fpath))
            # Look for results with ctm_ic
            results = d.get("results", [])
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict):
                        ic = r.get("ctm_ic", r.get("val_ic", r.get("ic", None)))
                        sh = r.get("best_sharpe", r.get("sharpe", None))
                        add_unit(rdir.split("/")[-1], fname,
                                 window=r.get("window", -1),
                                 ic=ic, sharpe=sh)
            # Also check windows
            windows = d.get("windows", [])
            if isinstance(windows, list):
                for w in windows:
                    if isinstance(w, dict):
                        add_unit(rdir.split("/")[-1], fname,
                                 window=w.get("window", -1),
                                 ic=w.get("ic"), sharpe=w.get("sharpe"))


# ──────────────────────────────────────────────
# Parser: portfolio_optimizer.json
# ──────────────────────────────────────────────
def parse_portfolio():
    path = os.path.join(BASE, "portfolio_optimizer.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    aggregate = d.get("aggregate", {})
    comparisons = d.get("comparisons", {})
    # These contain per-window results too
    window_results = d.get("window_results", [])
    for wr in window_results:
        if isinstance(wr, dict):
            for mk, mv in wr.items():
                if isinstance(mv, dict):
                    ic = mv.get("ic")
                    sh = mv.get("sharpe")
                    if ic is not None and sh is not None:
                        add_unit("portfolio", mk, "",
                                 window=wr.get("window", -1),
                                 ic=ic, sharpe=sh)


# ──────────────────────────────────────────────
# Parser: remote server2_v2, server2_gbdt_sweep, etc.
# ──────────────────────────────────────────────
def parse_misc_remote():
    misc_dirs = [
        "remote/server1_v2", "remote/server2_v2",
        "remote/server2_gbdt_sweep/results/gbdt_sweep_v2",
        "remote/server2_gbdt_sweep/results/gbdt_baseline_v2",
        "remote/tgpe_v4",
        "remote/server2_ensemble_old"
    ]
    for rdir in misc_dirs:
        ad = os.path.join(BASE, rdir)
        if not os.path.isdir(ad):
            continue
        for fpath in sorted(glob.glob(os.path.join(ad, "*.json"))):
            fname = f"{rdir.split('/')[-1]}/{os.path.basename(fpath).replace('.json','')}"
            try:
                d = json.load(open(fpath))
            except:
                continue
            # Generic extraction: search for any fold-level ic/sharpe
            _extract_units_recursive(d, fname, rdir.split("/")[-1])


def _extract_units_recursive(d, source_name: str, group: str, prefix: str = ""):
    """Recursively find dicts with both 'ic' and 'sharpe' keys."""
    if not isinstance(d, dict):
        return

    # Check if this dict itself is a unit
    ic = d.get("ic", d.get("ctm_ic", d.get("val_ic")))
    sh = d.get("sharpe", d.get("best_sharpe"))
    if ic is not None and sh is not None:
        if isinstance(ic, (int, float)) and isinstance(sh, (int, float)):
            if abs(ic) <= 1.0:
                add_unit(group, source_name, prefix.strip("/"),
                         window=d.get("window", -1),
                         ic=ic, sharpe=sh)

    # Recurse into sub-dicts and lists
    for k, v in d.items():
        if k in ("config", "metadata", "data", "features", "backtest",
                 "regime_performance", "regime_counts", "runtime_s",
                 "timestamp", "interpretation", "model_params",
                 "all_columns", "notes", "description"):
            continue
        if isinstance(v, dict):
            _extract_units_recursive(v, source_name, group, f"{prefix}/{k}")
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    _extract_units_recursive(item, source_name, group, f"{prefix}/{k}[{i}]")


# ──────────────────────────────────────────────
# Main: parse all sources, compute statistics
# ──────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Sharpe-IC Mismatch Analysis")
    print("=" * 70)

    # Step 1: Parse structured files (exact fold-level data)
    parsers = [
        ("pipeline_v2", parse_pipeline_v2),
        ("pipeline_v3", parse_pipeline_v3),
        ("benchmark", parse_benchmark),
        ("noise_free", parse_noise_free),
        ("us_baseline", parse_us_full_baseline),
        ("newdata", parse_newdata_baselines),
        ("v5_final", parse_v5_final),
        ("fe_lgb", parse_fe_lgb),
        ("archives", parse_archives),
        ("remote_v3", parse_remote_v3),
        ("portfolio", parse_portfolio),
    ]
    for name, parser in parsers:
        before = len(all_units)
        parser()
        after = len(all_units)
        if after > before:
            print(f"  [{name}]: +{after - before} units (total: {after})")

    # Step 2: Generic recursive extraction for remaining files
    before = len(all_units)
    parse_misc_remote()
    after = len(all_units)
    if after > before:
        print(f"  [misc_remote]: +{after - before} units (total: {after})")

    # Step 3: Deduplicate (same ic+sharpe+source = duplicate)
    seen = set()
    deduped = []
    for u in all_units:
        key = (u["source"], u["model"], u.get("horizon",""),
               u.get("window",-1), round(u["ic"], 6), round(u["sharpe"], 4))
        if key not in seen:
            seen.add(key)
            deduped.append(u)
    dup_count = len(all_units) - len(deduped)
    print(f"\n  Deduplication: removed {dup_count} duplicates, {len(deduped)} unique units")

    all_units.clear()
    all_units.extend(deduped)

    # ─── ANALYSIS ───
    n_total = len(all_units)
    if n_total == 0:
        print("\nERROR: No experimental units found!")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"TOTAL EXPERIMENTAL UNITS: {n_total}")
    print(f"{'='*70}")

    # By source
    source_counts = defaultdict(int)
    for u in all_units:
        source_counts[u["source"]] += 1
    print("\nUnits by source:")
    for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")

    # IC distribution
    ics = [u["ic"] for u in all_units]
    sharpes = [u["sharpe"] for u in all_units]
    import numpy as np
    ics = np.array(ics)
    sharpes = np.array(sharpes)

    print(f"\nIC distribution:")
    print(f"  mean={ics.mean():.4f}, std={ics.std():.4f}")
    print(f"  min={ics.min():.4f}, max={ics.max():.4f}")
    print(f"  P5={np.percentile(ics,5):.4f}, P25={np.percentile(ics,25):.4f}")
    print(f"  P50={np.percentile(ics,50):.4f}, P75={np.percentile(ics,75):.4f}")
    print(f"  P95={np.percentile(ics,95):.4f}")

    print(f"\nSharpe distribution:")
    print(f"  mean={sharpes.mean():.4f}, std={sharpes.std():.4f}")
    print(f"  min={sharpes.min():.4f}, max={sharpes.max():.4f}")

    # ═══════════════════════════════════════════
    # KEY METRIC: |IC| < THRESHOLD but Sharpe > 0
    # ═══════════════════════════════════════════

    # Primary condition
    low_ic = np.abs(ics) < IC_THRESHOLD
    positive_sharpe = sharpes > SHARPE_THRESHOLD
    both = low_ic & positive_sharpe

    n_low_ic = low_ic.sum()
    n_pos_sharpe = positive_sharpe.sum()
    n_both = both.sum()

    print(f"\n{'='*70}")
    print(f"PRIMARY METRIC: |IC| < {IC_THRESHOLD} AND Sharpe > {SHARPE_THRESHOLD}")
    print(f"{'='*70}")
    print(f"  |IC| < {IC_THRESHOLD}:              {n_low_ic}/{n_total} = {100*n_low_ic/n_total:.1f}%")
    print(f"  Sharpe > {SHARPE_THRESHOLD}:              {n_pos_sharpe}/{n_total} = {100*n_pos_sharpe/n_total:.1f}%")
    print(f"  BOTH:                          {n_both}/{n_total} = {100*n_both/n_total:.1f}%")

    # Breakdown by source
    print(f"\n  Breakdown by source:")
    for s in sorted(source_counts.keys()):
        mask = np.array([u["source"] == s for u in all_units])
        n_s = mask.sum()
        n_b = (mask & both).sum()
        if n_s > 0:
            print(f"    {s}: {n_b}/{n_s} = {100*n_b/n_s:.1f}%")

    # IC range of mismatched units
    mismatched_ics = ics[both]
    mismatched_sharpes = sharpes[both]
    if len(mismatched_ics) > 0:
        print(f"\n  Mismatched units IC range: [{mismatched_ics.min():.4f}, {mismatched_ics.max():.4f}]")
        print(f"  Mismatched units IC mean: {mismatched_ics.mean():.4f}")
        print(f"  Mismatched units Sharpe range: [{mismatched_sharpes.min():.2f}, {mismatched_sharpes.max():.2f}]")
        print(f"  Mismatched units Sharpe mean: {mismatched_sharpes.mean():.2f}")

    # ─── Alternative thresholds for sensitivity ───
    print(f"\n{'='*70}")
    print(f"SENSITIVITY ANALYSIS: different IC thresholds")
    print(f"{'='*70}")
    for thresh in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
        low = np.abs(ics) < thresh
        n_low = low.sum()
        n_b = (low & positive_sharpe).sum()
        pct = 100 * n_b / n_total
        print(f"  |IC| < {thresh:.3f}:  {n_low}/{n_total} low IC,  {n_b}/{n_total} = {pct:.1f}% mismatched")

    # ─── IC vs Sharpe scatter summary ───
    print(f"\n{'='*70}")
    print(f"IC-SHARPE CORRELATION")
    print(f"{'='*70}")
    ic_sh_corr = np.corrcoef(ics, sharpes)[0, 1]
    print(f"  Pearson r(IC, Sharpe) = {ic_sh_corr:.4f}")

    # ─── Per-source tabular output ───
    print(f"\n{'='*70}")
    print(f"DETAILED TABLE: all sources with IC/Sharpe stats")
    print(f"{'='*70}")
    print(f"{'Source':<30s} {'Units':>6s} {'IC mean':>8s} {'IC std':>8s} {'Sh mean':>8s} {'|IC|<0.02 & Sh>0':>16s}")
    print("-" * 80)
    for s in sorted(source_counts.keys()):
        mask = np.array([u["source"] == s for u in all_units])
        si = ics[mask]
        ss = sharpes[mask]
        n_s = mask.sum()
        n_b = (mask & both).sum()
        print(f"{s:<30s} {n_s:>6d} {si.mean():>8.4f} {si.std():>8.4f} {ss.mean():>8.4f} {n_b:>6d} ({100*n_b/n_s:5.1f}%)")

    # ─── Compute per-model/per-source details ───
    print(f"\n{'='*70}")
    print(f"FINER BREAKDOWN: by source + model")
    print(f"{'='*70}")
    model_groups = defaultdict(list)
    for u in all_units:
        key = f"{u['source']}/{u['model']}"
        model_groups[key].append(u)
    
    print(f"{'Group':<50s} {'Units':>6s} {'IC±':>16s} {'Sh±':>16s} {'Mismatch':>10s}")
    print("-" * 100)
    for key in sorted(model_groups.keys()):
        units = model_groups[key]
        si = np.array([u["ic"] for u in units])
        ss = np.array([u["sharpe"] for u in units])
        n_u = len(units)
        n_b = ((np.abs(si) < IC_THRESHOLD) & (ss > SHARPE_THRESHOLD)).sum()
        pct = 100 * n_b / n_u if n_u > 0 else 0
        print(f"{key:<50s} {n_u:>6d} {si.mean():>+7.4f}±{si.std():.4f} {ss.mean():>+7.2f}±{ss.std():.2f} {n_b:>4d}({pct:5.1f}%)")

    # Save full data for reproducibility
    out_path = os.path.join(BASE, "sharpe_ic_mismatch.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_total": n_total,
            "n_low_ic": int(n_low_ic),
            "n_positive_sharpe": int(n_pos_sharpe),
            "n_both": int(n_both),
            "pct_mismatch": round(100 * n_both / n_total, 1),
            "ic_threshold": IC_THRESHOLD,
            "sharpe_threshold": SHARPE_THRESHOLD,
            "ic_vs_sharpe_correlation": round(float(ic_sh_corr), 4),
            "ic_stats": {
                "mean": float(ics.mean()), "std": float(ics.std()),
                "min": float(ics.min()), "max": float(ics.max()),
            },
            "sharpe_stats": {
                "mean": float(sharpes.mean()), "std": float(sharpes.std()),
                "min": float(sharpes.min()), "max": float(sharpes.max()),
            },
            "units": all_units,
            "sensitivity": {
                str(t): round(100 * ((np.abs(ics) < t) & positive_sharpe).sum() / n_total, 1)
                for t in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
            }
        }, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
