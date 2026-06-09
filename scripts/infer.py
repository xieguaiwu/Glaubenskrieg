"""Standalone inference pipeline for CTM + GBDT ensemble.

Loads trained CTM and GBDT models from disk, runs inference on new data,
produces fused signals + feature importance + confidence metrics.

Usage:
    PYTHONPATH=. python scripts/infer.py \\
        --ctm-ckpt checkpoints/best.pt \\
        --gbdt-json models/gbdt_model.json \\
        --config configs/default.yaml \\
        --data data/new_data.csv \\
        --output results/predictions.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.data.features import compute_all_features
from src.data.dataset import create_sequences
from src.model.ctm_model import CTMStockModel
from src.model.multiasset_ctm import MultiAssetCTM
from src.model.loop_ctm import RecurrentCTM
from src.model.ensemble import EnsembleConfig, EnsembleSignal, evaluate_ensemble
from src.utils.serialization import load_ctm_model

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def _nan_safe_output(array: np.ndarray, name: str = "array") -> np.ndarray:
    """Replace NaN/Inf values with 0.0 and log a warning."""
    nan_mask = ~np.isfinite(array)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        logger.warning("%s contains %d NaN/Inf values — replacing with 0.0", name, n_nan)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


# ═══════════════════════════════════════════════════════════════════════════════
# Model builder
# ═══════════════════════════════════════════════════════════════════════════════


def _build_model_params(
    cfg: Dict[str, Any],
    input_dim: int,
    model_type: str,
    n_loop_iters: int = 1,
    n_assets: int = 1,
    use_cross_attention: bool = True,
) -> Dict[str, Any]:
    """Build model constructor params from config + overrides.

    Parameters
    ----------
    cfg : full config dict.
    input_dim : number of input features (detected from data).
    model_type : one of ``"ctm"``, ``"recurrent"``, ``"multiasset"``.
    n_loop_iters : required when ``model_type == "recurrent"``.
    n_assets : required when ``model_type == "multiasset"``.
    use_cross_attention : cross-asset attention toggle for multiasset.
    """
    model_cfg = cfg.get("model", {})
    scaling_cfg = cfg.get("scaling", {})

    common = {
        "input_dim": input_dim,
        "model_dim": model_cfg.get("model_dim", 64),
        "state_dim": model_cfg.get("state_dim", 16),
        "conv_kernel": model_cfg.get("conv_kernel", 3),
        "n_layers": model_cfg.get("n_layers", 3),
        "output_dim": model_cfg.get("output_dim", 1),
        "use_decomp": model_cfg.get("use_decomp", False),
        "bidirectional": model_cfg.get("bidirectional", False),
        "parallel_scan": model_cfg.get("parallel_scan", False),
    }

    if model_type == "multiasset":
        return {
            "n_assets": n_assets or scaling_cfg.get("n_assets", 1),
            "input_dim": input_dim,
            "model_dim": model_cfg.get("model_dim", 64),
            "state_dim": model_cfg.get("state_dim", 16),
            "n_layers": model_cfg.get("n_layers", 3),
            "output_dim": model_cfg.get("output_dim", 1),
            "dropout": model_cfg.get("dropout", 0.1),
            "use_cross_attention": use_cross_attention,
        }
    elif model_type == "recurrent":
        return {
            **common,
            "dropout": 0.0,  # no dropout at inference
            "n_loop_iters": n_loop_iters,
            "loop_dropout": 0.0,
        }
    else:  # ctm
        return {
            **common,
            "dropout": 0.0,
            "return_hidden": False,
        }


def _resolve_model_type(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
) -> Tuple[str, int, int]:
    """Determine model type, n_loop_iters, n_assets from CLI + config.

    Returns (model_type, n_loop_iters, n_assets).
    """
    model_cfg = cfg.get("model", {})
    scaling_cfg = cfg.get("scaling", {})

    # CLI override takes precedence
    if args.model_type:
        model_type = args.model_type
    elif scaling_cfg.get("n_assets", 1) > 1:
        model_type = "multiasset"
    elif model_cfg.get("n_loop_iters", 1) > 1:
        model_type = "recurrent"
    else:
        model_type = "ctm"

    n_loop_iters = args.n_loop if args.n_loop is not None else model_cfg.get("n_loop_iters", 1)
    if model_type == "recurrent":
        n_loop_iters = max(n_loop_iters, 2)

    n_assets = args.n_assets or scaling_cfg.get("n_assets", 1)

    return model_type, n_loop_iters, n_assets


# ═══════════════════════════════════════════════════════════════════════════════
# Inference runners
# ═══════════════════════════════════════════════════════════════════════════════


def _run_ctm_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    extract_hidden: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run CTM/RecurrentCTM inference over a DataLoader.

    Returns (predictions, hidden_states | None).
    predictions shape: (N,) — last-timestep first regression channel.
    hidden_states shape: (N, seq_len, model_dim) or None.
    """
    preds: List[np.ndarray] = []
    hiddens: List[np.ndarray] = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            if extract_hidden:
                # Use return_hidden for single-pass models; RecurrentCTM
                # returns the full-loop hidden state via this path.
                output, hidden = model(batch_x, return_hidden=True)  # type: ignore[call-arg]
                hiddens.append(hidden.cpu().numpy())
            else:
                output = model(batch_x)

            # output shape: (B, T, num_output_channels), first channel = regression
            if output.dim() == 3:
                pred = output[:, -1, 0].cpu().numpy()
            else:
                pred = output.cpu().numpy().ravel()
            preds.append(pred)

    preds_arr = np.concatenate(preds)
    preds_arr = _nan_safe_output(preds_arr, "CTM predictions")

    hiddens_arr: Optional[np.ndarray] = None
    if hiddens:
        hiddens_arr = np.concatenate(hiddens, axis=0)
        hiddens_arr = _nan_safe_output(hiddens_arr, "CTM hidden states")

    return preds_arr, hiddens_arr


def _run_multiasset_inference(
    model: MultiAssetCTM,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_assets: int,
) -> np.ndarray:
    """Run MultiAssetCTM inference, extracting per-asset regression predictions.

    Returns (N, n_assets) predictions array (last-timestep, per-asset reg channel).
    """
    preds: List[np.ndarray] = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            # MultiAssetCTM expects (B, N, T, D) — the loader gives (B, T, D).
            # Expand to (B, 1, T, D) if needed, or replicate.
            if batch_x.dim() == 3:
                # Single-asset input → replicate across assets
                batch_x = batch_x.unsqueeze(1).expand(-1, n_assets, -1, -1)
            output = model(batch_x)
            # output shape: (B, T, N * num_output_channels)
            # regression channels are first N*output_dim per time step
            B, T = output.shape[:2]
            output_dim = model.output_dim
            reg_out = output[:, -1, : n_assets * output_dim]  # last step, reg only
            reg_out = reg_out.reshape(B, n_assets, output_dim)
            # Take first reg dim per asset
            preds.append(reg_out[:, :, 0].cpu().numpy())

    preds_arr = np.concatenate(preds, axis=0)
    return _nan_safe_output(preds_arr, "MultiAssetCTM predictions")


# ═══════════════════════════════════════════════════════════════════════════════
# GBDT loading
# ═══════════════════════════════════════════════════════════════════════════════


def _try_load_gbdt(
    gbdt_json: str,
    gbdt_build_dir: Optional[str],
) -> Tuple[Optional[Any], Optional[Any]]:
    """Attempt to load a GBDT model from JSON. Returns (model, GBDTConfig) or (None, None)."""
    if gbdt_build_dir and gbdt_build_dir not in sys.path:
        sys.path.insert(0, gbdt_build_dir)

    try:
        from gbdt import GBDT  # type: ignore[import-untyped]
        from gbdt_python import GBDTConfig  # type: ignore[import-not-found]

        logger.info("Loading GBDT model from %s", gbdt_json)
        with open(gbdt_json) as f:
            json_str = f.read()
        model = GBDT(GBDTConfig())
        model.from_json(json_str)
        return model, None
    except ImportError as e:
        logger.warning("GBDT import failed (%s) — running CTM-only inference", e)
    except FileNotFoundError:
        logger.warning("GBDT JSON not found: %s — running CTM-only inference", gbdt_json)
    except Exception as e:
        logger.warning("GBDT load failed (%s) — running CTM-only inference", e)
    return None, None


def _extract_gbdt_importance(
    gbdt_model: Any,
    num_features: int,
) -> Optional[Dict[str, List[float]]]:
    """Extract feature importance from a GBDT model.

    Returns dict with 'frequency', 'gain', 'coverage' lists, or None on failure.
    """
    try:
        imp = gbdt_model.get_feature_importance_full(num_features)
        importance = {
            "frequency": list(imp["frequency"]),
            "gain": list(imp["gain"]),
            "coverage": list(imp["coverage"]),
        }
        top_k = np.argsort(imp["gain"])[-5:][::-1]
        logger.info("Top-5 GBDT features (by gain): %s", list(top_k))
        return importance
    except Exception as e:
        logger.warning("Feature importance extraction failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="CTM + GBDT ensemble inference")
    parser.add_argument("--ctm-ckpt", required=True, help="Path to CTM model .pt checkpoint")
    parser.add_argument("--gbdt-json", default=None, help="Path to GBDT model .json (omit for CTM-only)")
    parser.add_argument("--gbdt-build-dir", default=None, help="Hoffnung build dir (for GBDT import)")
    parser.add_argument("--config", default="configs/default.yaml", help="Model config YAML")
    parser.add_argument("--data", required=True, help="CSV with features (same columns as training)")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--output", default="results/predictions.csv", help="Output CSV path")
    parser.add_argument("--num-features", type=int, default=None, help="Number of features for GBDT importance")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length (from config if omitted)")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--model-type", choices=["ctm", "recurrent", "multiasset"],
                        default=None, help="Force model type")
    parser.add_argument("--n-loop", type=int, default=None, help="Override loop iterations for RecurrentCTM")
    parser.add_argument("--n-assets", type=int, default=None, help="Number of assets for MultiAssetCTM")
    parser.add_argument("--ctm-weight", type=float, default=0.5, help="Static CTM weight for ensemble fusion")
    parser.add_argument("--gbdt-weight", type=float, default=0.5, help="Static GBDT weight for ensemble fusion")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(args.device)

    # ── 1. Load config ────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    seq_len = args.seq_len or model_cfg.get("seq_len", 63)

    # ── 2. Load & prepare data ────────────────────────────────────────────────
    logger.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data, index_col=0, parse_dates=True)

    features_df = compute_all_features(df)
    features_df = features_df.dropna()

    feature_cols = list(features_df.columns)
    feature_array = features_df[feature_cols].values.astype(np.float32)
    input_dim = len(feature_cols)
    logger.info("Loaded %d samples, %d features", len(features_df), input_dim)

    # ── 3. Create sequences ───────────────────────────────────────────────────
    data_seq = create_sequences(feature_array, seq_len)
    if len(data_seq) == 0:
        logger.error("Not enough data to create sequences (data_rows=%d, seq_len=%d)",
                      len(feature_array), seq_len)
        sys.exit(1)
    logger.info("Created %d sequences of length %d", len(data_seq), seq_len)

    data_tensor = torch.from_numpy(data_seq).float()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data_tensor),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ── 4. Resolve model type & build params ──────────────────────────────────
    model_type, n_loop_iters, n_assets = _resolve_model_type(args, cfg)
    logger.info("Model type: %s (n_loop=%d, n_assets=%d)", model_type, n_loop_iters, n_assets)

    model_params = _build_model_params(cfg, input_dim, model_type, n_loop_iters, n_assets)

    if model_type == "multiasset":
        model_class: type = MultiAssetCTM
    elif model_type == "recurrent":
        model_class = RecurrentCTM
    else:
        model_class = CTMStockModel

    # ── 5. Load CTM model ─────────────────────────────────────────────────────
    logger.info("Loading CTM model (%s) from %s", model_class.__name__, args.ctm_ckpt)
    ctm_model = load_ctm_model(model_class, model_params, args.ctm_ckpt, device)
    ctm_model.eval()

    # ── 6. Run CTM inference ──────────────────────────────────────────────────
    want_hidden = args.gbdt_json is not None and model_type != "multiasset"
    if model_type == "multiasset":
        assert isinstance(ctm_model, MultiAssetCTM)
        ctm_preds = _run_multiasset_inference(ctm_model, loader, device, n_assets)
        ctm_hidden_np = None
        # For ensemble fusion, take mean across assets as aggregate signal
        if ctm_preds.ndim == 2:
            ctm_preds_1d = ctm_preds.mean(axis=1)
        else:
            ctm_preds_1d = ctm_preds
    else:
        ctm_preds_1d, ctm_hidden_np = _run_ctm_inference(
            ctm_model, loader, device, extract_hidden=want_hidden,
        )

    logger.info("CTM predictions: %d samples, mean=%.4f, std=%.4f",
                 len(ctm_preds_1d), float(ctm_preds_1d.mean()), float(ctm_preds_1d.std()))

    # ── 7. Load & run GBDT (if available) ─────────────────────────────────────
    gbdt_preds: Optional[np.ndarray] = None
    gbdt_importance: Optional[Dict[str, List[float]]] = None

    if args.gbdt_json is not None:
        gbdt_model, _ = _try_load_gbdt(args.gbdt_json, args.gbdt_build_dir)

        if gbdt_model is not None:
            from src.data.gbdt_features import build_gbdt_feature_matrix

            # Build GBDT feature matrix: aggregated raw features + CTM hidden
            if ctm_hidden_np is not None and ctm_hidden_np.size > 0:
                X_gbdt = build_gbdt_feature_matrix(
                    data_seq, ctm_hidden=ctm_hidden_np, include_ctm_features=True,
                )
            else:
                X_gbdt = build_gbdt_feature_matrix(data_seq, include_ctm_features=False)

            raw_preds = gbdt_model.predict(X_gbdt)
            if raw_preds is None:
                logger.warning("GBDT predict() returned None — skipping GBDT")
                gbdt_preds = None
            else:
                gbdt_preds = _nan_safe_output(np.asarray(raw_preds, dtype=np.float64).ravel(),
                                              "GBDT predictions")
                logger.info("GBDT predictions: %d samples, mean=%.4f, std=%.4f",
                             len(gbdt_preds), float(gbdt_preds.mean()), float(gbdt_preds.std()))

            # Feature importance
            num_feat = args.num_features or X_gbdt.shape[1]
            gbdt_importance = _extract_gbdt_importance(gbdt_model, num_feat)

    # ── 8. Ensemble fusion ────────────────────────────────────────────────────
    if gbdt_preds is not None and len(gbdt_preds) == len(ctm_preds_1d):
        # Clip GBDT preds to match CTM preds length (edge case)
        min_len = min(len(ctm_preds_1d), len(gbdt_preds))
        ctm_for_fusion = ctm_preds_1d[:min_len]
        gbdt_for_fusion = gbdt_preds[:min_len]

        # Static-weight fusion (no ground-truth labels at inference time)
        ensemble_cfg = EnsembleConfig(
            use_ic_weighting=False,
            ctm_weight=args.ctm_weight,
            gbdt_weight=args.gbdt_weight,
        )
        dummy_labels = np.zeros_like(ctm_for_fusion)
        try:
            result: EnsembleSignal = evaluate_ensemble(
                ctm_for_fusion, gbdt_for_fusion, dummy_labels, ensemble_cfg,
            )
            fused_signal = result.fused
            logger.info("Fused signal: mean=%.4f, std=%.4f", float(fused_signal.mean()), float(fused_signal.std()))
        except Exception as e:
            logger.warning("Ensemble fusion failed (%s), falling back to simple average", e)
            total = args.ctm_weight + args.gbdt_weight
            cw = args.ctm_weight / total if total > 0 else 0.5
            gw = args.gbdt_weight / total if total > 0 else 0.5
            fused_signal = cw * ctm_for_fusion + gw * gbdt_for_fusion
    else:
        fused_signal = ctm_preds_1d
        logger.info("CTM-only mode — fused signal = CTM predictions")

    # ── 9. Write output ───────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Align index: the last len(fused_signal) rows of features_df correspond
    # to the prediction targets.
    output_index = features_df.index[-len(fused_signal):]

    output_df = pd.DataFrame({
        "ctm_prediction": ctm_preds_1d[:len(fused_signal)],
        "fused_signal": fused_signal,
        "confidence": np.abs(fused_signal),
        "rank": np.argsort(np.argsort(-fused_signal)) + 1,
    }, index=output_index)

    if gbdt_preds is not None:
        output_df["gbdt_prediction"] = gbdt_preds[:len(fused_signal)]

    output_df.to_csv(args.output)
    logger.info("Predictions saved to %s (%d rows)", args.output, len(output_df))

    # ── 10. Save importance ───────────────────────────────────────────────────
    if gbdt_importance is not None:
        imp_path = args.output.replace(".csv", "_importance.json")
        with open(imp_path, "w") as f:
            json.dump(gbdt_importance, f, indent=2)
        logger.info("Feature importance saved to %s", imp_path)


if __name__ == "__main__":
    main()
