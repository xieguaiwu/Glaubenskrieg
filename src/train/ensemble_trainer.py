"""Ensemble trainer integrating GBDT with CTM in walk-forward training loop.

Adds GBDT (Gradient Boosted Decision Tree) training and IC-weighted
ensemble fusion to the CTM walk-forward validation pipeline.

Supports loading pre-trained CTM and GBDT weights for warm-start
training or inference-only ensemble fusion.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..data.gbdt_features import build_gbdt_feature_matrix, GBDTFeatureConfig
from ..model.ensemble import (
    compute_ic,
    evaluate_ensemble,
    EnsembleConfig,
    rankic_loss_gbdt_style,
)
from ..model.losses import LossConfig, composite_loss
from ..utils.metrics import sharpe_ratio_torch
from ._walk_forward_utils import TrainingResult, walk_forward_windows
from .advanced_trainer import WalkForwardTrainerAdvanced


# ═══════════════════════════════════════════════════════════════════
# GBDT module availability checks
# ═══════════════════════════════════════════════════════════════════

_GBDT_CPP_AVAILABLE = False
_GBDT_PYTHON_TRAINER_AVAILABLE = False

try:
    from gbdt_python import GBDT, GBDTConfig  # type: ignore[import-not-found]
    _GBDT_CPP_AVAILABLE = True
except ImportError:
    GBDT = None  # type: ignore[assignment, misc]
    GBDTConfig = None  # type: ignore[assignment, misc]

try:
    from gbdt import GBDTTrainer as _GBDTTrainerCls  # type: ignore[import-not-found]
    from gbdt.losses import rankic_loss as _gbdt_rankic_loss  # type: ignore[import-not-found]
    _GBDT_PYTHON_TRAINER_AVAILABLE = True
except ImportError:
    _GBDTTrainerCls = None
    _gbdt_rankic_loss = None


def _compute_sharpe(preds: np.ndarray, annual_factor: float = 252.0) -> float:
    """Annualised Sharpe ratio from a numpy prediction array."""
    preds = np.asarray(preds, dtype=np.float64).ravel()
    if len(preds) < 2:
        return 0.0
    t = torch.from_numpy(preds).float().unsqueeze(0)  # (1, N) so time dim at -1
    return float(sharpe_ratio_torch(t, annual_factor=annual_factor, ddof=1).item())


def _safe_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute IC with NaN/inf handling. Returns 0.0 on failure."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return 0.0
    return compute_ic(y_true[valid], y_pred[valid])


class EnsembleWalkForwardTrainer:
    """Walk-forward trainer integrating GBDT with CTM via IC-weighted fusion.

    At each walk-forward window:
    1. Train CTM model (same as :class:`WalkForwardTrainerAdvanced`).
    2. Extract CTM hidden states and build GBDT feature matrix.
    3. Train GBDT (native C++ or Python-based custom loss).
    4. Fuse CTM and GBDT predictions via rolling IC-weighted average.
    5. Report CTM-only, GBDT-only, and ensemble metrics.

    Supports loading pre-trained CTM and/or GBDT weights.  When
    ``skip_ctm_training`` or ``skip_gbdt_training`` is set, the
    corresponding stage is replaced by loading the pre-trained model.

    Parameters
    ----------
    model_class : nn.Module class
        CTM model class.
    model_params : dict
        Keyword arguments for model constructor.
    loss_config : LossConfig
        Composite loss configuration for CTM training.
    gbdt_config : dict or None
        GBDT hyperparameters. Keys match GBDTConfig attributes
        (num_trees, max_depth, learning_rate, subsample_col, etc.).
        If None, defaults from the C++ core are used.
    gbdt_loss : str
        GBDT loss type: ``"mse"``, ``"mae"``, ``"huber"``, or
        ``"rankic"``.  For ``"rankic"`` the Python ``GBDTTrainer``
        wrapper is preferred; falls back to a manual boosting loop
        with differentiable RankIC when wrapper unavailable.
    device : torch.device
    pretrained_ctm_state_dict : OrderedDict or None, default=None
        Pre-trained CTM ``state_dict()``.  When provided alongside
        ``skip_ctm_training=True``, CTM training is skipped and this
        state dict is used directly.  When ``skip_ctm_training=False``
        (default), the state dict is used as a warm-start initialisation
        for the first window's CTM model.
    pretrained_gbdt_json : str or None, default=None
        Path to a pre-trained GBDT JSON model file.  When provided
        alongside ``skip_gbdt_training=True``, GBDT training is skipped
        and this model is loaded from JSON.  When
        ``skip_gbdt_training=False`` (default), this parameter is
        ignored and GBDT is trained from scratch each window.
    skip_ctm_training : bool, default=False
        If True and ``pretrained_ctm_state_dict`` is provided, skip
        CTM training entirely and use the pre-trained model for all
        windows.  The CTM model is loaded once and shared across all
        walk-forward windows.
    skip_gbdt_training : bool, default=False
        If True and ``pretrained_gbdt_json`` is provided, skip GBDT
        training and load the pre-trained GBDT model from JSON.  The
        GBDT model is loaded once and shared across all windows.
    """

    def __init__(
        self,
        model_class: Type[nn.Module],
        model_params: Dict[str, Any],
        loss_config: LossConfig,
        gbdt_config: Optional[Dict[str, Any]] = None,
        gbdt_loss: str = "mse",
        device: torch.device | str = "cpu",
        loss_fn: Optional[Callable] = None,
        log_gradients: bool = False,
        pretrained_ctm_state_dict: Optional[OrderedDict] = None,
        pretrained_gbdt_json: Optional[str] = None,
        skip_ctm_training: bool = False,
        skip_gbdt_training: bool = False,
        gate_finetune_epochs: int = 0,
        gate_finetune_lr: float = 1e-4,
        freeze_ctm_backbone_for_gate: bool = True,
        gate_finetune_batch_size: int = 32,
        gate_finetune_patience: int = 5,
    ) -> None:
        self.model_class = model_class
        self.model_params = model_params
        self.loss_config = loss_config
        self.gbdt_config = gbdt_config if gbdt_config is not None else {}
        self.gbdt_loss = gbdt_loss
        self.device = torch.device(device)
        self.loss_fn = loss_fn
        self._last_gbdt_model: Any = None
        self._pretrained_ctm_sd = pretrained_ctm_state_dict
        self._pretrained_gbdt_path = pretrained_gbdt_json
        self._skip_ctm = skip_ctm_training
        self._skip_gbdt = skip_gbdt_training
        self._loaded_ctm_model: Optional[nn.Module] = None
        self._loaded_gbdt_model: Any = None
        self.gate_finetune_epochs = gate_finetune_epochs
        self.gate_finetune_lr = gate_finetune_lr
        self.freeze_ctm_backbone_for_gate = freeze_ctm_backbone_for_gate
        self.gate_finetune_batch_size = gate_finetune_batch_size
        self.gate_finetune_patience = gate_finetune_patience
        # Per-window best CTM state dicts for test-prep warm-start (P0 Fix 2)
        self._window_best_ctm_states: List[Dict[str, torch.Tensor]] = []

    def get_best_ctm_state_dict(self, idx: int) -> Dict[str, torch.Tensor] | None:
        """Return the best CTM state dict from walk-forward window ``idx``.

        Returns None if ``idx`` is out of range or no state dicts are stored.
        """
        if not self._window_best_ctm_states or idx < 0 or idx >= len(self._window_best_ctm_states):
            return None
        return self._window_best_ctm_states[idx]

    # ── Public API ─────────────────────────────────────────────

    def train_walk_forward(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        train_window: int,
        val_window: int,
        purge_period: int,
        step_size: int,
        n_epochs: int,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        patience: int = 10,
        warmup_steps: int = 2000,
        ramp_steps: int = 3000,
        lr_warmup_epochs: int = 5,
        class_targets_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        log_gradients: bool = False,
    ) -> List[TrainingResult]:
        """Run walk-forward training with CTM + GBDT ensemble.

        Parameters
        ----------
        data : (N, seq_len, D) pre-sequenced features.
        targets : (N, T, output_dim) or (N,) regression targets.
        train_window, val_window : fold sizes in samples.
        purge_period : gap between train and val to prevent leakage.
        step_size : walk-forward stride.

        Returns
        -------
        List of ``TrainingResult`` dataclass instances, one per window.
        """
        N = len(data)
        results: List[TrainingResult] = []

        if targets.dim() == 1:
            targets = targets.unsqueeze(-1).unsqueeze(-1)
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)

        for pos, train_end, purge_end, val_end in walk_forward_windows(
            N, train_window, val_window, purge_period, step_size
        ):
            train_data = data[pos:train_end]
            train_targ = targets[pos:train_end]
            val_data = data[purge_end:val_end]
            val_targ = targets[purge_end:val_end]

            train_loader = DataLoader(
                TensorDataset(train_data, train_targ),
                batch_size=batch_size,
                shuffle=False,
            )
            val_loader = DataLoader(
                TensorDataset(val_data, val_targ),
                batch_size=batch_size,
                shuffle=False,
            )

            state = self._run_window_stages_1_2(
                train_loader=train_loader,
                val_loader=val_loader,
                train_data=train_data,
                train_targ=train_targ,
                val_data=val_data,
                val_targ=val_targ,
                n_epochs=n_epochs,
                lr=lr,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                patience=patience,
                warmup_steps=warmup_steps,
                ramp_steps=ramp_steps,
                lr_warmup_epochs=lr_warmup_epochs,
                class_targets_fn=class_targets_fn,
                log_gradients=log_gradients,
            )

            ctm_model = state["ctm_model"]
            gbdt_model = state["gbdt_model"]
            X_gbdt_val = state["X_gbdt_val"]
            y_gbdt_val = state["y_gbdt_val"]

            # Store best CTM state dict for test-prep warm-start (P0 Fix 2)
            self._window_best_ctm_states.append(
                {k: v.cpu().clone() for k, v in ctm_model.state_dict().items()}
            )

            # ── 4. Get predictions for fusion ──────────────────
            with torch.no_grad():
                ctm_output = ctm_model(val_data.to(self.device))
                # multi-asset: output=(B,T,N*C); extract regression dim at last step, flatten to 1D
                n_assets_ctm = getattr(ctm_model, 'n_assets', 1)
                if ctm_output.dim() >= 3 and ctm_output.size(-1) >= n_assets_ctm:
                    ctm_pred_val = ctm_output[:, -1, :n_assets_ctm].reshape(-1).cpu().numpy()
                else:
                    ctm_pred_val = ctm_output[:, -1, 0].cpu().numpy()

            if gbdt_model is not None:
                gbdt_pred_val = gbdt_model.predict(X_gbdt_val)
                if np.isnan(gbdt_pred_val).any():
                    warnings.warn(
                        f"NaN in GBDT predictions at window {pos}. "
                        "Replacing with 0."
                    )
                    gbdt_pred_val = np.nan_to_num(gbdt_pred_val, nan=0.0)
                if np.std(gbdt_pred_val) < 1e-8:
                    warnings.warn(
                        f"GBDT predictions are near-constant at window {pos} "
                        f"(std={np.std(gbdt_pred_val):.2e}). GBDT signal is degenerate."
                    )
            else:
                gbdt_pred_val = np.zeros_like(ctm_pred_val)
                warnings.warn(
                    "GBDT model unavailable — ensemble predictions degraded "
                    "(GBDT predictions zeroed). Install Hoffnung for GBDT support."
                )

            # ── 5. IC-weighted ensemble fusion ─────────────────
            n_val = len(y_gbdt_val)
            ensemble_cfg = EnsembleConfig(
                use_ic_weighting=True,
                ic_lookback=min(252, max(5, n_val // 2)),
                min_samples_for_ic=min(20, n_val),
            )
            ensemble_result = evaluate_ensemble(
                ctm_pred_val, gbdt_pred_val, y_gbdt_val, ensemble_cfg,
            )

            ctm_sharpe = _compute_sharpe(np.sign(ctm_pred_val) * y_gbdt_val)
            gbdt_sharpe = _compute_sharpe(np.sign(gbdt_pred_val) * y_gbdt_val) if gbdt_pred_val is not None else None
            ctm_ic = float(ensemble_result.ic_ctm)
            ensemble_sharpe = _compute_sharpe(np.sign(ensemble_result.fused) * y_gbdt_val)

            # Directional accuracy (same as validate() ±0.5% tolerance)
            y_v = y_gbdt_val
            eps_dir = 0.005
            ctm_dir_acc = float(np.mean(((ctm_pred_val < -eps_dir) == (y_v < -eps_dir)) | ((ctm_pred_val > eps_dir) == (y_v > eps_dir))))
            if gbdt_pred_val is not None:
                gbdt_dir_acc = float(np.mean(((gbdt_pred_val < -eps_dir) == (y_v < -eps_dir)) | ((gbdt_pred_val > eps_dir) == (y_v > eps_dir))))
            else:
                gbdt_dir_acc = None
            fused = ensemble_result.fused
            ensemble_dir_acc = float(np.mean(((fused < -eps_dir) == (y_v < -eps_dir)) | ((fused > eps_dir) == (y_v > eps_dir))))

            ctm_w = np.nanmean(ensemble_result.weights["ctm_weight"]) if len(ensemble_result.weights["ctm_weight"]) else np.nan
            gbdt_w = np.nanmean(ensemble_result.weights["gbdt_weight"]) if len(ensemble_result.weights["gbdt_weight"]) else np.nan
            avg_ctm_w = 0.5 if np.isnan(ctm_w) else float(ctm_w)
            avg_gbdt_w = 0.5 if np.isnan(gbdt_w) else float(gbdt_w)

            # ── 6. Record window results ───────────────────────
            results.append(TrainingResult(
                window_start=pos,
                window_end=val_end,
                best_sharpe=state["best_sharpe"],
                epochs_run=state["epochs_run"],
                metrics=state["window_metrics"],
                ctm_sharpe=ctm_sharpe,
                ctm_ic=ctm_ic,
                ctm_dir_acc=ctm_dir_acc,
                gbdt_sharpe=gbdt_sharpe,
                gbdt_ic=_safe_ic(y_gbdt_val, gbdt_pred_val),
                gbdt_dir_acc=gbdt_dir_acc,
                ensemble_sharpe=ensemble_sharpe,
                ensemble_ic=float(ensemble_result.fused_ic),
                ensemble_dir_acc=ensemble_dir_acc,
                ctm_weight=avg_ctm_w,
                gbdt_weight=avg_gbdt_w,
                ctm_metrics=list(state["window_metrics"]),
                gbdt_metrics=state["gbdt_metrics"],
                gbdt_importance=state["gbdt_importance"],
            ))
        return results

    # ── Public API: Ensemble inference with pre-trained weights ──

    def run_ensemble_inference(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int = 32,
        ic_lookback: int = 252,
    ) -> Dict[str, Any]:
        """Run inference-only ensemble fusion with pre-trained models.

        Loads pre-trained CTM and GBDT models (must be provided via
        ``__init__`` params), runs forward passes, fuses predictions
        via IC weighting, and returns summary metrics.

        Parameters
        ----------
        data : (N, seq_len, D) pre-sequenced features.
        targets : (N, T, output_dim) regression targets.
        batch_size : Batch size for CTM inference.
        ic_lookback : Lookback window for rolling IC computation.

        Returns
        -------
        dict with keys: ``"ensemble_sharpe"``, ``"ctm_sharpe"``, ``"gbdt_sharpe"``,
        ``"fused_ic"``, ``"ctm_ic"``, ``"gbdt_ic"``,
        ``"directional_accuracy"``, ``"ctm_directional_accuracy"``, ``"gbdt_directional_accuracy"``,
        ``"ctm_weight"``, ``"gbdt_weight"``, ``"n_windows"`` (1).
        """
        if self._pretrained_ctm_sd is None:
            raise ValueError(
                "run_ensemble_inference requires pretrained_ctm_state_dict "
                "in EnsembleWalkForwardTrainer.__init__"
            )
        if self._pretrained_gbdt_path is None:
            raise ValueError(
                "run_ensemble_inference requires pretrained_gbdt_json "
                "in EnsembleWalkForwardTrainer.__init__"
            )

        # ── 1. Load pre-trained CTM model (strict — fail on mismatch) ──
        ctm_model = self.model_class(**self.model_params).to(self.device)
        try:
            ctm_model.load_state_dict(self._pretrained_ctm_sd, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Pretrained CTM state_dict incompatible with model. "
                f"Model params: {self.model_params}. Error: {e}"
            )
        ctm_model.eval()
        logging.info("CTM model loaded for ensemble inference")

        # ── 2. Load pre-trained GBDT model ──
        try:
            from gbdt_python import GBDT, GBDTConfig
            gbdt_config = GBDTConfig()
            _apply_config(gbdt_config, self.gbdt_config)
            gbdt_model = GBDT(gbdt_config)
            with open(self._pretrained_gbdt_path) as f:
                gbdt_model.from_json(f.read())
            logging.info("GBDT model loaded from %s", self._pretrained_gbdt_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pre-trained GBDT from "
                f"{self._pretrained_gbdt_path}: {exc}"
            )

        # ── 3. Build data loader for CTM forward pass ──
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # ── 4. CTM forward pass → get predictions ──
        all_ctm_pred: List[np.ndarray] = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                output = ctm_model(batch_x)
                pred = output[:, -1, 0].cpu().numpy()
                all_ctm_pred.append(pred)
        ctm_pred = np.concatenate(all_ctm_pred)

        # ── 5. Extract CTM hidden states for GBDT features ──
        all_hidden: List[np.ndarray] = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                hidden = ctm_model.extract_features(batch_x)
                all_hidden.append(hidden.cpu().numpy())
        hidden_full = np.concatenate(all_hidden)

        data_np = data.cpu().numpy()
        target_np = targets.cpu().numpy()

        # ── 6. Build GBDT feature matrix ──
        include_ctm = self.gbdt_config.get("include_ctm_features", True)
        X_gbdt, y_gbdt = self._prepare_gbdt_data(
            data_np, target_np,
            ctm_hidden=hidden_full,
            include_ctm=include_ctm,
            gbdt_config=self.gbdt_config,
        )

        # ── 7. GBDT predict ──
        X_gbdt_f32 = np.asarray(X_gbdt, dtype=np.float32)
        gbdt_pred = gbdt_model.predict(X_gbdt_f32)

        # ── 7b. NaN sanitisation ──
        if np.isnan(gbdt_pred).any():
            warnings.warn("NaN in GBDT predictions during ensemble inference. Replacing with 0.")
            gbdt_pred = np.nan_to_num(gbdt_pred, nan=0.0)

        # ── 8. IC-weighted ensemble fusion ──
        n_val = len(y_gbdt)
        ensemble_cfg = EnsembleConfig(
            use_ic_weighting=True,
            ic_lookback=min(ic_lookback, max(5, n_val // 2)),
            min_samples_for_ic=min(20, n_val),
        )

        ensemble_result = evaluate_ensemble(
            ctm_pred, gbdt_pred, y_gbdt, ensemble_cfg,
        )

        # ── 9. Compute metrics ──
        ctm_sharpe = _compute_sharpe(np.sign(ctm_pred) * y_gbdt)
        gbdt_sharpe = _compute_sharpe(np.sign(gbdt_pred) * y_gbdt)
        ensemble_sharpe = _compute_sharpe(np.sign(ensemble_result.fused) * y_gbdt)
        ctm_ic = float(ensemble_result.ic_ctm)
        gbdt_ic = float(ensemble_result.ic_gbdt)
        fused_ic = float(ensemble_result.fused_ic)

        eps_dir = 0.005
        ctm_dir = float(np.mean(((ctm_pred < -eps_dir) == (y_gbdt < -eps_dir))
                                | ((ctm_pred > eps_dir) == (y_gbdt > eps_dir))))
        gbdt_dir = float(np.mean(((gbdt_pred < -eps_dir) == (y_gbdt < -eps_dir))
                                 | ((gbdt_pred > eps_dir) == (y_gbdt > eps_dir))))
        fused_dir = float(np.mean(((ensemble_result.fused < -eps_dir) == (y_gbdt < -eps_dir))
                                  | ((ensemble_result.fused > eps_dir) == (y_gbdt > eps_dir))))

        summary = {
            "n_windows": 1,
            "ctm_sharpe": ctm_sharpe,
            "gbdt_sharpe": gbdt_sharpe,
            "ensemble_sharpe": ensemble_sharpe,
            "ctm_ic": ctm_ic,
            "gbdt_ic": gbdt_ic,
            "fused_ic": fused_ic,
            "ctm_directional_accuracy": ctm_dir,
            "gbdt_directional_accuracy": gbdt_dir,
            "directional_accuracy": fused_dir,
            "ctm_weight": float(np.nanmean(ensemble_result.weights.get("ctm_weight", [0.5]))),
            "gbdt_weight": float(np.nanmean(ensemble_result.weights.get("gbdt_weight", [0.5]))),
        }

        self._last_gbdt_model = gbdt_model
        logging.info(
            "Ensemble inference: CTM sharpe=%.4f, GBDT sharpe=%.4f, "
            "fused_IC=%.4f, dir_acc=%.4f",
            ctm_sharpe, gbdt_sharpe, fused_ic, fused_dir,
        )
        return summary

    def explain(
        self, results: List[TrainingResult], top_k: int = 10
    ) -> Dict[str, Any]:
        """Return per-window feature importance summary.

        Parameters
        ----------
        results : list of TrainingResult from train_walk_forward()
        top_k : number of top features to return per window.

        Returns
        -------
        dict with:
            - "per_window": list of dicts per window with importance data
            - "aggregated": dict with mean importance across windows
            - "top_features": list of top-k feature indices across all windows
        """
        # Collect valid importance dicts
        valid_importances = [
            r.gbdt_importance
            for r in results
            if r.gbdt_importance is not None
        ]
        if not valid_importances:
            return {
                "error": "No feature importance data available. "
                "Train with ensemble mode."
            }

        per_window: List[Dict[str, Any]] = []
        agg_gain: Dict[int, float] = {}
        agg_freq: Dict[int, float] = {}
        agg_cov: Dict[int, float] = {}
        count: Dict[int, int] = {}

        for idx, imp in enumerate(valid_importances):
            gain = imp.get("gain", [])
            freq = imp.get("frequency", [])
            cov = imp.get("coverage", [])
            n_features = len(gain)

            top_indices = sorted(
                range(n_features), key=lambda i: gain[i], reverse=True
            )[:top_k]

            window_info = {
                "window_idx": idx,
                "n_features": n_features,
                "top_features": [
                    {
                        "feature_idx": i,
                        "gain": gain[i],
                        "frequency": freq[i],
                        "coverage": cov[i],
                    }
                    for i in top_indices
                ],
            }
            per_window.append(window_info)

            for i in range(n_features):
                agg_gain[i] = agg_gain.get(i, 0.0) + gain[i]
                agg_freq[i] = agg_freq.get(i, 0.0) + freq[i]
                agg_cov[i] = agg_cov.get(i, 0.0) + cov[i]
                count[i] = count.get(i, 0) + 1

        aggregated_list: List[Dict[str, Any]] = []
        for i in sorted(count.keys()):
            aggregated_list.append({
                "feature_idx": i,
                "gain": agg_gain[i] / count[i],
                "frequency": agg_freq[i] / count[i],
                "coverage": agg_cov[i] / count[i],
            })
        aggregated_list.sort(key=lambda x: x["gain"], reverse=True)

        global_indices = [f["feature_idx"] for f in aggregated_list[:top_k]]

        return {
            "per_window": per_window,
            "aggregated": aggregated_list,
            "top_features": global_indices,
        }

    def explain_shap(
        self,
        results: List[TrainingResult],
        X_val: np.ndarray,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Compute TreeSHAP values for the last window's GBDT model.

        Parameters
        ----------
        results : list of TrainingResult from train_walk_forward()
        X_val : (N, D) feature matrix for the validation set.
        top_k : number of top features to highlight.

        Returns
        -------
        dict with:
            - ``"shap_values"`` — (N, D) array of per-feature contributions.
            - ``"mean_abs_shap"`` — (D,) array of mean |SHAP| per feature.
            - ``"top_features"`` — list of indices sorted by mean |SHAP|.
            - ``"error"`` — error message if GBDT model is unavailable.
        """
        if self._last_gbdt_model is None:
            return {"error": "No GBDT model available. Train with ensemble mode first."}

        X_val = np.asarray(X_val, dtype=np.float64)
        num_features = X_val.shape[1]

        try:
            from ..utils.shap_explainer import TreeSHAPExplainer

            lr = float(self.gbdt_config.get("learning_rate", 0.1))
            explainer = TreeSHAPExplainer(
                self._last_gbdt_model,
                num_features=num_features,
                learning_rate=lr,
            )
        except Exception as exc:
            return {"error": f"Failed to initialise TreeSHAPExplainer: {exc}"}

        try:
            shap_values = explainer.explain(X_val)
            mean_abs_shap = explainer.feature_importance_shap(X_val)
        except Exception as exc:
            return {"error": f"SHAP computation failed: {exc}"}

        top_indices = np.argsort(mean_abs_shap)[::-1][:top_k].tolist()

        return {
            "shap_values": shap_values,
            "mean_abs_shap": mean_abs_shap,
            "top_features": top_indices,
        }

    # ── Data preparation ───────────────────────────────────────

    @staticmethod
    def _prepare_gbdt_data(
        data_seq: np.ndarray,
        target_seq: np.ndarray,
        ctm_hidden: Optional[np.ndarray] = None,
        include_ctm: bool = True,
        gbdt_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert temporal sequences to GBDT tabular feature matrix.

        Supports both single-asset (3D) and multi-asset (4D) inputs.
        Multi-asset data with shape (W, N, T, D) is automatically
        flattened to (W * N, T, D) with corresponding target flattening.

        Parameters
        ----------
        data_seq : np.ndarray
            (N, seq_len, D) for single-asset, or (W, N, T, D) for multi-asset.
        target_seq : np.ndarray
            (N, T, output_dim) for single-asset, or (W, T, N) for multi-asset.
        ctm_hidden : np.ndarray or None
            CTM encoder states, shape matching data_seq layout.
        include_ctm : bool
            Append CTM hidden features when True and ``ctm_hidden`` is provided.
        gbdt_config : dict or None
            GBDT configuration dict.

        Returns
        -------
        X_gbdt : (N, n_features) flat feature matrix.
        y_gbdt : (N,) last-timestep targets.
        """
        # ── Handle multi-asset 4D → 3D flattening ──
        if data_seq.ndim == 4:
            W, N, T, D = data_seq.shape
            data_seq = data_seq.reshape(W * N, T, D)
            if ctm_hidden is not None:
                ctm_hidden = ctm_hidden.reshape(W * N, T, -1)
            y_gbdt = target_seq[:, -1, :].ravel()  # (W, T, N) → (W*N,)
        else:
            y_gbdt = target_seq[:, -1, 0]  # (N, T, 1) → (N,)

        feature_cfg = GBDTFeatureConfig()
        if gbdt_config is not None:
            if "ctm_hidden_method" in gbdt_config:
                feature_cfg.ctm_hidden_method = gbdt_config["ctm_hidden_method"]
            if "include_ctm_features" in gbdt_config:
                include_ctm = bool(gbdt_config["include_ctm_features"])
            if "normalize" in gbdt_config:
                feature_cfg.normalize = bool(gbdt_config["normalize"])

        X_gbdt = build_gbdt_feature_matrix(
            data_seq,
            ctm_hidden=ctm_hidden,
            include_ctm_features=include_ctm and ctm_hidden is not None,
            config=feature_cfg,
        )
        return X_gbdt, y_gbdt

    # ── Shared Stages 1+2 per window ────────────────────────────

    def _run_window_stages_1_2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_data: torch.Tensor,
        train_targ: torch.Tensor,
        val_data: torch.Tensor,
        val_targ: torch.Tensor,
        n_epochs: int,
        lr: float,
        weight_decay: float,
        grad_clip: float,
        patience: int,
        warmup_steps: int,
        ramp_steps: int,
        lr_warmup_epochs: int,
        class_targets_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        log_gradients: bool = False,
    ) -> Dict[str, Any]:
        """Run Stages 1 (CTM train) + 2 (GBDT train) for one window.

        Returns a dict with all intermediate state needed by callers
        (predictions, fusion, recording, or P3 Stage 3).
        """
        # ── Stage 1: Train CTM ───────────────────────────────────
        ctm_model: nn.Module
        ctm_model, window_metrics, best_sharpe, _, epochs_run = (
            self._train_ctm_window(
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                lr=lr,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                patience=patience,
                warmup_steps=warmup_steps,
                ramp_steps=ramp_steps,
                lr_warmup_epochs=lr_warmup_epochs,
                class_targets_fn=class_targets_fn,
                log_gradients=log_gradients,
            )
        )

        # ── Stage 2: Extract CTM hidden states ───────────────────
        with torch.no_grad():
            hidden_train = ctm_model.extract_features(  # type: ignore[union-attr]
                train_data.to(self.device)
            )
            hidden_val = ctm_model.extract_features(  # type: ignore[union-attr]
                val_data.to(self.device)
            )

        train_data_np = train_data.cpu().numpy()
        train_targ_np = train_targ.cpu().numpy()
        val_data_np = val_data.cpu().numpy()
        val_targ_np = val_targ.cpu().numpy()
        hidden_train_np = hidden_train.cpu().numpy()
        hidden_val_np = hidden_val.cpu().numpy()

        include_ctm = self.gbdt_config.get("include_ctm_features", True)

        # Skip GBDT feature building when GBDT training is skipped with a
        # pre-loaded model (features are still needed for GBDT inference
        # in the fusion step, but not for training)
        gbdt_skipped = self._skip_gbdt and self._loaded_gbdt_model is not None

        X_gbdt_train, y_gbdt_train = self._prepare_gbdt_data(
            train_data_np, train_targ_np,
            ctm_hidden=hidden_train_np,
            include_ctm=include_ctm,
            gbdt_config=self.gbdt_config,
        )
        X_gbdt_val, y_gbdt_val = self._prepare_gbdt_data(
            val_data_np, val_targ_np,
            ctm_hidden=hidden_val_np,
            include_ctm=include_ctm,
            gbdt_config=self.gbdt_config,
        )

        # ── Stage 2: Train GBDT ──────────────────────────────────
        gbdt_model, gbdt_metrics = self._train_gbdt(
            X_gbdt_train, y_gbdt_train,
            X_gbdt_val, y_gbdt_val,
        )
        self._last_gbdt_model = gbdt_model

        # ── Stage 2b: Finetune TimeDecayGate (if enabled) ─────────
        # P2 Fix 5: Only run gate finetuning when GBDT passes quality check.
        # If GBDT predictions collapsed (near-zero variance), finetuning
        # the gate against degenerate targets degrades the CTM model.
        _gbdt_quality_ok = True
        if (self.gate_finetune_epochs > 0
                and gbdt_model is not None):
            try:
                _gbdt_val_preds = gbdt_model.predict(
                    np.asarray(X_gbdt_val, dtype=np.float32)
                )
                _gbdt_pred_std = float(np.nanstd(_gbdt_val_preds))
                if _gbdt_pred_std < 1e-6:
                    _gbdt_quality_ok = False
                    warnings.warn(
                        f"GBDT predictions collapsed (std={_gbdt_pred_std:.2e}) — "
                        f"skipping TimeDecayGate finetuning to avoid degrading CTM"
                    )
            except Exception:
                _gbdt_quality_ok = True  # err on the side of caution

        if (self.gate_finetune_epochs > 0
                and gbdt_model is not None
                and _gbdt_quality_ok
                and hasattr(ctm_model, 'time_gate')
                and ctm_model.time_gate is not None):
            n_assets = getattr(ctm_model, 'n_assets', 1)
            seq_len_val = train_data.shape[2] if train_data.ndim == 4 else train_data.shape[1]
            ctm_model = self._finetune_time_gate(
                ctm_model=ctm_model,
                gbdt_model=gbdt_model,
                train_data=train_data,
                train_targ=train_targ,
                val_data=val_data,
                val_targ=val_targ,
                seq_len=seq_len_val,
                n_assets=n_assets,
                class_targets_fn=class_targets_fn,
            )

        # ── Stage 2: Extract feature importance from GBDT ───────
        gbdt_importance = None
        if gbdt_model is not None:
            try:
                num_features = X_gbdt_train.shape[1]
                imp_full = gbdt_model.get_feature_importance_full(num_features)
                gbdt_importance = {
                    "frequency": list(imp_full["frequency"]),
                    "gain": list(imp_full["gain"]),
                    "coverage": list(imp_full["coverage"]),
                }
            except Exception as exc:
                warnings.warn(f"Feature importance extraction failed: {exc}")

        return {
            "ctm_model": ctm_model,
            "window_metrics": window_metrics,
            "best_sharpe": best_sharpe,
            "epochs_run": epochs_run,
            "hidden_train": hidden_train,
            "hidden_val": hidden_val,
            "train_data_np": train_data_np,
            "train_targ_np": train_targ_np,
            "val_data_np": val_data_np,
            "val_targ_np": val_targ_np,
            "hidden_train_np": hidden_train_np,
            "hidden_val_np": hidden_val_np,
            "gbdt_model": gbdt_model,
            "gbdt_metrics": gbdt_metrics,
            "gbdt_importance": gbdt_importance,
            "X_gbdt_val": X_gbdt_val,
            "y_gbdt_val": y_gbdt_val,
        }

    # ── CTM window training ────────────────────────────────────

    def _train_ctm_window(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        lr: float,
        weight_decay: float,
        grad_clip: float,
        patience: int,
        warmup_steps: int,
        ramp_steps: int,
        lr_warmup_epochs: int,
        class_targets_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        log_gradients: bool = False,
    ) -> Tuple[nn.Module, List[Dict[str, float]], float, float, int]:
        """Train CTM for one window, optionally warm-starting from pretrained weights.

        When ``skip_ctm_training=True`` and a pretrained state dict is
        available, this method loads the pre-trained model once and
        returns it without any training (fine-tuning is skipped).
        The model is cached in ``self._loaded_ctm_model`` for reuse
        across windows.

        When ``skip_ctm_training=False`` but a pretrained state dict is
        provided, the model is initialised with those weights (warm-start)
        and fine-tuned normally through the walk-forward window.

        Returns
        -------
        (model, window_metrics, best_sharpe, best_ic, epochs_run)
        """
        # ── Skip mode: load pretrained CTM once, reuse across windows ──
        if self._skip_ctm and self._pretrained_ctm_sd is not None:
            if self._loaded_ctm_model is None:
                model = self.model_class(**self.model_params).to(self.device)
                missing, unexpected = model.load_state_dict(
                    self._pretrained_ctm_sd, strict=False
                )
                if missing:
                    warnings.warn(
                        f"CTM pretrained load: missing keys={missing[:5]}... "
                        f"({len(missing)} total)"
                    )
                if unexpected:
                    warnings.warn(
                        f"CTM pretrained load: unexpected keys={unexpected[:5]}... "
                        f"({len(unexpected)} total)"
                    )
                model.eval()
                self._loaded_ctm_model = model
                logging.info(
                    "CTM pretrained: loaded model, skipped training"
                )
            # Return cached model with placeholder metrics (best_ic=0.0)
            return self._loaded_ctm_model, [], 0.0, 0.0, 0

        # ── Warm-start or normal mode ──
        base_trainer = WalkForwardTrainerAdvanced(
            model_class=self.model_class,
            model_params=self.model_params,
            loss_config=self.loss_config,
            device=self.device,
        )

        init_sd = self._pretrained_ctm_sd if not self._skip_ctm else None
        if init_sd is not None:
            logging.info("CTM warm-start: initialised from pretrained checkpoint")

        result = base_trainer._train_single_window(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
            grad_clip=grad_clip, patience=patience,
            warmup_steps=warmup_steps, ramp_steps=ramp_steps,
            lr_warmup_epochs=lr_warmup_epochs,
            class_targets_fn=class_targets_fn,
            log_gradients=log_gradients,
            init_state_dict=init_sd,
        )
        # Unpack: model, window_metrics, best_sharpe, best_ic, epochs_run
        model, window_metrics, best_sharpe, best_ic, epochs_run = result
        return model, window_metrics, best_sharpe, best_ic, epochs_run

    # ── GBDT training dispatch ─────────────────────────────────

    def _train_gbdt(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train GBDT, dispatching to correct loss-specific method.

        When ``skip_gbdt_training=True`` and a pretrained GBDT JSON path
        is available, the pre-trained model is loaded once and returned
        without training (cached for reuse across windows).

        Returns (trained_model_or_None, metrics_dict).
        """
        # ── Skip mode: load pretrained GBDT once, reuse across windows ──
        if self._skip_gbdt and self._pretrained_gbdt_path is not None:
            if self._loaded_gbdt_model is None:
                try:
                    from gbdt_python import GBDT, GBDTConfig
                    config = GBDTConfig()
                    _apply_config(config, self.gbdt_config)
                    model = GBDT(config)
                    with open(self._pretrained_gbdt_path) as f:
                        model.from_json(f.read())
                    self._loaded_gbdt_model = model
                    logging.info(
                        "GBDT pretrained: loaded from %s, skipped training",
                        self._pretrained_gbdt_path,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Failed to load pretrained GBDT from "
                        f"{self._pretrained_gbdt_path}: {exc}. "
                        "Falling back to training GBDT from scratch."
                    )
                    # Fall through to normal training
                    return self._train_gbdt_fallback(X_train, y_train, X_val, y_val)
            return self._loaded_gbdt_model, {"gbdt_available": True, "pretrained": True}

        # ── Normal training ──
        return self._train_gbdt_fallback(X_train, y_train, X_val, y_val)

    def _train_gbdt_fallback(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train GBDT from scratch, dispatching to loss-specific method."""
        if not _GBDT_CPP_AVAILABLE:
            warnings.warn(
                "GBDT C++ module (gbdt_python) not available — "
                "skipping GBDT training.  Install Hoffnung or set "
                "``gbdt_loss`` to a native type."
            )
            return None, {"gbdt_available": False}

        if self.gbdt_loss == "rankic":
            return self._gbdt_rankic_fit(X_train, y_train, X_val, y_val)

        if self.gbdt_loss == "composite" and self.loss_fn is not None:
            return self._gbdt_composite_fit(X_train, y_train, X_val, y_val)

        return self._gbdt_native_fit(X_train, y_train, X_val, y_val)

    def _gbdt_composite_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train GBDT using the CTM composite loss bridge.

        Uses ``GBDTTrainer(config, loss_fn=loss_fn)`` when the Python
        wrapper is available; falls back to a manual boosting loop.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        if _GBDT_PYTHON_TRAINER_AVAILABLE:
            try:
                assert GBDTConfig is not None and _GBDTTrainerCls is not None
                config = GBDTConfig()
                _apply_config(config, self.gbdt_config)
                trainer = _GBDTTrainerCls(config, loss_fn=self.loss_fn)
                trainer.fit(X_train, y_train, X_val, y_val)

                metrics: Dict[str, Any] = {
                    "gbdt_available": True,
                    "loss_type": "composite",
                    "num_trees": trainer.num_trees,
                    "train_losses": list(trainer.train_losses),
                    "val_losses": list(trainer.val_losses),
                    "trainer": "gbdt_python",
                }
                return trainer, metrics
            except Exception as exc:
                warnings.warn(
                    f"GBDTTrainer with composite loss raised: {exc}. "
                    "Falling back to manual boosting loop."
                )

        assert GBDTConfig is not None and GBDT is not None
        config = GBDTConfig()
        _apply_config(config, self.gbdt_config)
        config.loss_type = "mse"
        model = GBDT(config)
        lr = float(config.learning_rate)

        init_pred = float(np.mean(y_train))
        y_pred = np.full(len(X_train), init_pred, dtype=np.float32)
        y_pred_val = np.full(len(X_val), init_pred, dtype=np.float32)

        trees: List = []
        step_sizes: List[float] = []
        train_losses: List[float] = []
        val_losses: List[float] = []

        assert self.loss_fn is not None
        for i in range(config.num_trees):
            y_true_t = torch.from_numpy(y_train)
            y_pred_t = torch.from_numpy(y_pred)

            loss, grads, hess = self.loss_fn(y_true_t, y_pred_t)
            train_losses.append(float(loss.item()))

            grads_np = grads.numpy()
            hess_np = hess.numpy()

            tree = model.fit_one_tree(X_train, grads_np, hess_np)
            trees.append(tree)

            tree_pred = model.predict_tree(tree, X_train)
            y_pred = y_pred + lr * tree_pred

            tree_pred_val = model.predict_tree(tree, X_val)
            y_pred_val = y_pred_val + lr * tree_pred_val
            val_l, _, _ = self.loss_fn(
                torch.from_numpy(y_val),
                torch.from_numpy(y_pred_val),
            )
            val_losses.append(float(val_l.item()))
            step_sizes.append(lr)

        model.set_state(init_pred, trees, step_sizes)

        metrics = {
            "gbdt_available": True,
            "loss_type": "composite",
            "num_trees": len(trees),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "trainer": "manual_loop",
        }
        return model, metrics

    def _gbdt_native_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train GBDT using native C++ ``fit()`` with standard loss.

        The loss type is set from ``self.gbdt_loss`` (``"mse"``,
        ``"mae"``, or ``"huber"``).
        """
        assert GBDTConfig is not None, "GBDT C++ module not loaded"
        config = GBDTConfig()
        _apply_config(config, self.gbdt_config)
        config.loss_type = self.gbdt_loss

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        assert GBDT is not None, "GBDT C++ module not loaded"
        model = GBDT(config)
        model.fit(X_train, y_train, X_val, y_val)

        # ── Quality gate: check for prediction collapse ───
        pred_std = float(np.std(model.predict(X_val)))
        if pred_std < 1e-6:
            warnings.warn(
                f"GBDT predictions collapsed (validation pred std={pred_std:.2e}). "
                f"Model predicts near-constant value. "
                f"Consider increasing num_trees, reducing learning_rate, "
                f"or checking feature/target quality."
            )

        metrics: Dict[str, Any] = {
            "gbdt_available": True,
            "loss_type": self.gbdt_loss,
            "num_trees": int(model.num_trees()),
            "train_losses": _to_list(model.train_losses),
            "val_losses": _to_list(model.val_losses),
            "pred_std_quality": pred_std,
        }
        return model, metrics

    def _gbdt_rankic_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train GBDT with differentiable RankIC loss.

        Attempts the Python ``GBDTTrainer`` wrapper first (from the
        ``gbdt`` package).  Falls back to a manual boosting loop that
        uses ``GBDT.fit_one_tree()`` with the differentiable rank
        approximation from ``src.model.ensemble``.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        # ── Attempt Python GBDTTrainer wrapper (preferred) ─────
        if _GBDT_PYTHON_TRAINER_AVAILABLE and _gbdt_rankic_loss is not None:
            try:
                assert GBDTConfig is not None
                config = GBDTConfig()
                _apply_config(config, self.gbdt_config)

                assert _GBDTTrainerCls is not None and _gbdt_rankic_loss is not None
                trainer = _GBDTTrainerCls(config, loss_fn=_gbdt_rankic_loss)
                trainer.fit(X_train, y_train, X_val, y_val)

                metrics: Dict[str, Any] = {
                    "gbdt_available": True,
                    "loss_type": "rankic",
                    "num_trees": trainer.num_trees,
                    "train_losses": list(trainer.train_losses),
                    "val_losses": list(trainer.val_losses),
                    "trainer": "gbdt_python",
                }
                return trainer, metrics
            except Exception as exc:
                warnings.warn(
                    f"GBDTTrainer with rankic_loss raised: {exc}. "
                    "Falling back to manual boosting loop."
                )

        # ── Manual boosting loop ───────────────────────────────
        if _gbdt_rankic_loss is None and _GBDT_PYTHON_TRAINER_AVAILABLE:
            warnings.warn(
                "rankic_loss not available in gbdt.losses. "
                "Using differentiable rank from src.model.ensemble."
            )

        assert GBDTConfig is not None and GBDT is not None
        config = GBDTConfig()
        _apply_config(config, self.gbdt_config)
        config.loss_type = "mse"

        model = GBDT(config)
        lr = float(config.learning_rate)

        init_pred = float(np.mean(y_train))
        y_pred = np.full(len(X_train), init_pred, dtype=np.float32)
        y_pred_val = np.full(len(X_val), init_pred, dtype=np.float32)

        trees: List = []
        step_sizes: List[float] = []
        train_losses: List[float] = []
        val_losses: List[float] = []

        for i in range(config.num_trees):
            y_true_t = torch.from_numpy(y_train)
            y_pred_t = torch.from_numpy(y_pred)

            loss, grads, hess = rankic_loss_gbdt_style(y_true_t, y_pred_t)
            train_losses.append(float(loss.item()))

            grads_np = grads.numpy()
            hess_np = hess.numpy()

            tree = model.fit_one_tree(X_train, grads_np, hess_np)
            trees.append(tree)

            tree_pred = model.predict_tree(tree, X_train)
            y_pred = y_pred + lr * tree_pred

            tree_pred_val = model.predict_tree(tree, X_val)
            y_pred_val = y_pred_val + lr * tree_pred_val
            val_losses.append(
                float(
                    rankic_loss_gbdt_style(
                        torch.from_numpy(y_val),
                        torch.from_numpy(y_pred_val),
                    )[0].item()
                )
            )
            step_sizes.append(lr)

        model.set_state(init_pred, trees, step_sizes)

        metrics = {
            "gbdt_available": True,
            "loss_type": "rankic",
            "num_trees": len(trees),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "trainer": "manual_loop",
        }
        return model, metrics


    # ── TimeDecayGate finetuning ──────────────────────────────

    def _finetune_time_gate(
        self,
        ctm_model: nn.Module,
        gbdt_model: Any,
        train_data: torch.Tensor,
        train_targ: torch.Tensor,
        val_data: torch.Tensor,
        val_targ: torch.Tensor,
        seq_len: int,
        n_assets: int,
        class_targets_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> nn.Module:
        """Fine-tune TimeDecayGate (α, β, γ) after CTM + GBDT training.

        Strategy:
        1. Freeze CTM backbone (optionally all non-gate params).
        2. Pre-compute GBDT predictions for train/val sets (detached).
        3. Forward: CTM predicts → gate blends with GBDT → ensemble output.
        4. Backprop: gradients flow through gate params (α,β,γ) into the
           ensemble loss.  GBDT weights are untouched.
        5. Early-stop on validation loss, restore best gate state.
        6. Unfreeze all params for subsequent walk-forward windows.

        Returns the CTM model with fine-tuned gate parameters.
        """
        from copy import deepcopy

        output_dim = self.model_params.get("output_dim", 1)
        device = self.device
        # ── 1. Pre-compute GBDT predictions (frozen backbone ⇒ static features) ──
        with torch.no_grad():
            hidden_train = ctm_model.extract_features(
                train_data.to(device)
            ).cpu().numpy()
            hidden_val = ctm_model.extract_features(
                val_data.to(device)
            ).cpu().numpy()

        train_data_np = train_data.cpu().numpy()
        train_targ_np = train_targ.cpu().numpy()
        val_data_np = val_data.cpu().numpy()
        val_targ_np = val_targ.cpu().numpy()

        include_ctm = self.gbdt_config.get("include_ctm_features", True)
        X_gbdt_train, _ = self._prepare_gbdt_data(
            train_data_np, train_targ_np,
            ctm_hidden=hidden_train,
            include_ctm=include_ctm,
            gbdt_config=self.gbdt_config,
        )
        X_gbdt_val, _ = self._prepare_gbdt_data(
            val_data_np, val_targ_np,
            ctm_hidden=hidden_val,
            include_ctm=include_ctm,
            gbdt_config=self.gbdt_config,
        )

        X_train_f32 = np.asarray(X_gbdt_train, dtype=np.float32)
        X_val_f32 = np.asarray(X_gbdt_val, dtype=np.float32)

        gbdt_pred_train = gbdt_model.predict(X_train_f32)  # (W*N,)
        gbdt_pred_val = gbdt_model.predict(X_val_f32)      # (W*N,)

        if np.isnan(gbdt_pred_train).any():
            warnings.warn(
                "NaN in GBDT training predictions during gate finetuning. "
                "Replacing with 0."
            )
            gbdt_pred_train = np.nan_to_num(gbdt_pred_train, nan=0.0)
        if np.isnan(gbdt_pred_val).any():
            warnings.warn(
                "NaN in GBDT validation predictions during gate finetuning. "
                "Replacing with 0."
            )
            gbdt_pred_val = np.nan_to_num(gbdt_pred_val, nan=0.0)

        W_train = train_data.shape[0]
        W_val = val_data.shape[0]
        T = seq_len

        # Reshape (W*N,) → (W, N) → (W, N, T) so forward() receives
        # per-(sequence, asset) scalar repeated across timesteps.
        gbdt_train_3d = torch.from_numpy(
            gbdt_pred_train.reshape(W_train, n_assets)
        ).float().unsqueeze(-1).expand(-1, -1, T)   # (W, N, T)

        gbdt_val_3d = torch.from_numpy(
            gbdt_pred_val.reshape(W_val, n_assets)
        ).float().unsqueeze(-1).expand(-1, -1, T)

        # ── 2. Freeze / unfreeze params ──
        _param_states: Dict[str, bool] = {}
        for name, param in ctm_model.named_parameters():
            _param_states[name] = param.requires_grad
            if 'time_gate' in name:
                param.requires_grad = True   # always trainable
            else:
                param.requires_grad = not self.freeze_ctm_backbone_for_gate

        # ── 3. Build optimiser (only for params that require grad) ──
        gate_params = [p for p in ctm_model.parameters() if p.requires_grad]
        if not gate_params:
            # Shouldn't happen (time_gate always trainable), but guard.
            warnings.warn("No trainable params for gate finetuning — skipping.")
            # Restore original requires_grad
            for name, param in ctm_model.named_parameters():
                param.requires_grad = _param_states.get(name, True)
            return ctm_model
        optimizer = torch.optim.Adam(gate_params, lr=self.gate_finetune_lr)

        # ── 4. Create dataloaders with GBDT predictions ──
        gate_train_ds = TensorDataset(train_data, gbdt_train_3d, train_targ)
        gate_val_ds = TensorDataset(val_data, gbdt_val_3d, val_targ)
        gate_train_loader = DataLoader(
            gate_train_ds, batch_size=self.gate_finetune_batch_size, shuffle=False,
        )
        gate_val_loader = DataLoader(
            gate_val_ds, batch_size=self.gate_finetune_batch_size, shuffle=False,
        )

        # ── 5. Loss config (deepcopy to avoid mutating shared config) ──
        loss_cfg = deepcopy(self.loss_config)

        # ── 6. Training loop ──
        ctm_model.train()
        best_val_loss = float("inf")
        best_gate_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0
        gate_metrics: List[Dict[str, float]] = []

        for epoch in range(self.gate_finetune_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_gbdt, batch_y in gate_train_loader:
                batch_x = batch_x.to(device)
                batch_gbdt = batch_gbdt.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                output = ctm_model(batch_x, gbdt_preds=batch_gbdt)

                cls_targets = (
                    class_targets_fn(batch_y) if class_targets_fn
                    else batch_y.new_zeros(0, dtype=torch.long)
                )
                if cls_targets.dim() > 1 and cls_targets.shape[-1] == 1:
                    cls_targets = cls_targets.squeeze(-1)

                loss = composite_loss(
                    predictions=output,
                    regression_target=batch_y,
                    class_targets=cls_targets,
                    model_parameters=[
                        p for p in ctm_model.parameters()
                        if p.requires_grad and p.ndim > 1
                    ],
                    config=loss_cfg,
                    num_regression=n_assets * output_dim,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ctm_model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # ── Validation ──
            ctm_model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_x, batch_gbdt, batch_y in gate_val_loader:
                    batch_x = batch_x.to(device)
                    batch_gbdt = batch_gbdt.to(device)
                    batch_y = batch_y.to(device)
                    output = ctm_model(batch_x, gbdt_preds=batch_gbdt)
                    cls_targets = (
                        class_targets_fn(batch_y) if class_targets_fn
                        else batch_y.new_zeros(0, dtype=torch.long)
                    )
                    if cls_targets.dim() > 1 and cls_targets.shape[-1] == 1:
                        cls_targets = cls_targets.squeeze(-1)
                    v_loss = composite_loss(
                        predictions=output,
                        regression_target=batch_y,
                        class_targets=cls_targets,
                        model_parameters=[
                            p for p in ctm_model.parameters()
                            if p.requires_grad and p.ndim > 1
                        ],
                        config=loss_cfg,
                        num_regression=n_assets * output_dim,
                    )
                    val_loss += v_loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            avg_train_loss = epoch_loss / max(n_batches, 1)
            gate_metrics.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            })

            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                best_gate_state = {
                    k: v.cpu().clone() for k, v in ctm_model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.gate_finetune_patience:
                    logging.info(
                        "  Gate finetune early stop at epoch %d/%d (val_loss=%.6f)",
                        epoch + 1, self.gate_finetune_epochs, avg_val_loss,
                    )
                    break

            ctm_model.train()

        # ── 7. Restore best gate state ──
        if best_gate_state is not None:
            ctm_model.load_state_dict(best_gate_state)

        # ── 8. Restore original requires_grad for next window ──
        for name, param in ctm_model.named_parameters():
            param.requires_grad = _param_states.get(name, True)

        logging.info(
            "TimeGate finetuned: %d epochs, best_val_loss=%.6f, "
            "gate α=%.4f β=%.4f γ=%.4f",
            len(gate_metrics), best_val_loss,
            ctm_model.time_gate.alpha.item(),
            ctm_model.time_gate.beta.item(),
            ctm_model.time_gate.gamma.item(),
        )

        return ctm_model


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════


def _apply_config(config: Any, cfg_dict: Dict[str, Any]) -> None:
    """Copy recognised keys from ``cfg_dict`` onto a ``GBDTConfig`` object.

    Skips keys not present as config attributes and CTM-only keys.
    """
    skip_keys = {"include_ctm_features", "ctm_hidden_method"}
    for key, val in cfg_dict.items():
        if key in skip_keys:
            continue
        if hasattr(config, key):
            setattr(config, key, val)
        else:
            warnings.warn(f"GBDTConfig has no attribute '{key}' — ignoring.")


def _to_list(x: Any) -> List[float]:
    """Safely convert a C++ sequence to a Python list."""
    return list(x) if x is not None else []
