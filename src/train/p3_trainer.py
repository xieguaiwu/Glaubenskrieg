"""P3 three-stage fusion trainer: CTM warmup → GBDT fit → Modulator fine-tune.

Extends ``EnsembleWalkForwardTrainer`` by adding a Stage 3 after GBDT training:
backbone (Mamba) frozen, ``FusedMultiHeadCrossAttention`` (including
``GBDTModulator``) + output heads fine-tuned with GBDT predictions injected
into the cross-asset attention as a pair-wise bias.
"""

from __future__ import annotations

import copy
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ._walk_forward_utils import TrainingResult, walk_forward_windows
from .ensemble_trainer import EnsembleWalkForwardTrainer, _compute_sharpe, _safe_ic
from ..model.ensemble import EnsembleConfig, evaluate_ensemble
from ..model.losses import LossConfig, composite_loss


class P3EnsembleTrainer(EnsembleWalkForwardTrainer):
    """Three-stage walk-forward trainer with CTM → GBDT → Modulator fine-tune.

    Stages per walk-forward window:
        1. CTM warmup (inherited from ``EnsembleWalkForwardTrainer``).
        2. GBDT fit (inherited).
        3. Modulator fine-tune (new):
           - Build ``MultiAssetCTM(use_fused_attention=True)``.
           - Transfer backbone weights from Stage 1.
           - Freeze backbone; train ``FusedMultiHeadCrossAttention`` + heads.
           - Inject GBDT predictions via ``model(x, gbdt_preds=…)``.
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
        *,
        modulator_epochs: int = 10,
        modulator_lr: float = 1e-4,
        modulator_patience: int = 3,
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
        super().__init__(
            model_class=model_class,
            model_params=model_params,
            loss_config=loss_config,
            gbdt_config=gbdt_config,
            gbdt_loss=gbdt_loss,
            device=device,
            loss_fn=loss_fn,
            log_gradients=log_gradients,
            pretrained_ctm_state_dict=pretrained_ctm_state_dict,
            pretrained_gbdt_json=pretrained_gbdt_json,
            skip_ctm_training=skip_ctm_training,
            skip_gbdt_training=skip_gbdt_training,
            gate_finetune_epochs=gate_finetune_epochs,
            gate_finetune_lr=gate_finetune_lr,
            freeze_ctm_backbone_for_gate=freeze_ctm_backbone_for_gate,
            gate_finetune_batch_size=gate_finetune_batch_size,
            gate_finetune_patience=gate_finetune_patience,
        )
        self.modulator_epochs = modulator_epochs
        self.modulator_lr = modulator_lr
        self.modulator_patience = modulator_patience

    def get_stage3_model(self):
        """Return the Stage 3 fine-tuned MultiAssetCTM for serialization."""
        return self._stage3_model if hasattr(self, '_stage3_model') else None

    # ── Override: insert Stage 3 after GBDT training ──────────────

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
            hidden_train_np = state["hidden_train_np"]
            hidden_val_np = state["hidden_val_np"]
            val_data_np = state["val_data_np"]

            # ── Stage 3: Modulator fine-tune ──────────────────────
            n_val_samples = val_data_np.shape[0]
            n_assets = self.model_params.get("n_assets", 1)
            seq_len = (
                val_data.shape[2] if val_data.ndim == 4
                else val_data.shape[1]
            )

            if n_assets <= 1:
                raise ValueError(
                    "P3 requires multi-asset mode (n_assets > 1). "
                    "Use --multi-asset --n-assets N."
                )

            stage3_model = self._build_stage3_model(ctm_model)

            include_ctm = self.gbdt_config.get("include_ctm_features", True)

            train_per_asset = self._prepare_per_asset_gbdt_preds(
                gbdt_model, state["train_data_np"], hidden_train_np,
                include_ctm=include_ctm, n_assets=n_assets,
            )
            val_per_asset = self._prepare_per_asset_gbdt_preds(
                gbdt_model, val_data_np, hidden_val_np,
                include_ctm=include_ctm, n_assets=n_assets,
            )

            stage3_model, stage3_metrics = self._finetune_stage3(
                model=stage3_model,
                train_data=train_data, train_targ=train_targ,
                val_data=val_data, val_targ=val_targ,
                gbdt_preds_train=train_per_asset,
                gbdt_preds_val=val_per_asset,
                n_assets=n_assets,
                batch_size=batch_size,
                seq_len=seq_len,
                class_targets_fn=class_targets_fn,
            )
            self._stage3_model = stage3_model  # store fine-tuned model

            # ── Stage 3 evaluation ───────────────────────────────
            with torch.no_grad():
                gbdt_preds_batched = (
                    torch.from_numpy(val_per_asset)
                    .float()
                    .unsqueeze(-1)
                    .expand(n_val_samples, n_assets, seq_len)
                    .to(self.device)
                )
                stage3_output = stage3_model(
                    val_data.to(self.device),
                    gbdt_preds=gbdt_preds_batched,
                )
                stage3_pred_val = stage3_output[:, -1, 0].cpu().numpy()

            stage3_sharpe = float(_compute_sharpe(stage3_pred_val))
            stage3_ic = _safe_ic(y_gbdt_val, stage3_pred_val)

            # ── Stage 3 + GBDT ensemble ──────────────────────────
            if gbdt_model is not None:
                gbdt_pred_val = gbdt_model.predict(X_gbdt_val)
                if np.isnan(gbdt_pred_val).any():
                    warnings.warn(
                        "NaN in GBDT predictions during P3 ensemble. "
                        "Replacing with 0."
                    )
                    gbdt_pred_val = np.nan_to_num(gbdt_pred_val, nan=0.0)
            else:
                gbdt_pred_val = np.zeros_like(stage3_pred_val)

            ensemble_cfg = EnsembleConfig(
                use_ic_weighting=True,
                ic_lookback=min(252, max(5, n_val_samples // 2)),
                min_samples_for_ic=min(20, n_val_samples),
            )
            s3_ensemble_result = evaluate_ensemble(
                stage3_pred_val, gbdt_pred_val, y_gbdt_val, ensemble_cfg,
            )

            # ── Baseline fusion (Stage 1 CTM + GBDT) ─────────────
            with torch.no_grad():
                ctm_output = ctm_model(val_data.to(self.device))
                ctm_pred_val = ctm_output[:, -1, 0].cpu().numpy()
            ctm_sharpe = float(_compute_sharpe(ctm_pred_val))
            baseline_ensemble = evaluate_ensemble(
                ctm_pred_val, gbdt_pred_val, y_gbdt_val, ensemble_cfg,
            )

            # ── Record ───────────────────────────────────────────
            results.append(TrainingResult(
                window_start=pos,
                window_end=val_end,
                best_sharpe=state["best_sharpe"],
                epochs_run=state["epochs_run"],
                metrics=state["window_metrics"],
                ctm_sharpe=ctm_sharpe,
                gbdt_ic=_safe_ic(y_gbdt_val, gbdt_pred_val),
                ensemble_sharpe=float(_compute_sharpe(baseline_ensemble.fused)),
                ensemble_ic=float(baseline_ensemble.fused_ic),
                ctm_weight=0.5,
                gbdt_weight=0.5,
                ctm_metrics=list(state["window_metrics"]),
                gbdt_metrics=state["gbdt_metrics"],
                gbdt_importance=state["gbdt_importance"],
                p3_sharpe=stage3_sharpe,
                p3_ic=stage3_ic,
                p3_ensemble_sharpe=float(_compute_sharpe(s3_ensemble_result.fused)),
            ))

        return results

    # ── Stage 3 helpers ──────────────────────────────────────────

    def _build_stage3_model(self, stage1_model: nn.Module) -> nn.Module:
        """Build ``MultiAssetCTM(use_fused_attention=True)`` and transfer
        backbone weights from the Stage 1 model.

        Weight transfer strategy:
        - Copy  ``single_asset_model.*`` (backbone — base params match via
          inheritance, ``loop_norm`` / ``loop_dropout`` stay random).
        - Copy  ``asset_embed.*``, ``cond_proj.*``.
        - Copy  ``head_regression.*``, ``head_classification.*``.
        - **Skip** ``cross_attn.*`` — different architecture.
        """
        from ..model.multiasset_ctm import MultiAssetCTM

        params = dict(self.model_params)
        params.pop("output_dim", None)  # backbone output_dim=0; heads are MultiAssetCTM's own

        stage3_model = MultiAssetCTM(
            **params,
            use_fused_attention=True,
        ).to(self.device)

        stage1_sd = stage1_model.state_dict()
        stage3_sd = stage3_model.state_dict()

        transfer_keys = {
            k for k in stage1_sd
            if k in stage3_sd and not k.startswith("cross_attn.")
        }
        transferred = {k: stage1_sd[k] for k in transfer_keys}
        missing = stage3_model.load_state_dict(transferred, strict=False)
        if missing.unexpected_keys:
            warnings.warn(
                f"Unexpected keys during Stage 3 weight transfer: "
                f"{missing.unexpected_keys}"
            )
        return stage3_model

    def _prepare_per_asset_gbdt_preds(
        self,
        gbdt_model: Any,
        data_np: np.ndarray,
        hidden_np: np.ndarray | None,
        include_ctm: bool,
        n_assets: int,
    ) -> np.ndarray:
        from ..data.gbdt_features import build_gbdt_feature_matrix

        n_samples = data_np.shape[0]
        d_in = data_np.shape[-1] // n_assets if data_np.ndim >= 2 else 1

        per_asset_preds = np.zeros((n_samples, n_assets), dtype=np.float32)
        for a in range(n_assets):
            asset_slice = data_np[:, :, a * d_in:(a + 1) * d_in]
            if include_ctm and hidden_np is not None:
                model_dim = hidden_np.shape[-1]
                h_reshaped = hidden_np.reshape(n_samples, n_assets, -1, model_dim)
                asset_hidden = h_reshaped[:, a, :, :].reshape(n_samples, -1, model_dim)
            else:
                asset_hidden = None

            X_a = build_gbdt_feature_matrix(
                asset_slice, ctm_hidden=asset_hidden,
                include_ctm_features=include_ctm and asset_hidden is not None,
            )
            if gbdt_model is not None:
                preds = gbdt_model.predict(X_a)
                per_asset_preds[:, a] = np.nan_to_num(preds, nan=0.0)
        return per_asset_preds

    def _finetune_stage3(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_targ: torch.Tensor,
        val_data: torch.Tensor,
        val_targ: torch.Tensor,
        gbdt_preds_train: np.ndarray,
        gbdt_preds_val: np.ndarray,
        n_assets: int,
        batch_size: int,
        seq_len: int,
        class_targets_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> Tuple[nn.Module, List[Dict[str, Any]]]:
        device = self.device

        def _expand_gbdt(preds: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(preds).float().to(device)
            return t.unsqueeze(-1).expand(-1, -1, seq_len)

        train_gbdt_exp = _expand_gbdt(gbdt_preds_train)
        val_gbdt_exp = _expand_gbdt(gbdt_preds_val)

        train_loader = DataLoader(
            TensorDataset(train_data, train_targ, train_gbdt_exp),
            batch_size=batch_size, shuffle=False,
        )
        val_loader = DataLoader(
            TensorDataset(val_data, val_targ, val_gbdt_exp),
            batch_size=batch_size, shuffle=False,
        )

        model = model.to(device)
        model.train()

        for name, param in model.named_parameters():
            if name.startswith("single_asset_model."):
                if "loop_norm" in name or "loop_dropout" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            elif name.startswith("cross_attn."):
                param.requires_grad = True
            elif name.startswith("head_"):
                param.requires_grad = True
            elif name.startswith("asset_embed.") or name.startswith("cond_proj."):
                param.requires_grad = False

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.modulator_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.modulator_epochs,
        )

        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        metrics: List[Dict[str, Any]] = []

        output_dim = self.model_params.get("output_dim", 1)

        for epoch in range(self.modulator_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_x, batch_y, batch_gbdt in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_gbdt = batch_gbdt.to(device)
                optimizer.zero_grad()
                output = model(batch_x, gbdt_preds=batch_gbdt)
                cls_targets = (
                    class_targets_fn(batch_y) if class_targets_fn
                    else batch_y
                )
                loss = composite_loss(
                    predictions=output,
                    regression_target=batch_y,
                    class_targets=cls_targets,
                    model_parameters=[p for p in model.parameters() if p.requires_grad and p.ndim > 1],
                    config=self.loss_config,
                    num_regression=n_assets * output_dim,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y, batch_gbdt in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_gbdt = batch_gbdt.to(device)
                    output = model(batch_x, gbdt_preds=batch_gbdt)
                    cls_targets = (
                        class_targets_fn(batch_y) if class_targets_fn
                        else batch_y
                    )
                    v_loss = composite_loss(
                        predictions=output,
                        regression_target=batch_y,
                        class_targets=cls_targets,
                        model_parameters=[p for p in model.parameters() if p.requires_grad and p.ndim > 1],
                        config=self.loss_config,
                        num_regression=n_assets * output_dim,
                    )
                    val_loss += v_loss.item()

            avg_val_loss = val_loss / max(len(val_loader), 1)
            metrics.append({
                "epoch": epoch,
                "train_loss": epoch_loss / max(len(train_loader), 1),
                "val_loss": avg_val_loss,
            })

            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.modulator_patience:
                    print(
                        f"  Stage 3 early stopping at epoch {epoch} "
                        f"(val_loss={avg_val_loss:.6f})"
                    )
                    break

        model.load_state_dict(best_state)
        return model, metrics
