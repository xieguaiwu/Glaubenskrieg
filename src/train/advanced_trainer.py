"""Advanced trainer with composite loss, warmup schedules, and walk-forward CV."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..model.losses import LossConfig, LearnableWeights, composite_loss
from ._walk_forward_utils import TrainingResult, walk_forward_windows
from .trainer import validate
import logging

logger = logging.getLogger(__name__)


class LossWrapper(nn.Module):
    """Wraps composite_loss with fixed config and learnable weights.

    Provides the simple ``loss_fn(pred, target)`` interface expected by the
    training loop, internally handling all 6 arguments of composite_loss.
    """

    def __init__(
        self,
        config: LossConfig,
        model: nn.Module,
        class_targets_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        learnable_weights: bool = True,
        num_regression: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.class_targets_fn = class_targets_fn
        self.learnable_weights = LearnableWeights() if learnable_weights else None

        # Determine total regression output channels for composite_loss slicing.
        # MultiAssetCTM: N assets each with output_dim regression channels → N * output_dim.
        # CTMStockModel (single-asset): output_dim regression channels.
        if num_regression is not None:
            self.num_regression = num_regression
        elif hasattr(model, 'n_assets') and hasattr(model, 'output_dim'):
            self.num_regression = cast(int, model.n_assets) * cast(int, model.output_dim)
        elif hasattr(model, 'output_dim'):
            self.num_regression = cast(int, model.output_dim)
        else:
            raise ValueError(
                "LossWrapper cannot determine num_regression from model. "
                "Model must have 'output_dim' attribute or pass num_regression explicitly."
            )

    def forward(
        self, pred: torch.Tensor, regression_target: torch.Tensor
    ) -> torch.Tensor:
        class_targets = (
            self.class_targets_fn(regression_target)
            if self.class_targets_fn is not None
            else regression_target.new_zeros(0, dtype=torch.long)
        )
        if class_targets.dim() > 1 and class_targets.shape[-1] == 1:
            class_targets = class_targets.squeeze(-1)
        return composite_loss(
            predictions=pred,
            regression_target=regression_target,
            class_targets=class_targets,
            model_parameters=[p for p in self.model.parameters() if p.ndim > 1],
            config=self.config,
            learnable_weights=self.learnable_weights,
            num_regression=self.num_regression,
        )


class LossWarmupScheduler:
    """Linear warmup schedule for Sharpe loss weight.

    Phase 1 (warmup_steps): MSE-only (lambda_sharpe = 0)
    Phase 2 (ramp_steps):  lambda_sharpe linearly increases from 0 to target
    Phase 3 (steady):      target lambda_sharpe
    """

    def __init__(
        self, target_lambda_sharpe: float, warmup_steps: int = 2000, ramp_steps: int = 3000
    ) -> None:
        self.target = target_lambda_sharpe
        self.warmup_steps = warmup_steps
        self.ramp_steps = ramp_steps

    def get_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return 0.0
        elapsed = step - self.warmup_steps
        ratio = min(1.0, elapsed / max(self.ramp_steps, 1))
        return self.target * ratio

    def adjust_config(self, config: LossConfig, step: int) -> None:
        config.lambda_sharpe = self.get_lambda(step)


def lr_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 200,
    total_epochs: int = 100,
) -> torch.optim.lr_scheduler.SequentialLR:
    """LR schedule: linear warmup → cosine decay.

    The scheduler is stepped **per epoch** (called by ``_train_single_window``
    after each epoch loop). ``total_epochs`` must be ≥ ``warmup_epochs`` to
    ensure a positive ``T_max`` for the cosine phase.
    """
    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sch, cosine_sch],
        milestones=[warmup_epochs],
    )


def train_epoch_advanced(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_wrapper: LossWrapper,
    loss_warmup: LossWarmupScheduler | None,
    device: torch.device,
    global_step: int,
    grad_clip: float = 1.0,
    log_gradients: bool = False,
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch using composite_loss with warmup support.

    Returns (metrics_dict, updated_global_step).
    """
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    total_max_grad = 0.0
    total_pred_std = 0.0
    total_pred_mean = 0.0
    n_batches = 0
    n_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if loss_warmup is not None:
            loss_warmup.adjust_config(loss_wrapper.config, global_step)

        pred = model(batch_x)
        loss = loss_wrapper(pred, batch_y)

        # Sanity monitor: track prediction distribution
        # Variance collapse (std → 0) is the key early-warning signal
        # for Sharpe loss gaming (see defect_patterns.md).
        with torch.no_grad():
            reg_part = pred[..., :loss_wrapper.num_regression] if hasattr(loss_wrapper, 'num_regression') else pred
            total_pred_std += float(reg_part.std().item())
            total_pred_mean += float(reg_part.mean().item())

        optimizer.zero_grad()
        loss.backward()

        actual_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if log_gradients:
            total_grad_norm += float(actual_norm)
            total_max_grad = max(total_max_grad, float(actual_norm))
        optimizer.step()

        n = batch_x.size(0)
        total_loss += loss.item() * n
        n_batches += 1
        n_samples += n
        global_step += 1

    avg_loss = total_loss / max(n_samples, 1)
    avg_grad_norm = total_grad_norm / max(n_batches, 1)
    max_grad = total_max_grad
    avg_pred_std = total_pred_std / max(n_batches, 1)
    avg_pred_mean = total_pred_mean / max(n_batches, 1)
    return {
        "avg_loss": avg_loss,
        "grad_norm": avg_grad_norm,
        "max_grad": max_grad,
        "pred_std": avg_pred_std,
        "pred_mean": avg_pred_mean,
    }, global_step


@torch.no_grad()
def validate_advanced(
    model: nn.Module,
    dataloader: DataLoader,
    loss_wrapper: LossWrapper,
    device: torch.device,
    num_regression: int | None = None,
    is_multi_asset: bool | None = None,
) -> Dict[str, float]:
    """Validation metrics: loss, sharpe ratio, directional accuracy.

    Delegates to :func:`validate` since ``LossWrapper`` matches the
    ``loss_fn(pred, target) -> scalar`` interface.
    """
    return validate(
        model, dataloader, loss_wrapper, device,
        num_regression=num_regression, is_multi_asset=is_multi_asset,
    )


class WalkForwardTrainerAdvanced:
    """Walk-forward validation with composite loss, warmup, and LR schedules.

    Parameters
    ----------
    model_class : nn.Module class to instantiate for each window.
    model_params : dict of constructor kwargs.
    loss_config : LossConfig for composite loss.
    device : torch device.
    """

    def __init__(
        self,
        model_class: Type[nn.Module],
        model_params: Dict[str, Any],
        loss_config: LossConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model_class = model_class
        self.model_params = model_params
        self.loss_config = loss_config
        self.device = torch.device(device)
        # Per-window best state dicts for test-prep warm-start (P0 Fix 2)
        self._window_best_state_dicts: List[Dict[str, torch.Tensor]] = []

    def get_best_ctm_state_dict(self, idx: int) -> Dict[str, torch.Tensor] | None:
        """Return the best state dict from walk-forward window ``idx``.

        Returns None if ``idx`` is out of range or no state dicts are stored.
        """
        if not self._window_best_state_dicts or idx < 0 or idx >= len(self._window_best_state_dicts):
            return None
        return self._window_best_state_dicts[idx]

    def _train_single_window(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        patience: int = 10,
        warmup_steps: int = 2000,
        ramp_steps: int = 3000,
        lr_warmup_epochs: int = 5,
        class_targets_fn: Callable | None = None,
        log_gradients: bool = False,
        is_multi_asset: bool | None = None,
        num_regression: int | None = None,
        init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[nn.Module, List[Dict[str, float]], float, float, int]:
        """Train CTM model for one walk-forward window.

        Parameters
        ----------
        init_state_dict : optional OrderedDict
            Pre-trained state dict to initialise the model.  When
            provided, the model is warm-started from these weights
            before walk-forward training begins.  Useful for
            continuing training from a previous checkpoint.

        Returns (model, window_metrics, best_sharpe, best_ic, epochs_run).
        """
        if lr_warmup_epochs >= n_epochs:
            orig_warmup = lr_warmup_epochs
            lr_warmup_epochs = max(5, n_epochs // 5)
            logging.warning(
                f"lr_warmup_epochs={orig_warmup} >= n_epochs={n_epochs}, clamped to {lr_warmup_epochs}"
            )

        model = self.model_class(**self.model_params).to(self.device)
        if init_state_dict is not None:
            model.load_state_dict(init_state_dict)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_wrapper = LossWrapper(
            config=deepcopy(self.loss_config),
            model=model,
            class_targets_fn=class_targets_fn,
            num_regression=num_regression,
        ).to(self.device)
        loss_warmup = LossWarmupScheduler(
            target_lambda_sharpe=self.loss_config.lambda_sharpe,
            warmup_steps=warmup_steps,
            ramp_steps=ramp_steps,
        )
        lr_scheduler = lr_warmup_cosine(
            optimizer,
            warmup_epochs=lr_warmup_epochs,
            total_epochs=n_epochs,
        )

        best_sharpe = -float("inf")
        best_ic = -float("inf")
        best_state = None
        no_improve = 0
        global_step = 0
        window_metrics = []

        for epoch in range(n_epochs):
            train_m, global_step = train_epoch_advanced(
                model, train_loader, optimizer, loss_wrapper, loss_warmup,
                self.device, global_step, grad_clip, log_gradients=log_gradients,
            )
            if is_multi_asset is None:
                is_multi_asset_val = bool(hasattr(model, 'n_assets') and getattr(model, 'n_assets', 1) > 1)
            else:
                is_multi_asset_val = is_multi_asset
            val_m = validate_advanced(
                model, val_loader, loss_wrapper, self.device,
                num_regression=loss_wrapper.num_regression,
                is_multi_asset=is_multi_asset_val,
            )
            lr_scheduler.step()

            val_ic = val_m.get("spearman_ic", 0.0)
            window_metrics.append({
                "epoch": epoch,
                "train_loss": train_m["avg_loss"],
                "val_loss": val_m["avg_loss"],
                "val_sharpe": val_m["sharpe_ratio"],
                "val_ic": val_ic,
                "val_dir_acc": val_m["directional_accuracy"],
                "grad_norm": train_m.get("grad_norm", 0.0),
                "max_grad": train_m.get("max_grad", 0.0),
                "current_lambda_sharpe": loss_wrapper.config.lambda_sharpe,
                # Sanity monitors for Sharpe gaming detection
                "pred_std": train_m.get("pred_std", 0.0),
                "pred_mean": train_m.get("pred_mean", 0.0),
            })

            if epoch % 5 == 0 or epoch == n_epochs - 1:
                logger.info(
                    "    Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_sharpe=%.4f val_ic=%.4f lr=%.2e",
                    epoch + 1, n_epochs,
                    train_m["avg_loss"], val_m["avg_loss"],
                    val_m["sharpe_ratio"], val_ic,
                    optimizer.param_groups[0]["lr"],
                )

            if val_ic > best_ic:
                best_sharpe = val_m["sharpe_ratio"]
                best_ic = val_ic
                best_state = deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, window_metrics, best_sharpe, best_ic, len(window_metrics)

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
        """Run walk-forward training with purge period and warmup.

        Parameters
        ----------
        data : (N, seq_len, D) pre-sequenced features.
        targets : (N, T, output_dim) regression targets, or (N,) 1-step targets.
        train_window, val_window : fold sizes.
        purge_period : gap (in samples) between train and val to prevent leakage.
        step_size : walk-forward stride.
        warmup_steps, ramp_steps : loss warmup params (Phase 5-4e).
        lr_warmup_epochs : number of epochs for LR linear warmup (applied per-epoch, not per-batch).
        class_targets_fn : optional fn(target)→class_labels for directional loss.

        Returns list of ``TrainingResult`` dataclass instances, one per window.
        """
        N = len(data)
        results: List[TrainingResult] = []

        # ── Determine target shape ──
        # If targets are 1D (N,), reshape to (N, 1, 1) for consistency
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)  # (N, T, 1)

        for pos, train_end, purge_end, val_end in walk_forward_windows(
            N, train_window, val_window, purge_period, step_size
        ):
            train_data = data[pos:train_end]
            train_targ = targets[pos:train_end]
            val_data = data[purge_end:val_end]
            val_targ = targets[purge_end:val_end]

            train_ds = TensorDataset(train_data, train_targ)
            val_ds = TensorDataset(val_data, val_targ)

            # No shuffle — walk-forward windows already provide causality.
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            model, window_metrics, best_sharpe, best_ic, epochs_run = self._train_single_window(
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

            # Store best state dict for test-prep warm-start (P0 Fix 2)
            self._window_best_state_dicts.append(
                {k: v.cpu().clone() for k, v in model.state_dict().items()}
            )

            results.append(TrainingResult(
                window_start=pos,
                window_end=val_end,
                best_sharpe=best_sharpe,
                epochs_run=epochs_run,
                metrics=window_metrics,
                ctm_ic=best_ic,
            ))

        return results
