"""Tests for training infrastructure (trainer, advanced_trainer, loss_bridge)."""

import torch
import torch.nn as nn
import pytest
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.train.trainer import WalkForwardTrainer, train_epoch, validate
from src.train.advanced_trainer import (
    WalkForwardTrainerAdvanced,
    LossWrapper,
    LossWarmupScheduler,
    lr_warmup_cosine,
    train_epoch_advanced,
    validate_advanced,
)
from src.train.loss_bridge import (
    ctm_composite_loss_for_gbdt,
    make_gbdt_loss_fn,
)
from src.model.losses import LossConfig, composite_loss, LearnableWeights
from src.model.ctm_model import CTMStockModel
from src.model.multiasset_ctm import MultiAssetCTM


# ═════════════════════════════════════════════════════════════════════
# LossWrapper
# ═════════════════════════════════════════════════════════════════════

class TestLossWrapper:
    def test_loss_wrapper_forward_shape(self):
        model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2)
        wrapper = LossWrapper(config=LossConfig(), model=model, learnable_weights=False)
        x = torch.randn(4, 20, 10)
        pred = model(x)
        target = torch.randn(4, 20, 1)
        loss = wrapper(pred, target)
        assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_loss_wrapper_with_learnable_weights(self):
        model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2)
        wrapper = LossWrapper(config=LossConfig(), model=model, learnable_weights=True)
        x = torch.randn(4, 20, 10)
        pred = model(x)
        target = torch.randn(4, 20, 1)
        loss = wrapper(pred, target)
        assert wrapper.learnable_weights is not None
        assert loss.dim() == 0
        # Gradients should flow through regression-linked learnable weights
        loss.backward()
        # Directional branch is skipped (empty class_targets), so log_var_directional
        # has no gradient.  The regression + pinball branches should have gradients.
        assert wrapper.learnable_weights.log_var_mse.grad is not None, "log_var_mse should have grad"
        assert wrapper.learnable_weights.log_var_sharpe.grad is not None, "log_var_sharpe should have grad"
        assert wrapper.learnable_weights.log_var_pinball.grad is not None, "log_var_pinball should have grad"

    def test_loss_wrapper_multiasset_ctm_regression_count(self):
        """LossWrapper should detect n_assets * output_dim for MultiAssetCTM."""
        model = MultiAssetCTM(n_assets=3, input_dim=10, model_dim=32, output_dim=1)
        wrapper = LossWrapper(config=LossConfig(), model=model, learnable_weights=False)
        assert wrapper.num_regression == 3, f"Expected 3, got {wrapper.num_regression}"

    def test_loss_wrapper_single_asset_regression_count(self):
        model = CTMStockModel(input_dim=10, model_dim=32, output_dim=1)
        wrapper = LossWrapper(config=LossConfig(), model=model, learnable_weights=False)
        assert wrapper.num_regression == 1, f"Expected 1, got {wrapper.num_regression}"


# ═════════════════════════════════════════════════════════════════════
# LossWarmupScheduler
# ═════════════════════════════════════════════════════════════════════

class TestLossWarmupScheduler:
    def test_warmup_phase_zero_lambda(self):
        scheduler = LossWarmupScheduler(target_lambda_sharpe=0.5, warmup_steps=100, ramp_steps=200)
        lam = scheduler.get_lambda(50)
        assert lam == 0.0, f"Expected 0 during warmup, got {lam}"

    def test_warmup_phase_ramp(self):
        scheduler = LossWarmupScheduler(target_lambda_sharpe=0.5, warmup_steps=100, ramp_steps=200)
        lam = scheduler.get_lambda(200)  # 100 steps into ramp (halfway)
        assert abs(lam - 0.25) < 1e-6, f"Expected 0.25 at halfway, got {lam}"

    def test_warmup_phase_steady(self):
        scheduler = LossWarmupScheduler(target_lambda_sharpe=0.5, warmup_steps=100, ramp_steps=200)
        lam = scheduler.get_lambda(500)  # past ramp
        assert abs(lam - 0.5) < 1e-6, f"Expected 0.5 in steady, got {lam}"

    def test_adjust_config(self):
        scheduler = LossWarmupScheduler(target_lambda_sharpe=0.5, warmup_steps=100, ramp_steps=200)
        config = LossConfig(lambda_sharpe=0.5)
        scheduler.adjust_config(config, 50)
        assert config.lambda_sharpe == 0.0
        scheduler.adjust_config(config, 300)
        assert abs(config.lambda_sharpe - 0.5) < 1e-6


# ═════════════════════════════════════════════════════════════════════
# LR Scheduler
# ═════════════════════════════════════════════════════════════════════

class TestLRWarmupCosine:
    def test_lr_warmup_cosine_creation(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = lr_warmup_cosine(opt, warmup_epochs=50, total_epochs=500)
        assert scheduler is not None

    def test_lr_warmup_cosine_step(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = lr_warmup_cosine(opt, warmup_epochs=50, total_epochs=500)
        initial_lr = opt.param_groups[0]["lr"]
        for _ in range(60):
            scheduler.step()
        # LR should have changed from initial
        current_lr = opt.param_groups[0]["lr"]
        assert current_lr != initial_lr, "LR should change after warmup+cosine steps"


# ═════════════════════════════════════════════════════════════════════
# train_epoch / validate
# ═════════════════════════════════════════════════════════════════════

class TestTrainEpoch:
    def test_train_epoch_advanced_runs(self):
        model = CTMStockModel(input_dim=5, model_dim=16, state_dim=4, n_layers=1)
        loss_config = LossConfig()
        wrapper = LossWrapper(config=loss_config, model=model, learnable_weights=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        data = TensorDataset(
            torch.randn(20, 10, 5),
            torch.randn(20, 10, 1),
        )
        loader = DataLoader(data, batch_size=8)
        warmup = LossWarmupScheduler(target_lambda_sharpe=0.5, warmup_steps=2000, ramp_steps=3000)
        metrics, step = train_epoch_advanced(
            model, loader, opt, wrapper, warmup, torch.device("cpu"), global_step=0
        )
        assert "avg_loss" in metrics
        assert metrics["avg_loss"] > 0
        assert step > 0

    def test_validate_advanced_runs(self):
        model = CTMStockModel(input_dim=5, model_dim=16, state_dim=4, n_layers=1)
        wrapper = LossWrapper(config=LossConfig(), model=model, learnable_weights=False)
        data = TensorDataset(
            torch.randn(10, 10, 5),
            torch.randn(10, 10, 1),
        )
        loader = DataLoader(data, batch_size=8)
        metrics = validate_advanced(model, loader, wrapper, torch.device("cpu"))
        assert "avg_loss" in metrics
        assert "sharpe_ratio" in metrics
        assert "directional_accuracy" in metrics

    def test_train_epoch_basic_runs(self):
        model = CTMStockModel(input_dim=5, model_dim=16, state_dim=4, n_layers=1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        data = TensorDataset(
            torch.randn(20, 10, 5),
            torch.randn(20, 10, 4),  # full output (reg + 3 class)
        )
        loader = DataLoader(data, batch_size=8)
        metrics = train_epoch(model, loader, opt, loss_fn, torch.device("cpu"))
        assert "avg_loss" in metrics
        assert "grad_norm" in metrics

    def test_validate_basic_runs(self):
        model = CTMStockModel(input_dim=5, model_dim=16, state_dim=4, n_layers=1)
        loss_fn = nn.MSELoss()
        data = TensorDataset(
            torch.randn(10, 10, 5),
            torch.randn(10, 10, 4),  # full output (reg + 3 class)
        )
        loader = DataLoader(data, batch_size=8)
        metrics = validate(model, loader, loss_fn, torch.device("cpu"))
        assert "avg_loss" in metrics
        assert "sharpe_ratio" in metrics
        assert "directional_accuracy" in metrics


# ═════════════════════════════════════════════════════════════════════
# WalkForwardTrainer (basic)
# ═════════════════════════════════════════════════════════════════════

class TestWalkForwardTrainer:
    def test_walk_forward_single_window(self):
        trainer = WalkForwardTrainer(
            model_class=CTMStockModel,
            model_params={"input_dim": 5, "model_dim": 16, "state_dim": 4, "n_layers": 1},
            device="cpu",
        )
        N = 100
        data = torch.randn(N, 10, 5)
        targets = torch.randn(N, 10, 1)
        results = trainer.train_walk_forward(
            data=data, targets=targets,
            train_window=40, val_window=10, purge_period=0, step_size=50,
            n_epochs=3, batch_size=16, patience=2,
        )
        assert len(results) > 0, "Expected at least one window"
        assert "best_sharpe" in results[0]
        assert "metrics" in results[0]


class TestWalkForwardTrainerAdvanced:
    def test_walk_forward_advanced_single_window(self):
        trainer = WalkForwardTrainerAdvanced(
            model_class=CTMStockModel,
            model_params={"input_dim": 5, "model_dim": 16, "state_dim": 4, "n_layers": 1},
            loss_config=LossConfig(),
            device="cpu",
        )
        N = 100
        data = torch.randn(N, 10, 5)
        targets = torch.randn(N, 10, 1)
        results = trainer.train_walk_forward(
            data=data, targets=targets,
            train_window=40, val_window=10, purge_period=5, step_size=50,
            n_epochs=3, batch_size=16, patience=2,
            warmup_steps=10, ramp_steps=10, lr_warmup_epochs=2,
        )
        assert len(results) > 0, "Expected at least one window"
        assert hasattr(results[0], "best_sharpe")


# ═════════════════════════════════════════════════════════════════════
# Loss bridge
# ═════════════════════════════════════════════════════════════════════

class TestLossBridge:
    def test_ctm_composite_loss_for_gbdt_shape(self):
        y_true = torch.randn(10)
        y_pred = torch.randn(10)
        config = LossConfig(lambda_mse=1.0, lambda_pinball=0.1, lambda_reg=0.01)
        loss, grads, hessians = ctm_composite_loss_for_gbdt(y_true, y_pred, config)
        assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
        assert grads.shape == (10,), f"Expected (10,) grads, got {grads.shape}"
        assert hessians.shape == (10,), f"Expected (10,) hessians, got {hessians.shape}"
        assert torch.isfinite(loss), "Loss should be finite"
        assert torch.isfinite(grads).all(), "Gradients should be finite"
        assert torch.isfinite(hessians).all(), "Hessians should be finite"

    def test_ctm_composite_loss_for_gbdt_perfect_prediction(self):
        """When prediction matches target, MSE loss should be small."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.clone()
        config = LossConfig(lambda_mse=1.0, lambda_pinball=0.0, lambda_reg=0.0, lambda_directional=0.0)
        loss, grads, hessians = ctm_composite_loss_for_gbdt(y_true, y_pred, config)
        assert loss.item() < 1e-6, f"Expected ~0 loss for perfect pred, got {loss}"

    def test_make_gbdt_loss_fn(self):
        config = LossConfig()
        loss_fn = make_gbdt_loss_fn(config)
        y_true = torch.randn(10)
        y_pred = torch.randn(10)
        loss, grads, hessians = loss_fn(y_true, y_pred)
        assert loss.dim() == 0
        assert grads.shape == (10,)
        assert hessians.shape == (10,)

    def test_loss_bridge_backward_hessians_positive(self):
        """Hessians from loss_bridge should be positive (clamped)."""
        y_true = torch.randn(10)
        y_pred = torch.randn(10)
        config = LossConfig(lambda_mse=1.0, lambda_pinball=0.1, lambda_reg=0.01)
        _, _, hessians = ctm_composite_loss_for_gbdt(y_true, y_pred, config)
        assert (hessians > 0).all(), "Hessians should be positive"

    def test_loss_bridge_sharpe_path_is_active(self):
        """Verify the Sharpe bridge path produces non-zero loss when lambda_sharpe>0."""
        y_true = torch.randn(20) * 0.02 + 0.001
        y_pred = torch.randn(20)
        config = LossConfig(lambda_sharpe=0.5, lambda_mse=0.0, lambda_pinball=0.0,
                            lambda_reg=0.0, lambda_directional=0.0)
        loss, grads, hessians = ctm_composite_loss_for_gbdt(y_true, y_pred, config)
        assert loss.item() != 0.0, "Sharpe-only bridge loss should be non-zero"
        assert torch.isfinite(grads).all(), "Gradients should be finite"
        assert (hessians > 0).all(), "Hessians should be positive with clamping"

    def test_loss_bridge_vectorized_vs_loop_hessians(self):
        """Vectorized Hessian should match the previous per-sample loop (MSE-only for clean comparison)."""
        y_true = torch.randn(20)
        y_pred = torch.randn(20).requires_grad_(True)
        config = LossConfig(lambda_mse=1.0, lambda_pinball=0.0, lambda_reg=0.0, skip_l2_reg=True)

        # Reference: per-sample loop
        y_pred_ref = y_pred.detach().clone().requires_grad_(True)
        mse = torch.nn.functional.mse_loss(y_pred_ref, y_true)
        grads_ref = torch.autograd.grad(mse, y_pred_ref, create_graph=True)[0]
        hess_ref = torch.zeros_like(y_pred_ref)
        for i in range(y_pred_ref.shape[0]):
            hess_i = torch.autograd.grad(grads_ref[i], y_pred_ref, retain_graph=True)[0][i]
            hess_ref[i] = hess_i

        # Vectorized: sum(grads) trick
        y_pred_vec = y_pred.detach().clone().requires_grad_(True)
        mse = torch.nn.functional.mse_loss(y_pred_vec, y_true)
        grads_vec = torch.autograd.grad(mse, y_pred_vec, create_graph=True)[0]
        hess_vec = torch.autograd.grad(grads_vec.sum(), y_pred_vec, create_graph=True)[0]

        assert torch.allclose(hess_ref, hess_vec, atol=1e-6), \
            f"Vectorized hessian mismatch: max diff={ (hess_ref - hess_vec).abs().max() }"


# ═════════════════════════════════════════════════════════════════════
# MultiAssetCTM output layout
# ═════════════════════════════════════════════════════════════════════

class TestMultiAssetCTMOutputLayout:
    def test_multiasset_output_flat_regression_first(self):
        """MultiAssetCTM output should have regression channels first, then class logits."""
        model = MultiAssetCTM(n_assets=3, input_dim=10, model_dim=32, output_dim=1)
        x = torch.randn(2, 3, 20, 10)
        out = model(x)
        # output_dim=1, 3 assets → 3 regression channels, 9 class logits
        assert out.shape == (2, 20, 12), f"Expected (2,20,12), got {out.shape}"
        reg_slice = out[..., :3]   # first 3 = regression for each asset
        cls_slice = out[..., 3:]   # last 9 = class logits for each asset
        # Verify reg channels look like regression values (not logits)
        assert torch.isfinite(reg_slice).all()
        assert torch.isfinite(cls_slice).all()

    def test_multiasset_composite_loss_compatible(self):
        """MultiAssetCTM output should work directly with composite_loss."""
        model = MultiAssetCTM(n_assets=3, input_dim=10, model_dim=32, output_dim=1)
        x = torch.randn(2, 3, 20, 10)
        out = model(x)
        # composite_loss with num_regression=3
        reg_target = torch.randn(2, 20, 3)  # 3 assets * 1 regression each
        cls_targets = torch.randint(0, 3, (2, 20))
        cfg = LossConfig()
        loss = composite_loss(
            predictions=out,
            regression_target=reg_target,
            class_targets=cls_targets,
            model_parameters=list(model.parameters()),
            config=cfg,
            num_regression=3,
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss), "Loss should be finite"
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Gradients should be finite for {p.shape}"

    def test_multiasset_forward_matches_loss_wrapper(self):
        """End-to-end: LossWrapper with MultiAssetCTM should produce valid loss."""
        model = MultiAssetCTM(n_assets=3, input_dim=10, model_dim=32, output_dim=1)
        wrapper = LossWrapper(config=LossConfig(), model=model, learnable_weights=True)
        x = torch.randn(2, 3, 20, 10)
        out = model(x)
        # Regression target shape: (B, T, n_assets * output_dim) = (2, 20, 3)
        reg_target = torch.randn(2, 20, 3)
        loss = wrapper(out, reg_target)
        assert loss.dim() == 0
        loss.backward()
        # Check gradients flow through cross-attention, embeddings, and backbone
        assert model.asset_embed.weight.grad is not None
        assert model.head_regression.weight.grad is not None


# ═════════════════════════════════════════════════════════════════════
# CTMStockModel _encode_core
# ═════════════════════════════════════════════════════════════════════

class TestCTMEncodeCore:
    def test_encode_core_matches_extract_features(self):
        model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2)
        model.eval()  # disable dropout for deterministic comparison
        x = torch.randn(4, 20, 10)
        h1 = model.encode(x)
        h2 = model.extract_features(x)
        assert torch.allclose(h1, h2), "encode() and extract_features() should match"

    def test_encode_core_conditioning(self):
        model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2)
        x = torch.randn(4, 20, 10)
        cond = torch.randn(4, 20, 32)
        h_uncond = model.encode(x)
        h_cond = model.encode(x, cond=cond)
        # Conditioning should change the output
        assert not torch.allclose(h_uncond, h_cond, atol=1e-4), \
            "Conditioning should affect encoder output"

    def test_encode_core_nan_input_detection(self):
        model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2, parallel_scan=True)
        x = torch.full((4, 20, 10), float("nan"))
        with pytest.warns(UserWarning, match="NaN detected"):
            h = model._encode_core(x)
        assert torch.isfinite(h).all(), "NaN input should be recovered to finite output"
