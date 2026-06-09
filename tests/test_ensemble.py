"""Tests for EnsembleWalkForwardTrainer and helper functions."""

import numpy as np
import pytest
import torch

from src.model.ctm_model import CTMStockModel
from src.model.losses import LossConfig

# ── Import ensemble trainer (scipy-dependent) ──────────────────
# scipy is a hard dependency (pyproject.toml), but guard for edge cases.
try:
    from src.train.ensemble_trainer import (
        EnsembleWalkForwardTrainer,
        TrainingResult,
        _compute_sharpe,
        _safe_ic,
    )
    _ENSEMBLE_AVAILABLE = True
except ImportError:
    _ENSEMBLE_AVAILABLE = False


pytestmark = pytest.mark.ensemble

_requires_ensemble = pytest.mark.skipif(
    not _ENSEMBLE_AVAILABLE, reason="ensemble_trainer requires scipy"
)


# ═══════════════════════════════════════════════════════════════════
# Helper function tests (no model dependency)
# ═══════════════════════════════════════════════════════════════════


class TestComputeSharpe:
    @_requires_ensemble
    def test_positive_returns(self):
        preds = np.array([0.01, 0.02, 0.015, 0.025, 0.01], dtype=np.float64)
        sr = _compute_sharpe(preds)
        assert sr > 0, f"Expected positive Sharpe, got {sr}"

    @_requires_ensemble
    def test_constant_returns_zero(self):
        preds = np.ones(10, dtype=np.float64)
        sr = _compute_sharpe(preds)
        assert abs(sr) < 1e-10, f"Constant preds should give 0 Sharpe, got {sr}"

    @_requires_ensemble
    def test_single_element_zero(self):
        sr = _compute_sharpe(np.array([0.05]))
        assert abs(sr) < 1e-10

    @_requires_ensemble
    def test_empty_array_zero(self):
        sr = _compute_sharpe(np.array([]))
        assert abs(sr) < 1e-10

    @_requires_ensemble
    def test_annual_factor_parameter(self):
        preds = np.array([0.01, 0.03, 0.02, -0.01], dtype=np.float64)
        sr1 = _compute_sharpe(preds, annual_factor=252.0)
        sr12 = _compute_sharpe(preds, annual_factor=12.0)
        ratio = sr1 / sr12 if abs(sr12) > 1e-12 else float("inf")
        expected_ratio = (252.0 / 12.0) ** 0.5
        assert abs(ratio - expected_ratio) < 0.01, (
            f"Sharpe ratio should scale with sqrt(annual_factor): "
            f"got {sr1}/{sr12} = {ratio:.4f}, expected ~{expected_ratio:.4f}"
        )


class TestSafeIC:
    @_requires_ensemble
    def test_perfect_positive(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = _safe_ic(y_true, y_pred)
        assert abs(ic - 1.0) < 1e-6, f"Expected 1.0, got {ic}"

    @_requires_ensemble
    def test_perfect_negative(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ic = _safe_ic(y_true, y_pred)
        assert abs(ic - (-1.0)) < 1e-6, f"Expected -1.0, got {ic}"

    @_requires_ensemble
    def test_nan_in_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, np.nan, 3.0])
        ic = _safe_ic(y_true, y_pred)
        assert abs(ic - 1.0) < 1e-6, f"NaN should be filtered, got {ic}"

    @_requires_ensemble
    def test_too_few_samples(self):
        ic = _safe_ic(np.array([1.0]), np.array([1.0]))
        assert abs(ic) < 1e-10

    @_requires_ensemble
    def test_constant_true_values(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        ic = _safe_ic(y_true, y_pred)
        assert abs(ic) < 1e-10, f"Constant y_true should give 0 IC, got {ic}"


# ═══════════════════════════════════════════════════════════════════
# EnsembleWalkForwardTrainer tests
# ═══════════════════════════════════════════════════════════════════


class TestEnsembleWalkForwardTrainerInit:
    @_requires_ensemble
    def test_initializes_with_ctm_model(self):
        trainer = EnsembleWalkForwardTrainer(
            model_class=CTMStockModel,
            model_params={"input_dim": 5, "model_dim": 16, "state_dim": 4, "n_layers": 1},
            loss_config=LossConfig(),
            gbdt_loss="mse",
            device="cpu",
        )
        assert trainer.model_class is CTMStockModel
        assert trainer.device == torch.device("cpu")
        assert trainer.gbdt_loss == "mse"

    @_requires_ensemble
    def test_default_gbdt_config(self):
        trainer = EnsembleWalkForwardTrainer(
            model_class=CTMStockModel,
            model_params={"input_dim": 5, "model_dim": 16, "state_dim": 4, "n_layers": 1},
            loss_config=LossConfig(),
            device="cpu",
        )
        assert trainer.gbdt_config == {}

    @_requires_ensemble
    def test_custom_gbdt_config(self):
        trainer = EnsembleWalkForwardTrainer(
            model_class=CTMStockModel,
            model_params={"input_dim": 5, "model_dim": 16, "state_dim": 4, "n_layers": 1},
            loss_config=LossConfig(),
            gbdt_config={"num_trees": 100, "max_depth": 6, "include_ctm_features": False},
            device="cpu",
        )
        assert trainer.gbdt_config["num_trees"] == 100
        assert trainer.gbdt_config["include_ctm_features"] is False

    @_requires_ensemble
    def test_device_string_conversion(self):
        trainer = EnsembleWalkForwardTrainer(
            model_class=CTMStockModel,
            model_params={"input_dim": 5, "model_dim": 16, "state_dim": 4, "n_layers": 1},
            loss_config=LossConfig(),
            device="cpu",
        )
        assert isinstance(trainer.device, torch.device)


class TestPrepareGBDTData:
    @_requires_ensemble
    def test_output_shapes_without_ctm(self):
        data_seq = np.random.RandomState(42).randn(10, 20, 5).astype(np.float32)
        target_seq = np.random.RandomState(42).randn(10, 20, 1).astype(np.float32)
        X, y = EnsembleWalkForwardTrainer._prepare_gbdt_data(
            data_seq, target_seq, ctm_hidden=None, include_ctm=False,
        )
        N, _, D = data_seq.shape
        assert X.shape == (N, 6 * D), f"Expected ({N}, {6*D}), got {X.shape}"
        assert y.shape == (N,), f"Expected ({N},), got {y.shape}"
        np.testing.assert_allclose(y, target_seq[:, -1, 0])

    @_requires_ensemble
    def test_output_shapes_with_ctm_hidden(self):
        rng = np.random.RandomState(42)
        data_seq = rng.randn(8, 15, 4).astype(np.float32)
        target_seq = rng.randn(8, 15, 1).astype(np.float32)
        hidden = rng.randn(8, 15, 16).astype(np.float32)
        X, y = EnsembleWalkForwardTrainer._prepare_gbdt_data(
            data_seq, target_seq, ctm_hidden=hidden, include_ctm=True,
        )
        N, _, D = data_seq.shape
        d_model = hidden.shape[-1]
        expected_feats = 6 * D + 2 * d_model
        assert X.shape == (N, expected_feats), (
            f"Expected ({N}, {expected_feats}), got {X.shape}"
        )
        assert y.shape == (N,)

    @_requires_ensemble
    def test_include_ctm_false_ignores_provided_hidden(self):
        rng = np.random.RandomState(42)
        data_seq = rng.randn(6, 12, 3).astype(np.float32)
        target_seq = rng.randn(6, 12, 1).astype(np.float32)
        hidden = rng.randn(6, 12, 8).astype(np.float32)
        X, _ = EnsembleWalkForwardTrainer._prepare_gbdt_data(
            data_seq, target_seq, ctm_hidden=hidden, include_ctm=False,
        )
        assert X.shape == (6, 6 * 3)

    @_requires_ensemble
    def test_single_sample(self):
        data_seq = np.random.RandomState(42).randn(1, 10, 5).astype(np.float32)
        target_seq = np.random.RandomState(42).randn(1, 10, 1).astype(np.float32)
        X, y = EnsembleWalkForwardTrainer._prepare_gbdt_data(
            data_seq, target_seq, ctm_hidden=None, include_ctm=False,
        )
        assert X.shape == (1, 30)
        assert y.shape == (1,)

    @_requires_ensemble
    def test_large_output_dimension_target(self):
        rng = np.random.RandomState(42)
        data_seq = rng.randn(5, 10, 5).astype(np.float32)
        target_seq = rng.randn(5, 10, 3).astype(np.float32)
        X, y = EnsembleWalkForwardTrainer._prepare_gbdt_data(
            data_seq, target_seq, ctm_hidden=None, include_ctm=False,
        )
        assert X.shape == (5, 30)
        assert y.shape == (5,)
        np.testing.assert_allclose(y, target_seq[:, -1, 0])


class TestTrainWalkForward:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.trainer = EnsembleWalkForwardTrainer(
            model_class=CTMStockModel,
            model_params={"input_dim": 5, "model_dim": 16, "state_dim": 4, "n_layers": 1},
            loss_config=LossConfig(),
            device="cpu",
        )

    @_requires_ensemble
    def test_returns_list_of_training_results(self):
        N = 100
        data = torch.randn(N, 10, 5)
        targets = torch.randn(N, 10, 1)
        results = self.trainer.train_walk_forward(
            data=data,
            targets=targets,
            train_window=40,
            val_window=10,
            purge_period=5,
            step_size=55,
            n_epochs=2,
            batch_size=8,
            patience=2,
            warmup_steps=2,
            ramp_steps=3,
            lr_warmup_epochs=1,
        )
        assert isinstance(results, list), f"Expected list, got {type(results)}"
        assert len(results) >= 1, "Expected at least one window"
        assert isinstance(results[0], TrainingResult)

    @_requires_ensemble
    def test_result_attributes(self):
        data = torch.randn(100, 10, 5)
        targets = torch.randn(100, 10, 1)
        results = self.trainer.train_walk_forward(
            data=data,
            targets=targets,
            train_window=40,
            val_window=10,
            purge_period=5,
            step_size=55,
            n_epochs=2,
            batch_size=8,
            patience=2,
            warmup_steps=2,
            ramp_steps=3,
            lr_warmup_epochs=1,
        )
        expected_attrs = {
            "window_start", "window_end", "ctm_sharpe", "gbdt_ic",
            "ensemble_sharpe", "ensemble_ic", "ctm_weight", "gbdt_weight",
            "best_sharpe", "epochs_run", "metrics", "ctm_metrics",
            "gbdt_metrics",
        }
        for attr in expected_attrs:
            assert hasattr(results[0], attr), f"Missing attribute '{attr}' in TrainingResult"

    @_requires_ensemble
    def test_1d_targets_reshaped(self):
        data = torch.randn(100, 10, 5)
        targets = torch.randn(100)  # 1D
        results = self.trainer.train_walk_forward(
            data=data,
            targets=targets,
            train_window=40,
            val_window=10,
            purge_period=5,
            step_size=55,
            n_epochs=2,
            batch_size=8,
            patience=2,
            warmup_steps=2,
            ramp_steps=3,
            lr_warmup_epochs=1,
        )
        assert len(results) >= 1

    @_requires_ensemble
    def test_2d_targets_reshaped(self):
        data = torch.randn(100, 10, 5)
        targets = torch.randn(100, 10)  # 2D
        results = self.trainer.train_walk_forward(
            data=data,
            targets=targets,
            train_window=40,
            val_window=10,
            purge_period=5,
            step_size=55,
            n_epochs=2,
            batch_size=8,
            patience=2,
            warmup_steps=2,
            ramp_steps=3,
            lr_warmup_epochs=1,
        )
        assert len(results) >= 1

    @_requires_ensemble
    def test_train_ensemble_returns_ctm_sharpe(self):
        data = torch.randn(100, 10, 5)
        targets = torch.randn(100, 10, 1)
        results = self.trainer.train_walk_forward(
            data=data,
            targets=targets,
            train_window=40,
            val_window=10,
            purge_period=5,
            step_size=55,
            n_epochs=2,
            batch_size=8,
            patience=2,
            warmup_steps=2,
            ramp_steps=3,
            lr_warmup_epochs=1,
        )
        assert len(results) >= 1
        assert hasattr(results[0], "gbdt_metrics")
        assert hasattr(results[0], "ctm_sharpe")
        assert isinstance(results[0].ctm_sharpe, (float, int))

    @_requires_ensemble
    def test_ensemble_sharpe_is_finite(self):
        data = torch.randn(100, 10, 5)
        targets = torch.randn(100, 10, 1)
        results = self.trainer.train_walk_forward(
            data=data,
            targets=targets,
            train_window=40,
            val_window=10,
            purge_period=5,
            step_size=55,
            n_epochs=2,
            batch_size=8,
            patience=2,
            warmup_steps=2,
            ramp_steps=3,
            lr_warmup_epochs=1,
        )
        for r in results:
            assert np.isfinite(r.ensemble_sharpe), (
                f"ensemble_sharpe should be finite, got {r.ensemble_sharpe}"
            )
            assert np.isfinite(r.ctm_sharpe)
            assert np.isfinite(r.ensemble_ic)

    @_requires_ensemble
    def test_weights_in_valid_range(self):
        data = torch.randn(100, 10, 5)
        targets = torch.randn(100, 10, 1)
        results = self.trainer.train_walk_forward(
            data=data,
            targets=targets,
            train_window=40,
            val_window=10,
            purge_period=5,
            step_size=55,
            n_epochs=2,
            batch_size=8,
            patience=2,
            warmup_steps=2,
            ramp_steps=3,
            lr_warmup_epochs=1,
        )
        for r in results:
            assert 0.0 <= r.ctm_weight <= 1.0, (
                f"ctm_weight out of range: {r.ctm_weight}"
            )
            assert 0.0 <= r.gbdt_weight <= 1.0
