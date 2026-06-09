"""Tests for P3EnsembleTrainer — three-stage fusion trainer smoke tests."""

import torch
import pytest

from src.model.losses import LossConfig

try:
    from src.train.p3_trainer import P3EnsembleTrainer
    from src.train.ensemble_trainer import EnsembleWalkForwardTrainer
    _P3_AVAILABLE = True
except ImportError:
    _P3_AVAILABLE = False
    P3EnsembleTrainer = None  # type: ignore
    EnsembleWalkForwardTrainer = None  # type: ignore

pytestmark = pytest.mark.ensemble

_requires_p3 = pytest.mark.skipif(
    not _P3_AVAILABLE, reason="P3EnsembleTrainer requires ensemble module (scipy)"
)


# ── Minimal mock model for init tests ──────────────────────────────


class _MockModel(torch.nn.Module):
    """Minimal nn.Module for testing P3EnsembleTrainer.__init__."""

    def __init__(self, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(5, 1)

    def forward(self, x):
        return self.linear(x.mean(dim=1))


# ═══════════════════════════════════════════════════════════════════
# P3EnsembleTrainer smoke tests
# ═══════════════════════════════════════════════════════════════════


class TestP3EnsembleTrainer:
    """Smoke tests for P3EnsembleTrainer initialisation and class hierarchy."""

    @_requires_p3
    def test_init_requires_multi_asset(self):
        """Init with n_assets=1 does NOT raise — the check is in train_walk_forward."""
        trainer = P3EnsembleTrainer(  # type: ignore[misc]
            model_class=_MockModel,
            model_params={"n_assets": 1, "input_dim": 5, "model_dim": 16},
            loss_config=LossConfig(),
        )
        assert trainer.modulator_epochs == 10
        assert trainer.modulator_lr == 1e-4
        assert trainer.modulator_patience == 3

    @_requires_p3
    def test_init_accepts_multi_asset_params(self):
        """Init with n_assets > 1 succeeds and stores modulator settings."""
        trainer = P3EnsembleTrainer(  # type: ignore[misc]
            model_class=_MockModel,
            model_params={"n_assets": 3, "input_dim": 15, "model_dim": 32},
            loss_config=LossConfig(),
        )
        assert trainer.device == torch.device("cpu")
        assert trainer.modulator_epochs == 10

    @_requires_p3
    def test_init_custom_modulator_params(self):
        """Custom modulator_epochs, modulator_lr, modulator_patience are stored."""
        trainer = P3EnsembleTrainer(  # type: ignore[misc]
            model_class=_MockModel,
            model_params={"n_assets": 5, "input_dim": 10, "model_dim": 16},
            loss_config=LossConfig(),
            modulator_epochs=20,
            modulator_lr=5e-4,
            modulator_patience=5,
        )
        assert trainer.modulator_epochs == 20
        assert trainer.modulator_lr == 5e-4
        assert trainer.modulator_patience == 5

    @_requires_p3
    def test_is_subclass(self):
        """P3EnsembleTrainer is a subclass of EnsembleWalkForwardTrainer."""
        assert issubclass(P3EnsembleTrainer, EnsembleWalkForwardTrainer), (  # type: ignore[arg-type]
            "P3EnsembleTrainer must inherit from EnsembleWalkForwardTrainer"
        )
