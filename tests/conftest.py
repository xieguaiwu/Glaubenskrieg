"""Shared pytest fixtures and configuration for Glaubenskrieg tests."""

import numpy as np
import pytest
import torch

from src.model.ctm_model import CTMStockModel
from src.model.losses import LossConfig


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "ensemble: marks tests requiring ensemble (GBDT) module")


# ═══════════════════════════════════════════════════════════════════
# Core model fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture(scope="function")
def small_model() -> CTMStockModel:
    """Small CTMStockModel for fast tests (seed=42).

    input_dim=5, model_dim=16, state_dim=4, n_layers=1.
    """
    torch.manual_seed(42)
    model = CTMStockModel(input_dim=5, model_dim=16, state_dim=4, n_layers=1)
    model.eval()
    return model


@pytest.fixture(scope="module")
def small_ctm_model(small_model: CTMStockModel) -> CTMStockModel:
    return small_model


@pytest.fixture(scope="session")
def small_ctm_params() -> dict:
    """Minimal CTMStockModel constructor parameters."""
    return {
        "input_dim": 5,
        "model_dim": 16,
        "state_dim": 4,
        "n_layers": 1,
        "output_dim": 1,
    }


# ═══════════════════════════════════════════════════════════════════
# Tensor fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def random_input() -> torch.Tensor:
    """Random input tensor with seed for reproducibility: (4, 10, 5)."""
    torch.manual_seed(42)
    return torch.randn(4, 10, 5)


@pytest.fixture
def random_target() -> torch.Tensor:
    """Random regression target: (4, 10, 1)."""
    torch.manual_seed(42)
    return torch.randn(4, 10, 1)


@pytest.fixture
def random_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Random (input, target) tuple for batch training: (8, 10, 5), (8, 10, 1)."""
    torch.manual_seed(42)
    x = torch.randn(8, 10, 5)
    y = torch.randn(8, 10, 1)
    return x, y


@pytest.fixture
def random_batch_large() -> tuple[torch.Tensor, torch.Tensor]:
    """Larger random batch for walk-forward tests: (100, 10, 5), (100, 10, 1)."""
    torch.manual_seed(42)
    x = torch.randn(100, 10, 5)
    y = torch.randn(100, 10, 1)
    return x, y


@pytest.fixture
def random_sequences() -> np.ndarray:
    """Random numpy sequences for gbdt_features tests: (10, 20, 5)."""
    rng = np.random.RandomState(42)
    return rng.randn(10, 20, 5)


@pytest.fixture
def random_hidden_states() -> np.ndarray:
    """Random CTM hidden states: (10, 20, 16)."""
    rng = np.random.RandomState(42)
    return rng.randn(10, 20, 16)


# ═══════════════════════════════════════════════════════════════════
# Config fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def loss_config() -> LossConfig:
    """Default LossConfig for tests."""
    return LossConfig()


@pytest.fixture
def mse_only_loss_config() -> LossConfig:
    """LossConfig with only MSE enabled (all other lambdas = 0)."""
    return LossConfig(
        lambda_mse=1.0,
        lambda_sharpe=0.0,
        lambda_directional=0.0,
        lambda_pinball=0.0,
        lambda_reg=0.0,
    )


# ═══════════════════════════════════════════════════════════════════
# Gradient check fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def linear_grad_model() -> torch.nn.Linear:
    """Simple Linear model for gradient checking."""
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 1)
    model.eval()
    return model


@pytest.fixture
def grad_check_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Input/target pair for gradient check: (4, 3) → (4, 1)."""
    torch.manual_seed(42)
    x = torch.randn(4, 3)
    y = torch.randn(4, 1)
    return x, y
