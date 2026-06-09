"""Tests for gradient_check utility."""

import torch
import torch.nn as nn

from src.utils.gradient_check import compare_gradients


class TestCompareGradients:
    def test_linear_mse_gradients_close(self, linear_grad_model, grad_check_data):
        x, y = grad_check_data
        results = compare_gradients(
            linear_grad_model, nn.MSELoss(), x, y,
            epsilon=1e-4,
        )
        assert len(results) > 0, "Should have at least one parameter"
        for name, err in results.items():
            assert err >= 0.0, f"{name}: relative error should be >= 0, got {err}"
            assert err < 1.0, f"{name}: gradient mismatch too large: {err:.6f}"

    def test_mse_loss_gradients_close(self, linear_grad_model, grad_check_data):
        x, y = grad_check_data
        results = compare_gradients(
            linear_grad_model, nn.MSELoss(), x, y,
            epsilon=1e-4,
        )
        for name, err in results.items():
            assert err < 0.01, (
                f"{name}: relative error {err:.6f} exceeds strict "
                f"tolerance for MSE+Linear"
            )

    def test_single_element_parameter(self, grad_check_data):
        x, y = grad_check_data
        model = nn.Linear(1, 1, bias=False)
        torch.manual_seed(42)
        nn.init.constant_(model.weight, 2.0)
        model.eval()
        results = compare_gradients(model, nn.MSELoss(), x[:, :1], y, epsilon=1e-4)
        assert "weight" in results
        assert 0.0 <= results["weight"] < 1.0, (
            f"Single-param grad error too large: {results['weight']:.6f}"
        )

    def test_no_grad_parameter_handled(self):
        model = nn.Linear(3, 1)
        torch.manual_seed(42)
        x = torch.randn(4, 3)
        y = torch.randn(4, 1)
        with torch.no_grad():
            model.weight.grad = None
            model.bias.grad = None
        results = compare_gradients(model, nn.MSELoss(), x, y, epsilon=1e-4)
        assert len(results) == 2

    def test_compare_gradients_returns_dict(self, linear_grad_model, grad_check_data):
        x, y = grad_check_data
        results = compare_gradients(linear_grad_model, nn.MSELoss(), x, y)
        assert isinstance(results, dict)
        assert all(isinstance(k, str) for k in results)
        assert all(isinstance(v, float) for v in results.values())
