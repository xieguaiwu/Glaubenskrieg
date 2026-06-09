"""Gradient checking utility for CTM model verification.

Compares analytical (autograd) vs numerical (finite difference) gradients.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def compare_gradients(
    model: nn.Module,
    loss_fn: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 1e-4,
) -> Dict[str, float]:
    """Compute relative error between analytical and numerical gradients.

    Parameters
    ----------
    model : nn.Module with parameters to check.
    loss_fn : callable(pred, target) → scalar loss.
    x : input tensor.
    y : target tensor.
    epsilon : perturbation step for finite difference.

    Returns
    -------
    dict mapping param_name → relative_error. error < 0 means no grad.
    """
    model.eval()

    # Analytical gradients
    model.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()

    results: Dict[str, float] = {}

    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            results[name] = -1.0
            continue

        n = param.numel()
        k = min(n, max(20, n // 10))
        flat_indices = torch.randperm(n)[:k]

        g_ana = torch.zeros(k)
        g_num = torch.zeros(k)
        orig = param.data.clone()

        for i, idx in enumerate(flat_indices):
            orig_val = orig.flatten()[idx].item()

            param.data.flatten()[idx] = orig_val + epsilon
            loss_plus = loss_fn(model(x), y).item()

            param.data.flatten()[idx] = orig_val - epsilon
            loss_minus = loss_fn(model(x), y).item()

            param.data.flatten()[idx] = orig_val  # restore

            g_num[i] = (loss_plus - loss_minus) / (2.0 * epsilon)
            g_ana[i] = grad.flatten()[idx].item()

        param.data.copy_(orig)  # ensure clean restore

        diff = (g_ana - g_num).norm().item()
        denom = max(g_ana.norm().item(), g_num.norm().item(), 1e-12)
        results[name] = diff / denom

    return results

