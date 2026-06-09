"""Generalized Mean Absolute Directional Loss (GMADL).

Reference: Glaubenskrieg v5 — custom loss function for directional-aware
regression.  Penalizes sign-misaligned predictions more heavily than
directionally-correct ones, while preserving the L1 (MAE) sensitivity to
magnitude error.

Formula
-------
.. math::

    L = \\frac{1}{N} \\sum_{i=1}^{N}
        |y_i - \\hat{y}_i| \\cdot w_i

where the per-sample directional weight :math:`w_i` is:

.. math::

    w_i =
    \\begin{cases}
        \\exp(-\\alpha) & \\text{if } \\operatorname{sign}(\\hat{y}_i) =
            \\operatorname{sign}(y_i) \\\\
        \\exp(+\\alpha) & \\text{if } \\operatorname{sign}(\\hat{y}_i) \\neq
            \\operatorname{sign}(y_i)
    \\end{cases}

:math:`\\alpha \\geq 0` controls the directional penalty gap.
When :math:`\\alpha = 0`, GMADL reduces to standard MAE.
Default :math:`\\alpha = 0.5` gives :math:`\\exp(\\pm 0.5) \\approx
[0.6065, 1.6487]` — approximately 2.7× penalty for sign-misaligned
predictions.

Motivation
----------
In financial return prediction, getting the *direction* right is often
more important than exact magnitude.  GMADL encodes this prior directly
into the loss landscape without requiring a separate classification head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GMADLLoss(nn.Module):
    """Generalized Mean Absolute Directional Loss.

    Parameters
    ----------
    alpha : float, default 0.5
        Directional penalty parameter.  Higher values widen the gap
        between same-sign and opposite-sign sample weights.
        Must be ≥ 0.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the GMADL scalar loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values, any shape — internally flattened.
        y_true : torch.Tensor
            Ground-truth values, same shape as ``y_pred``.

        Returns
        -------
        torch.Tensor
            Scalar loss (0-dimensional tensor).
        """
        return compute(y_pred, y_true, self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha:.4g}"


def compute(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Standalone GMADL computation.

    Can be used without instantiating ``GMADLLoss``, e.g. for logging,
    evaluation, or one-off calculations.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted values, any shape.
    y_true : torch.Tensor
        Ground-truth values, same shape as ``y_pred``.
    alpha : float, default 0.5
        Directional penalty parameter.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        raise TypeError("Both y_pred and y_true must be torch.Tensor instances")

    # Flatten to support arbitrary shapes
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    if y_pred.numel() == 0:
        return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

    abs_errors = torch.abs(y_pred - y_true)

    # Directional weight: exp(-alpha) for same sign, exp(+alpha) for opposite
    same_sign = (torch.sign(y_pred) == torch.sign(y_true)).float()
    opp_sign = 1.0 - same_sign
    w_pos = math.exp(-alpha)
    w_neg = math.exp(+alpha)

    weighted_errors = abs_errors * (same_sign * w_pos + opp_sign * w_neg)

    loss = weighted_errors.mean()
    return loss


__all__ = [
    "GMADLLoss",
    "compute",
]
