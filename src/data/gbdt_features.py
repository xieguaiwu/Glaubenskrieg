"""GBDT feature aggregation from temporal sequences.

Converts CTM's temporal sequences into tabular features suitable
for GBDT (Gradient Boosted Decision Tree) cross-sectional training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def aggregate_sequence_features(sequences: np.ndarray) -> np.ndarray:
    """Aggregate temporal sequences into tabular features.

    Applies 6 aggregation methods per original feature:
    last, mean, std, min, max, and linear slope (trend).

    Parameters
    ----------
    sequences : np.ndarray
        Shape (N, seq_len, D) — temporal feature sequences.

    Returns
    -------
    np.ndarray
        Shape (N, 6 * D) — aggregated features in the order:
        [last, mean, std, min, max, slope] for each original dimension.
    """
    N, T, D = sequences.shape

    last = sequences[:, -1, :]
    mean = np.nanmean(sequences, axis=1)
    std = np.nanstd(sequences, axis=1, ddof=1)
    min_ = np.nanmin(sequences, axis=1)
    max_ = np.nanmax(sequences, axis=1)
    slope = _compute_slope(sequences)

    return np.column_stack([last, mean, std, min_, max_, slope])


def _compute_slope(sequences: np.ndarray) -> np.ndarray:
    """Compute linear trend (slope) for each sequence and feature.

    Uses OLS: slope = Cov(x, y) / Var(x) where x is the time index.
    NaN values are masked out from both numerator and denominator.

    Parameters
    ----------
    sequences : np.ndarray
        Shape (N, T, D).

    Returns
    -------
    np.ndarray
        Shape (N, D) — slope coefficients. Zero when insufficient valid
        data points (<2) exist for a feature.
    """
    N, T, D = sequences.shape
    x = np.arange(T, dtype=sequences.dtype)
    x_centered = x - x.mean()

    y_mean = np.nanmean(sequences, axis=1, keepdims=True)
    y_centered = sequences - y_mean

    valid = ~np.isnan(sequences)
    x_bc = x_centered[np.newaxis, :, np.newaxis]

    numerator = np.where(valid, x_bc * y_centered, 0.0).sum(axis=1)
    denominator = np.where(valid, x_bc ** 2, 0.0).sum(axis=1)

    return np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def extract_ctm_hidden_features(
    hidden_states: np.ndarray,
    method: str = "last",
) -> np.ndarray:
    """Extract a fixed-size representation from CTM encoder hidden states.

    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (N, seq_len, d_model) — CTM encoder outputs at each timestep.
    method : str
        Pooling strategy:

        - ``"last"`` — take the final timestep only → (N, d_model)
        - ``"mean"`` — mean over the sequence → (N, d_model)
        - ``"both"`` — concatenate last and mean → (N, 2 * d_model)

    Returns
    -------
    np.ndarray
        Shape (N, d_model * method_factor).
    """
    if method == "last":
        return hidden_states[:, -1, :]
    if method == "mean":
        return np.nanmean(hidden_states, axis=1)
    if method == "both":
        last = hidden_states[:, -1, :]
        mean = np.nanmean(hidden_states, axis=1)
        return np.column_stack([last, mean])
    raise ValueError(
        f"Unknown CTM hidden method '{method}'. "
        f"Expected one of: 'last', 'mean', 'both'."
    )


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
    causal: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalise features, optionally reusing precomputed statistics.

    Parameters
    ----------
    features : np.ndarray
        Shape (N, D) — feature matrix to normalise.
    mean : np.ndarray or None
        Precomputed per-feature mean. Computed from features if None.
    std : np.ndarray or None
        Precomputed per-feature standard deviation. Computed from features if None.
    eps : float
        Small constant for numerical stability in division.
    causal : bool
        If True, compute expanding-window statistics instead of full-dataset
        statistics. This prevents future information leakage when used with
        walk-forward validation. Ignored when precomputed mean/std are provided.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (features_norm, mean, std) where features_norm has shape (N, D).
        When causal=True, returns the final expanding-window statistics.
    """
    if causal and mean is None and std is None:
        # Expanding-window normalization (causal — no future leakage)
        n = len(features)
        mean_arr = np.nancumsum(features, axis=0) / np.arange(1, n + 1)[:, np.newaxis]
        # Use expanding variance formula: Var = E[X²] - E[X]²
        sq_mean = np.nancumsum(features ** 2, axis=0) / np.arange(1, n + 1)[:, np.newaxis]
        var = sq_mean - mean_arr ** 2
        var = np.maximum(var, 1e-12)
        std_arr = np.sqrt(var)
        X_norm = (features - mean_arr) / (std_arr + eps)
        return X_norm, mean_arr[-1].copy(), std_arr[-1].copy()

    mean_arr: np.ndarray = np.nanmean(features, axis=0) if mean is None else mean
    std_arr: np.ndarray = np.nanstd(features, axis=0, ddof=1) if std is None else std
    std_safe = np.where(~np.isfinite(std_arr) | (std_arr < eps), 1.0, std_arr)
    X_norm = (features - mean_arr) / std_safe
    return X_norm, mean_arr, std_safe


@dataclass
class GBDTFeatureConfig:
    """Configuration for the GBDT feature engineering pipeline.

    Note: aggregation methods are always all 6 (last, mean, std, min, max, slope).
    ``seq_agg_methods`` was removed — the aggregation function is fixed.

    Attributes
    ----------
    ctm_hidden_method : str
        Pooling strategy for CTM hidden states.
    include_ctm_features : bool
        Whether to concatenate CTM hidden features into the matrix.
    normalize : bool
        Apply z-score normalisation after building the feature matrix.
    """
    ctm_hidden_method: str = "both"
    include_ctm_features: bool = False  # aligned with build_gbdt_feature_matrix default
    normalize: bool = True


def build_gbdt_feature_matrix(
    raw_sequences: np.ndarray,
    ctm_hidden: Optional[np.ndarray] = None,
    include_ctm_features: bool = False,
    config: Optional[GBDTFeatureConfig] = None,
) -> np.ndarray:
    """Build a flat GBDT-ready feature matrix from temporal sequences.

    Aggregates raw feature sequences into tabular form and optionally
    appends CTM hidden-state features.

    Parameters
    ----------
    raw_sequences : np.ndarray
        Shape (N, seq_len, D) — raw feature sequences (e.g. technical
        indicators).
    ctm_hidden : np.ndarray or None
        Shape (N, seq_len, d_model) — CTM encoder hidden states.
        Required when ``include_ctm_features`` is True.
    include_ctm_features : bool
        If True, concatenate CTM hidden features pooled via ``"both"``.
    config : GBDTFeatureConfig or None
        Optional configuration override. When provided, ``ctm_hidden_method``
        and ``normalize`` are applied instead of defaults.

    Returns
    -------
    np.ndarray
        Shape (N, total_features) ready for GBDT model input.
    """
    seq_features = aggregate_sequence_features(raw_sequences)

    if not include_ctm_features:
        result = seq_features
    else:
        if ctm_hidden is None:
            raise ValueError(
                "ctm_hidden must be provided when include_ctm_features=True"
            )
        method = config.ctm_hidden_method if config is not None else "both"
        ctm_features = extract_ctm_hidden_features(ctm_hidden, method=method)
        result = np.column_stack([seq_features, ctm_features])

    if config is not None and config.normalize:
        result, _, _ = normalize_features(result)

    return result
