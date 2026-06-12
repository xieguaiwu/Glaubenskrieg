"""Data pipeline utilities for GBDT quant research."""

import numpy as np
import torch
from typing import Tuple, Optional, List, Dict, Callable, Union, Any


def make_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 10,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data for testing.

    The target is constructed as a non-linear function of three informative
    features plus Gaussian noise, mimicking a realistic quant factor.

        y = sin(f0) + 0.5 * cos(f1) + 0.3 * f2² + ε

    where ε ~ N(0, noise).

    Args:
        n_samples: Number of samples.
        n_features: Number of features (only first 3 are signal, rest are noise).
        noise: Standard deviation of additive Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        (X, y) where:
            X — shape [n_samples, n_features], drawn from N(0, 1).
            y — shape [n_samples], non-linear function of first 3 columns + noise.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    # Non-linear target using the first 3 features
    f0 = X[:, 0]
    f1 = X[:, 1]
    f2 = X[:, 2]
    y = np.sin(f0) + 0.5 * np.cos(f1) + 0.3 * f2 ** 2
    y += noise * rng.standard_normal(n_samples)
    y = y.astype(np.float32)

    return X, y


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split (X, y) into training and validation sets.

    Args:
        X: Feature array, shape [N, D].
        y: Target array, shape [N].
        val_ratio: Fraction of samples for validation (0 < val_ratio < 1).
        random_seed: Seed for reproducible shuffling.  If None, no seed.

    Returns:
        (X_train, X_val, y_train, y_val)
    """
    N = X.shape[0]
    indices = np.arange(N)

    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)

    split = int(N * (1.0 - val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Information Coefficient — Spearman rank correlation.

    Measures the monotonic relationship between predictions and true values.
    In quant finance, IC is the standard metric for evaluating factor quality.

    Args:
        y_true: Ground truth values, shape [N].
        y_pred: Predicted values, shape [N].

    Returns:
        Spearman rank correlation coefficient in [-1, 1].
        1  → perfect positive rank correlation.
        -1 → perfect negative rank correlation.
        0  → no rank correlation.
    """
    from scipy.stats import spearmanr  # type: ignore[import-untyped]
    correlation, _ = spearmanr(y_true, y_pred)
    return float(correlation)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────
#  Purged Walk-Forward Cross-Validation
# ─────────────────────────────────────────────────────────────────


def purged_walk_forward_split(
    n_samples: int,
    n_folds: int,
    train_size: float = 0.6,
    gap_size: int = 0,
    sliding: bool = False,
    return_indices: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged walk-forward train/test index pairs.

    Financial time series cannot use standard K-fold CV because of
    temporal leakage.  This function implements the Purged Walk-Forward
    CV from *Advances in Financial ML* (López de Prado, Chapter 12).

    Each fold splits data into three contiguous regions::

        | TRAIN |  GAP  | TEST |
                        | TRAIN |  GAP  | TEST |
                                        | TRAIN |  GAP  | TEST |

    Where:
      - The **gap** between train and test prevents label overlap leakage.
      - The **test** set always comes *after* training (no lookahead).
      - ``sliding=False`` → growing window (train expands at each fold).
      - ``sliding=True``  → fixed-size sliding window.

    Args:
        n_samples: Total number of samples (time steps).
        n_folds: Number of train/test splits.
        train_size: Fraction of total data for the **first** training
            window.  Other windows expand (or slide) from this base.
        gap_size: Number of samples to skip between train and test
            (prevents label leakage).  0 means no gap.
        sliding: If True, all training windows have the same fixed size.
            If False (default), training window expands (growing window,
            uses all earlier data).
        return_indices: If True (default), returns ``(train_idx, test_idx)``
            arrays.  If False, returns ``(train_slice, test_slice)`` as
            ``(start, end)`` tuples.

    Returns:
        List of ``(train_indices, test_indices)`` tuples for each fold.
        Each element is either:
        - ``(np.ndarray, np.ndarray)`` if ``return_indices=True``
        - ``((int, int), (int, int))`` if ``return_indices=False``

    Example::

        >>> purged_walk_forward_split(100, 3, 0.6, gap=5)
        Fold 0: train=[0,59]  gap=[60,64]  test=[65,~79]
        Fold 1: train=[0,74]  gap=[75,79]  test=[80,~89]   (growing)
        Fold 2: train=[0,84]  gap=[85,89]  test=[90,~99]   (growing)

    With ``sliding=True``, each train window is fixed size::

        Fold 0: train=[0,59]   gap=[60,64]  test=[65,~]
        Fold 1: train=[13,72]  gap=[73,77]  test=[78,~]    (sliding)
        Fold 2: train=[27,86]  gap=[87,91]  test=[92,~]    (sliding)
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if not (0.0 < train_size < 1.0):
        raise ValueError(f"train_size must be in (0, 1), got {train_size}")
    if gap_size < 0:
        raise ValueError(f"gap_size must be >= 0, got {gap_size}")

    first_train = int(train_size * n_samples)
    if first_train < 1:
        raise ValueError(
            f"train_size={train_size} yields 0 training samples "
            f"for n_samples={n_samples}"
        )

    remaining = n_samples - first_train
    total_gap = gap_size * n_folds

    if remaining < total_gap + n_folds:
        raise ValueError(
            f"Not enough remaining samples after first train ({remaining}) "
            f"to accommodate {n_folds} folds with gap={gap_size}. "
            f"Reduce n_folds, gap_size, or train_size."
        )

    total_test_allocation = remaining - total_gap

    # Distribute test sizes as evenly as possible (later folds may get more)
    base_test = total_test_allocation // n_folds
    remainder = total_test_allocation % n_folds
    test_sizes = [
        base_test + (1 if (n_folds - 1 - i) < remainder else 0)
        for i in range(n_folds)
    ]

    folds: List[Any] = []
    current_pos = first_train

    first_train_size = first_train  # for sliding window

    for i in range(n_folds):
        test_start = current_pos + gap_size
        test_end = test_start + test_sizes[i]

        if sliding:
            train_start = max(0, current_pos - first_train_size)
        else:
            train_start = 0
        train_end = current_pos

        if return_indices:
            folds.append((
                np.arange(train_start, train_end),
                np.arange(test_start, min(test_end, n_samples)),
            ))
        else:
            folds.append((
                (train_start, train_end),
                (test_start, min(test_end, n_samples)),
            ))

        current_pos = test_end

    return folds


def purged_walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_train_fn: Callable,
    n_folds: int = 5,
    train_size: float = 0.6,
    gap_size: int = 0,
    sliding: bool = False,
    model_predict_fn: Optional[Callable] = None,
    metric_fn: Optional[Callable] = None,
    verbose: bool = True,
) -> Dict:
    """Run purged walk-forward cross-validation.

    Trains a model on each fold's training window and evaluates it on
    the corresponding test window.  Aggregates predictions across folds
    for IC / metric calculation.

    Args:
        X: Feature array.  Supports two shapes:
            - **2D flat**: ``[N_time, D]`` — time-ordered flat data.
            - **3D panel**: ``[T, N_assets, D]`` — panel data; splitting
              is performed along the time axis ``T``, preserving the
              cross-section structure ``N_assets`` within each fold.
        y: Target array.  Shape ``[N_time]`` or ``[T, N_assets]``
            (matches the first dimension of *X*).
        model_train_fn: Callable ``(X_train, y_train) -> fitted_model``.
        n_folds: Number of CV folds.
        train_size: Fraction of samples in the first training window.
        gap_size: Purging gap between train and test sets.
        sliding: If True, fixed-size training windows.  If False, growing.
        model_predict_fn: Optional ``(model, X_test) -> predictions``.
            If None, uses ``model.predict(X_test)``.
        metric_fn: Optional ``(y_true, y_pred) -> float``.
            If None, defaults to Spearman rank correlation (IC).
        verbose: If True, print fold progress.

    Returns:
        dict with keys:
        - ``'fold_results'``: list of dicts, one per fold, containing
          ``{'fold': int, 'train_idx': slice, 'test_idx': slice,
          'train_rows': int, 'test_rows': int, 'pred': np.ndarray,
          'metric': float}``
        - ``'overall_metric'``: float, mean metric across folds.
        - ``'all_preds'``: np.ndarray, concatenated test predictions
          (in original time order).
        - ``'all_true'``: np.ndarray, concatenated test targets
          (in original time order).
        - ``'cv_config'``: dict of configuration parameters.
    """
    if metric_fn is None:
        from scipy.stats import spearmanr  # type: ignore[import-untyped]

        def _default_metric(yt: np.ndarray, yp: np.ndarray) -> float:
            rho, _ = spearmanr(yt, yp)
            return float(rho)  # type: ignore[arg-type]

        metric_fn = _default_metric

    # Determine the time dimension
    if X.ndim == 2:
        # Flat data: [T, D]
        n_time = X.shape[0]
        is_panel = False
    elif X.ndim == 3:
        # Panel data: [T, N, D]
        n_time = X.shape[0]
        is_panel = True
    else:
        raise ValueError(f"X must be 2D [T, D] or 3D [T, N, D], got shape {X.shape}")

    # Generate fold splits (slices for convenience)
    fold_splits = purged_walk_forward_split(
        n_samples=n_time,
        n_folds=n_folds,
        train_size=train_size,
        gap_size=gap_size,
        sliding=sliding,
        return_indices=False,
    )

    fold_results: List[Dict] = []
    all_preds_list: List[np.ndarray] = []
    all_true_list: List[np.ndarray] = []
    metrics = []

    for i, ((train_s, train_e), (test_s, test_e)) in enumerate(fold_splits):
        if is_panel:
            X_train = X[train_s:train_e]  # [t, N, D]
            X_test = X[test_s:test_e]
            y_train = y[train_s:train_e]  # [t, N]
            y_test = y[test_s:test_e]
        else:
            X_train = X[train_s:train_e]  # [t, D]
            X_test = X[test_s:test_e]
            y_train = y[train_s:train_e]  # [t]
            y_test = y[test_s:test_e]

        train_rows = train_e - train_s
        test_rows = test_e - test_s

        if verbose:
            print(
                f"Fold {i + 1}/{n_folds}: "
                f"train=[{train_s}:{train_e}] ({train_rows}), "
                f"test=[{test_s}:{test_e}] ({test_rows})"
            )

        model = model_train_fn(X_train, y_train)

        if model_predict_fn is not None:
            y_pred = np.asarray(model_predict_fn(model, X_test), dtype=np.float64)
        else:
            y_pred = np.asarray(model.predict(X_test), dtype=np.float64)

        y_true_flat = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred_flat = y_pred.ravel()

        fold_metric = metric_fn(y_true_flat, y_pred_flat)
        metrics.append(fold_metric)

        fold_results.append({
            "fold": i,
            "train_idx": (train_s, train_e),
            "test_idx": (test_s, test_e),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "pred": y_pred,
            "metric": fold_metric,
        })

        all_preds_list.append(y_pred_flat)
        all_true_list.append(y_true_flat)

        if verbose:
            print(f"  metric = {fold_metric:.6f}")

    all_preds = np.concatenate(all_preds_list) if all_preds_list else np.array([])
    all_true = np.concatenate(all_true_list) if all_true_list else np.array([])
    overall_metric = float(np.mean(metrics)) if metrics else float("nan")

    return {
        "fold_results": fold_results,
        "overall_metric": overall_metric,
        "all_preds": all_preds,
        "all_true": all_true,
        "cv_config": {
            "n_folds": n_folds,
            "train_size": train_size,
            "gap_size": gap_size,
            "sliding": sliding,
            "n_samples": n_time,
            "is_panel": is_panel,
        },
    }
