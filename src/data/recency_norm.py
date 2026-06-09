"""Recency-aware walk-forward normalization.

Implements the method from Bai et al. (Symmetry 2026):
- For each walk-forward window, compute statistics from training data only
- Apply expanding-window or fit-latest-W standardization
- Strictly no look-ahead: test/val data never used for statistics
"""

import numpy as np


class RecencyAwareScaler:
    """Standardization with recency-aware statistics.

    Two modes:
    - "expanding": Use all training data from the window
    - "latest": Use only the most recent W samples (fit-latest-W)
    """

    def __init__(self, mode: str = "expanding", latest_w: int = 252):
        self.mode = mode
        self.latest_w = latest_w
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """Compute statistics from training data.

        Parameters
        ----------
        X : (N, D) array — training features
        """
        if self.mode == "latest" and X.shape[0] > self.latest_w:
            X = X[-self.latest_w:]
        self.mean_ = np.nanmean(X, axis=0, keepdims=True)
        self.std_ = np.nanstd(X, axis=0, keepdims=True)
        # Guard against NaN (all-NaN columns) and near-zero std
        self.std_[~np.isfinite(self.std_) | (self.std_ < 1e-8)] = 1.0
        self.mean_[~np.isfinite(self.mean_)] = 0.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization using training statistics.

        Parameters
        ----------
        X : (N, D) array

        Returns
        -------
        X_scaled : (N, D) array
        """
        if self.mean_ is None:
            raise ValueError("Must call fit() before transform()")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and transform X."""
        self.fit(X)
        return self.transform(X)


def walk_forward_normalize(
    features: np.ndarray,
    targets: np.ndarray,
    train_window: int,
    val_window: int,
    step_size: int,
    purge_period: int = 126,
    mode: str = "expanding",
    latest_w: int = 252,
) -> tuple:
    """Walk-forward normalization with strict no-look-ahead.

    For each window:
    1. Fit scaler on train data only
    2. Transform train, val, (optional) test using train statistics

    Parameters
    ----------
    features : (T, N, D) or (T, D) array
    targets : (T, N) or (T,) array
    train_window : int — number of sequences for training
    val_window : int — number of sequences for validation
    step_size : int — walk-forward step
    purge_period : int — gap between train and val
    mode : str — "expanding" or "latest"
    latest_w : int — window size for "latest" mode

    Yields
    ------
    (train_X, train_y, val_X, val_y, window_start) per window
    """
    T = len(features)
    pos = 0

    while pos + train_window + purge_period + val_window <= T:
        train_end = pos + train_window
        val_start = train_end + purge_period
        val_end = val_start + val_window

        train_X = features[pos:train_end]
        train_y = targets[pos:train_end]
        val_X = features[val_start:val_end]
        val_y = targets[val_start:val_end]

        # Flatten for scaler if multi-asset
        if train_X.ndim == 4:  # (B, N, T, D)
            B_tr, N, T_seq, D = train_X.shape
            train_X_flat = train_X.reshape(B_tr * N * T_seq, D)
            val_X_flat = val_X.reshape(val_X.shape[0] * N * T_seq, D)
        elif train_X.ndim == 3:  # (B, T, D)
            B_tr, T_seq, D = train_X.shape
            train_X_flat = train_X.reshape(B_tr * T_seq, D)
            val_X_flat = val_X.reshape(val_X.shape[0] * T_seq, D)
        else:
            train_X_flat = train_X
            val_X_flat = val_X

        scaler = RecencyAwareScaler(mode=mode, latest_w=latest_w)
        scaler.fit(train_X_flat)

        train_X_norm = scaler.transform(train_X_flat).reshape(train_X.shape)
        val_X_norm = scaler.transform(val_X_flat).reshape(val_X.shape)

        yield train_X_norm, train_y, val_X_norm, val_y, pos

        pos += step_size
