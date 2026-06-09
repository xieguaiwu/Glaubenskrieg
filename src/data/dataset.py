"""Stock dataset pipeline with causal normalization and temporal splits."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """PyTorch dataset for stock price sequences.

    Constructs sliding windows of length seq_len with chronological ordering.
    Normalization uses trailing rolling z-score (no look-ahead bias).

    Parameters
    ----------
    prices_df : pd.DataFrame
        Must contain a 'close' column and feature columns.
    feature_cols : list of str
        Column names to use as input features.
    target_col : str
        Column name for the target (e.g., 'next_return').
    seq_len : int
        Length of each input sequence.
    normalize : bool, default=True
        Apply rolling z-score normalization (trailing window only).
    """
    def __init__(
        self,
        prices_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        seq_len: int,
        normalize: bool = True,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if len(prices_df) < seq_len + 1:
            raise ValueError(
                f"DataFrame has {len(prices_df)} rows, need at least {seq_len + 1} "
                f"for seq_len={seq_len}"
            )
        self.seq_len = seq_len
        self.feature_cols = list(feature_cols)
        self.target_col = target_col

        missing = [c for c in self.feature_cols if c not in prices_df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in prices_df: {missing}")

        features = prices_df[self.feature_cols].values.astype(np.float32)
        targets = prices_df[[target_col]].values.astype(np.float32)

        if normalize:
            features = self._rolling_normalize(features)

        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    @staticmethod
    def _rolling_normalize(data: np.ndarray, window: int = 252) -> np.ndarray:
        """Apply trailing rolling z-score normalization (no look-ahead)."""
        N, D = data.shape
        out = np.zeros_like(data)
        cumsum = np.cumsum(data, axis=0)
        cumsum_sq = np.cumsum(data ** 2, axis=0)

        # Expanding window (t < window): expanding mean/std from t=1 onward
        exp_end = min(window, N)
        if exp_end > 1:
            counts = np.arange(2, exp_end + 1, dtype=data.dtype).reshape(-1, 1)
            mu_exp = cumsum[1:exp_end] / counts
            var_exp = cumsum_sq[1:exp_end] / counts - mu_exp ** 2
            var_exp = np.maximum(var_exp, 0)
            out[1:exp_end] = (data[1:exp_end] - mu_exp) / (np.sqrt(var_exp) + 1e-12)

        # Rolling window (t >= window): fixed-length trailing window
        if N > window:
            sum_roll = cumsum[window:] - cumsum[:N - window]
            sq_roll = cumsum_sq[window:] - cumsum_sq[:N - window]
            mu_roll = sum_roll / window
            var_roll = sq_roll / window - mu_roll ** 2
            var_roll = np.maximum(var_roll, 0)
            out[window:] = (data[window:] - mu_roll) / (np.sqrt(var_roll) + 1e-12)

        # t=0 remains zero (no prior data)
        return out

    def __len__(self) -> int:
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x_seq, y_target) where x_seq is (seq_len, n_features)
        and y_target is the next-step target, shape (1,)."""
        # Prohibit negative indices to prevent temporal look-ahead in financial time series data
        if idx < 0:
            raise IndexError(f"Negative index not allowed: {idx}")
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, y


def create_sequences(
    data: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    """Create sliding window sequences of length seq_len from (N, D) data.

    Returns (N-seq_len, seq_len, D) array.
    """
    N = len(data)
    n_seq = max(0, N - seq_len)
    if n_seq == 0:
        return np.empty((0, seq_len, data.shape[-1]), dtype=data.dtype)

    shape = (n_seq, seq_len) + data.shape[1:]
    strides = (data.strides[0],) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides).copy()


def train_val_test_split(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temporal (chronological) split respecting time order.

    Returns (train, val, test) arrays.
    """
    N = len(data)
    if N < 2:
        raise ValueError(f"Need at least 2 samples, got {N}")
    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]
