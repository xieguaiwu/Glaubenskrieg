"""Tests for data pipeline (dataset, features, gbdt_features)."""

import numpy as np
import torch
import pytest

from src.data.dataset import create_sequences, train_val_test_split
from src.data.features import compute_returns, compute_rsi, compute_sma, compute_bollinger_bands
from src.data.gbdt_features import (
    aggregate_sequence_features,
    extract_ctm_hidden_features,
    normalize_features,
    build_gbdt_feature_matrix,
)


def test_create_sequences_shape():
    data = np.random.randn(100, 5)
    seqs = create_sequences(data, seq_len=10)
    assert seqs.shape == (90, 10, 5), f"Expected (90,10,5), got {seqs.shape}"


def test_create_sequences_too_short():
    data = np.random.randn(5, 3)
    seqs = create_sequences(data, seq_len=10)
    assert seqs.shape == (0, 10, 3), f"Expected (0,10,3), got {seqs.shape}"


def test_train_val_test_split_proportions():
    data = np.random.randn(100, 5)
    train, val, test = train_val_test_split(data, 0.7, 0.15)
    assert len(train) == 70, f"Expected 70 train, got {len(train)}"
    assert len(val) == 15, f"Expected 15 val, got {len(val)}"
    assert len(test) == 15, f"Expected 15 test, got {len(test)}"


def test_train_val_test_split_too_small():
    data = np.random.randn(1, 5)
    with pytest.raises(ValueError, match="Need at least 2 samples"):
        train_val_test_split(data, 0.7, 0.15)


def test_rolling_normalize_no_leakage():
    from src.data.dataset import StockDataset
    import pandas as pd

    N = 50
    df = pd.DataFrame(
        {"close": np.random.randn(N) + 100, "feat": np.random.randn(N)},
        index=pd.date_range("2020-01-01", periods=N, freq="D"),
    )
    ds = StockDataset(df, feature_cols=["close", "feat"], target_col="close", seq_len=10, normalize=True)
    features = ds.features.numpy()
    assert np.allclose(features[0], 0.0, atol=1e-6), "First timestep should be 0 (no prior data)"


def test_compute_returns_shape():
    prices = torch.tensor([100.0, 102.0, 105.0, 103.0])
    rets = compute_returns(prices)
    assert rets.shape == (4,), f"Expected (4,), got {rets.shape}"
    assert rets[0].item() == 0.0, "First return should be 0"


def test_compute_rsi_values():
    prices = torch.linspace(100, 200, 50) + torch.randn(50) * 2
    rsi = compute_rsi(prices, period=14)
    assert (rsi >= 0).all() and (rsi <= 100).all(), f"RSI should be in [0,100], got [{rsi.min()}, {rsi.max()}]"


def test_compute_sma_length():
    prices = torch.randn(50)
    sma = compute_sma(prices, period=10)
    assert sma.shape == (50,), f"Expected (50,), got {sma.shape}"


def test_compute_bollinger_position():
    period = 20
    prices = torch.randn(100).cumsum(0) + 100
    sma, upper, lower = compute_bollinger_bands(prices, period=period)
    valid = slice(period - 1, None)
    assert (upper[valid] >= sma[valid]).all(), "Upper band should be >= SMA"
    assert (lower[valid] <= sma[valid]).all(), "Lower band should be <= SMA"
    assert upper[valid][0] >= sma[valid][0] >= lower[valid][0]


def test_aggregate_sequence_features_shape():
    seqs = np.random.randn(15, 10, 4)
    agg = aggregate_sequence_features(seqs)
    assert agg.shape == (15, 24), f"Expected (15,24), got {agg.shape}"


def test_aggregate_sequence_features_last_equals_slice():
    seqs = np.random.randn(8, 12, 3)
    agg = aggregate_sequence_features(seqs)
    last_block = agg[:, 0:3]
    np.testing.assert_allclose(last_block, seqs[:, -1, :], atol=1e-6)


def test_extract_ctm_hidden_features_methods():
    hidden = np.random.randn(5, 10, 32)
    last = extract_ctm_hidden_features(hidden, method="last")
    assert last.shape == (5, 32), f"Expected (5,32), got {last.shape}"
    mean = extract_ctm_hidden_features(hidden, method="mean")
    assert mean.shape == (5, 32), f"Expected (5,32), got {mean.shape}"
    both = extract_ctm_hidden_features(hidden, method="both")
    assert both.shape == (5, 64), f"Expected (5,64), got {both.shape}"


def test_build_gbdt_feature_matrix_no_ctm():
    seqs = np.random.randn(10, 20, 5)
    X = build_gbdt_feature_matrix(seqs, include_ctm_features=False)
    assert X.shape == (10, 30), f"Expected (10,30), got {X.shape}"


def test_normalize_features():
    X = np.random.randn(100, 5) * 3.0 + 10.0
    X_norm, mean, std = normalize_features(X)
    assert np.allclose(np.mean(X_norm, axis=0), 0.0, atol=1e-10), "Normalized mean should be ~0"
    assert np.allclose(np.std(X_norm, axis=0, ddof=1), 1.0, atol=1e-10), "Normalized std should be ~1"
