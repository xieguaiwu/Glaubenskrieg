"""Tests for gbdt_features module (feature aggregation, normalization, config)."""

import numpy as np
import pytest

from src.data.gbdt_features import (
    GBDTFeatureConfig,
    _compute_slope,
    aggregate_sequence_features,
    build_gbdt_feature_matrix,
    extract_ctm_hidden_features,
    normalize_features,
)


class TestGBDTFeatureConfig:
    def test_defaults(self):
        cfg = GBDTFeatureConfig()
        assert cfg.ctm_hidden_method == "both"
        assert cfg.include_ctm_features is False  # aligned with build_gbdt_feature_matrix default
        assert cfg.normalize is True

    def test_custom_values(self):
        cfg = GBDTFeatureConfig(
            ctm_hidden_method="last",
            include_ctm_features=False,
            normalize=False,
        )
        assert cfg.ctm_hidden_method == "last"
        assert cfg.include_ctm_features is False
        assert cfg.normalize is False


class TestComputeSlope:
    def test_constant_sequence_returns_zero(self):
        seq = np.ones((3, 10, 2), dtype=np.float64)
        slope = _compute_slope(seq)
        expected = np.zeros((3, 2), dtype=np.float64)
        np.testing.assert_allclose(slope, expected, atol=1e-10)

    def test_linear_increasing(self):
        T = 10
        seq = np.linspace(0, 1, T).reshape(1, T, 1).repeat(2, axis=0)
        slope = _compute_slope(seq)
        assert (slope > 0).all(), "Slope should be positive for increasing sequence"

    def test_slope_shape(self):
        seq = np.random.RandomState(42).randn(5, 12, 4)
        slope = _compute_slope(seq)
        assert slope.shape == (5, 4)

    def test_slope_single_timestep(self):
        seq = np.random.RandomState(42).randn(3, 1, 2)
        slope = _compute_slope(seq)
        np.testing.assert_allclose(slope, 0.0, atol=1e-10)


class TestAggregateSequenceFeatures:
    def test_output_shape(self, random_sequences):
        agg = aggregate_sequence_features(random_sequences)
        N, _, D = random_sequences.shape
        assert agg.shape == (N, 6 * D)

    def test_last_block_matches_input_slice(self):
        rng = np.random.RandomState(42)
        seq = rng.randn(4, 8, 3)
        agg = aggregate_sequence_features(seq)
        last_block = agg[:, 0:3]
        np.testing.assert_allclose(last_block, seq[:, -1, :], atol=1e-10)

    def test_aggregate_constant_sequence(self):
        seq = np.ones((3, 10, 2), dtype=np.float64)
        agg = aggregate_sequence_features(seq)
        # last=1, mean=1, std=0, min=1, max=1, slope=0
        # Column layout: [last0, last1, mean0, mean1, std0, std1,
        #                 min0, min1, max0, max1, slope0, slope1]
        np.testing.assert_allclose(agg[:, 4:6], 0.0, atol=1e-10)


class TestExtractCTMHiddenFeatures:
    def test_last_method(self, random_hidden_states):
        result = extract_ctm_hidden_features(random_hidden_states, method="last")
        N, _, d = random_hidden_states.shape
        assert result.shape == (N, d)
        np.testing.assert_allclose(
            result, random_hidden_states[:, -1, :], atol=1e-10,
        )

    def test_mean_method(self, random_hidden_states):
        result = extract_ctm_hidden_features(random_hidden_states, method="mean")
        N, _, d = random_hidden_states.shape
        assert result.shape == (N, d)
        expected = np.nanmean(random_hidden_states, axis=1)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_both_method(self, random_hidden_states):
        result = extract_ctm_hidden_features(random_hidden_states, method="both")
        N, _, d = random_hidden_states.shape
        assert result.shape == (N, 2 * d)

    def test_unknown_method_raises(self):
        hidden = np.random.RandomState(42).randn(2, 5, 8)
        with pytest.raises(ValueError, match="Unknown CTM hidden method"):
            extract_ctm_hidden_features(hidden, method="invalid")


class TestNormalizeFeatures:
    def test_zero_mean_after_normalization(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5) * 3.0 + 10.0
        X_norm, mean, std = normalize_features(X)
        assert np.allclose(np.mean(X_norm, axis=0), 0.0, atol=1e-10)
        assert np.allclose(np.std(X_norm, axis=0, ddof=1), 1.0, atol=1e-10)

    def test_precomputed_stats_reuse(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 4) * 2.0 + 5.0
        X_test = rng.randn(30, 4) * 2.0 + 5.0
        _, mean, std = normalize_features(X_train)
        X_test_norm, mean2, std2 = normalize_features(X_test, mean=mean, std=std)
        np.testing.assert_allclose(mean, mean2)
        np.testing.assert_allclose(std, std2)

    def test_single_sample(self):
        X = np.array([[1.0, 2.0, 3.0]])
        X_norm, mean, std = normalize_features(X)
        assert X_norm.shape == (1, 3)
        assert np.allclose(mean, [1.0, 2.0, 3.0])
        assert np.allclose(X_norm, 0.0, atol=1e-10)

    def test_zero_std_column_handled(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]], dtype=np.float64)
        X_norm, mean, std = normalize_features(X)
        assert np.allclose(X_norm[:, 0], 0.0, atol=1e-10)
        assert not np.allclose(X_norm[:, 1], 0.0, atol=1e-10)


class TestBuildGBDTFeatureMatrix:
    def test_without_ctm_features(self, random_sequences):
        N, _, D = random_sequences.shape
        X = build_gbdt_feature_matrix(random_sequences, include_ctm_features=False)
        assert X.shape == (N, 6 * D)

    def test_with_ctm_features(self, random_sequences, random_hidden_states):
        N, _, D = random_sequences.shape
        _, _, d = random_hidden_states.shape
        X = build_gbdt_feature_matrix(
            random_sequences,
            ctm_hidden=random_hidden_states,
            include_ctm_features=True,
        )
        assert X.shape == (N, 6 * D + 2 * d)

    def test_missing_ctm_hidden_raises(self, random_sequences):
        with pytest.raises(ValueError, match="ctm_hidden must be provided"):
            build_gbdt_feature_matrix(
                random_sequences,
                ctm_hidden=None,
                include_ctm_features=True,
            )

    def test_include_false_ignores_ctm_hidden(self, random_sequences, random_hidden_states):
        N, _, D = random_sequences.shape
        X = build_gbdt_feature_matrix(
            random_sequences,
            ctm_hidden=random_hidden_states,
            include_ctm_features=False,
        )
        assert X.shape == (N, 6 * D)
