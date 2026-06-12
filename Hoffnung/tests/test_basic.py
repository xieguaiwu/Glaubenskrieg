"""Basic tests for the GBDT Python package.

These tests require the compiled C++ module ``gbdt_python``.
If the module is not available, the tests are skipped with a clear message.

Run with::

    python -m pytest tests/test_basic.py -v
    # or
    python tests/test_basic.py
"""

import unittest
import numpy as np


# ──────────────────────────────────────────────
#  Attempt C++ module import
# ──────────────────────────────────────────────

import torch

# Python-side utilities
from gbdt.pipeline import make_synthetic_data, compute_ic, purged_walk_forward_split, purged_walk_forward_cv
from gbdt.losses import (
    mse_loss,
    mae_loss,
    huber_loss,
    quantile_loss,
    rankic_loss,
    compute_gradients,
)

# C++ module import (lazy — checked inside integration test class)
_CPP_MODULE_AVAILABLE: bool = False
_CPP_IMPORT_ERROR: str = ""
try:
    from gbdt_python import GBDT, GBDTConfig  # noqa: F401
    _CPP_MODULE_AVAILABLE = True
except ImportError as e:
    _CPP_IMPORT_ERROR = str(e)


def _import_cpp():
    """Import C++ module or raise SkipTest with build instructions."""
    if not _CPP_MODULE_AVAILABLE:
        raise unittest.SkipTest(
            f"C++ module 'gbdt_python' not available: {_CPP_IMPORT_ERROR}\n"
            "Build it first with:\n"
            "    cd build && cmake .. && make"
        )
    from gbdt_python import GBDT, GBDTConfig  # noqa: F811
    return GBDT, GBDTConfig


# ──────────────────────────────────────────────
#  Loss function tests  (don't need C++ module)
# ──────────────────────────────────────────────

class TestLossFunctions(unittest.TestCase):
    """Verify loss functions return correct shapes and basic properties."""

    def setUp(self):
        torch.manual_seed(42)
        self.N = 16
        self.y_true = torch.randn(self.N)
        self.y_pred = torch.randn(self.N)

    def test_mse_loss_shape(self):
        loss, grad, hess = mse_loss(self.y_true, self.y_pred)
        self.assertEqual(loss.ndim, 0, "MSE loss should be scalar")
        self.assertEqual(grad.shape, (self.N,))
        self.assertEqual(hess.shape, (self.N,))

    def test_mse_loss_value(self):
        loss, _, _ = mse_loss(self.y_true, self.y_pred)
        expected = torch.nn.functional.mse_loss(self.y_pred, self.y_true)
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)

    def test_mse_loss_gradient_agrees_with_autograd(self):
        yp = self.y_pred.detach().requires_grad_(True)
        # Sum-based MSE: autograd of sum matches raw gradients (no /N)
        loss_val = torch.nn.functional.mse_loss(yp, self.y_true, reduction="sum")
        auto_grad = torch.autograd.grad(loss_val, yp)[0]
        _, grad, _ = mse_loss(self.y_true, self.y_pred)
        torch.testing.assert_close(grad, auto_grad, rtol=1e-5, atol=1e-6)

    def test_mse_loss_hessian_positive(self):
        _, _, hess = mse_loss(self.y_true, self.y_pred)
        self.assertTrue((hess >= 0).all(), "MSE Hessians should be non-negative")

    def test_mae_loss_shape(self):
        loss, grad, hess = mae_loss(self.y_true, self.y_pred)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(grad.shape, (self.N,))
        self.assertEqual(hess.shape, (self.N,))

    def test_mae_loss_gradient_sign(self):
        _, grad, _ = mae_loss(self.y_true, self.y_pred)
        # Gradient should be sign(y_pred - y_true) (raw, no /N)
        expected = torch.sign(self.y_pred - self.y_true)
        torch.testing.assert_close(grad, expected, rtol=1e-5, atol=1e-6)

    def test_mae_loss_hessian_nonzero(self):
        _, _, hess = mae_loss(self.y_true, self.y_pred, eps=1e-8)
        self.assertTrue((hess > 0).all(), "MAE Hessians should be > 0 (eps)")

    def test_huber_loss_mse_region(self):
        # When residual is small, Huber per-sample = 0.5 * MSE per-sample
        y_true = torch.zeros(self.N)
        y_pred = 0.1 * torch.randn(self.N)
        loss_h, grad_h, hess_h = huber_loss(y_true, y_pred, delta=1.0)
        loss_m, grad_m, hess_m = mse_loss(y_true, y_pred)
        self.assertAlmostEqual(2.0 * loss_h.item(), loss_m.item(), places=5)

    def test_huber_loss_shape(self):
        loss, grad, hess = huber_loss(self.y_true, self.y_pred)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(grad.shape, (self.N,))
        self.assertEqual(hess.shape, (self.N,))

    def test_rankic_loss_shape(self):
        loss, grad, hess = rankic_loss(self.y_true, self.y_pred)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(grad.shape, (self.N,))
        self.assertEqual(hess.shape, (self.N,))

    def test_rankic_loss_range(self):
        # Perfect positive correlation → loss ≈ 0
        loss, _, _ = rankic_loss(self.y_true, self.y_true)
        self.assertLess(loss.item(), 0.1, "RankIC for identical vectors should be near 0")

    def test_rankic_loss_negative_corr(self):
        # Perfect negative correlation → loss ≈ 2
        loss, _, _ = rankic_loss(self.y_true, -self.y_true)
        self.assertGreater(loss.item(), 1.0, "RankIC for opposite vectors should be > 1")

    def test_compute_gradients_mse(self):
        loss, grad, hess = compute_gradients(
            self.y_true, self.y_pred,
            lambda yt, yp: torch.nn.functional.mse_loss(yp, yt)
        )
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(grad.shape, (self.N,))
        self.assertEqual(hess.shape, (self.N,))
        self.assertTrue((hess >= 0).all())

    def test_losses_return_tensors(self):
        for fn in [mse_loss, mae_loss]:
            with self.subTest(fn=fn.__name__):
                loss, grad, hess = fn(self.y_true, self.y_pred)
                self.assertIsInstance(loss, torch.Tensor)
                self.assertIsInstance(grad, torch.Tensor)
                self.assertIsInstance(hess, torch.Tensor)

    def test_huber_delta_transition(self):
        # Large residual → Huber should be more robust than MSE
        y_true = torch.zeros(self.N)
        y_pred = torch.full((self.N,), 10.0)  # huge residual
        loss_h, grad_h, _ = huber_loss(y_true, y_pred, delta=1.0)
        loss_m, grad_m, _ = mse_loss(y_true, y_pred)
        # Huber gradient magnitude should be capped at delta (raw, no /N)
        self.assertTrue(
            (torch.abs(grad_h) <= 1.0 + 1e-6).all(),
            "Huber gradient should be capped at delta in linear region",
        )
        self.assertLess(loss_h, loss_m, "Huber should be lower than MSE for large residuals")


# ──────────────────────────────────────────────
#  Pipeline tests  (don't need C++ module)
# ──────────────────────────────────────────────

class TestPipeline(unittest.TestCase):
    """Verify pipeline utilities work correctly."""

    def test_make_synthetic_data_shape(self):
        X, y = make_synthetic_data(1000, 5)
        self.assertEqual(X.shape, (1000, 5))
        self.assertEqual(y.shape, (1000,))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)

    def test_make_synthetic_data_deterministic(self):
        X1, y1 = make_synthetic_data(100, 3, seed=42)
        X2, y2 = make_synthetic_data(100, 3, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_compute_ic_perfect(self):
        y_true = np.arange(100, dtype=np.float32)
        rho = compute_ic(y_true, y_true)
        self.assertAlmostEqual(rho, 1.0, places=10)

    def test_compute_ic_negative(self):
        y_true = np.arange(100, dtype=np.float32)
        rho = compute_ic(y_true, -y_true)
        self.assertAlmostEqual(rho, -1.0, places=10)

    def test_compute_ic_noise(self):
        rng = np.random.default_rng(42)
        y_true = np.arange(100, dtype=np.float32)
        y_pred = y_true + 0.5 * rng.standard_normal(100).astype(np.float32)
        rho = compute_ic(y_true, y_pred)
        self.assertGreater(rho, 0.5, "IC should be > 0.5 for noisy but correlated data")
        self.assertLessEqual(rho, 1.0)

    def test_purged_walk_forward_split_basic(self):
        """Test basic growing window split."""
        folds = purged_walk_forward_split(100, 3, 0.6, 0)
        self.assertEqual(len(folds), 3)
        # First train window ≈ train_size * n_samples (float precision varies)
        first_train = len(folds[0][0])
        self.assertGreaterEqual(first_train, 58)
        self.assertLessEqual(first_train, 61)
        # Test window is evenly distributed remainder
        self.assertEqual(len(folds[0][1]), 13)

    def test_purged_walk_forward_split_with_gap(self):
        """Test gap between train and test."""
        folds = purged_walk_forward_split(100, 3, 0.6, 5)
        # Check gap exists between train end and test start
        for train, test in folds:
            self.assertGreater(test[0], train[-1] + 4)

    def test_purged_walk_forward_split_sliding(self):
        """Test sliding window - same size across folds."""
        folds = purged_walk_forward_split(100, 3, 0.6, 0, sliding=True)
        sizes = [len(train) for train, _ in folds]
        self.assertEqual(len(set(sizes)), 1)

    def test_purged_walk_forward_split_no_overlap(self):
        """Test that train and test indices do not overlap."""
        folds = purged_walk_forward_split(200, 5, 0.5, 3)
        for train, test in folds:
            overlap = np.intersect1d(train, test)
            self.assertEqual(len(overlap), 0, "Train and test must not overlap")

    def test_purged_walk_forward_split_slice_mode(self):
        """Test return_indices=False returns slice tuples."""
        folds = purged_walk_forward_split(100, 3, 0.6, 0, return_indices=False)
        self.assertEqual(len(folds), 3)
        for train, test in folds:
            self.assertIsInstance(train, tuple)
            self.assertIsInstance(test, tuple)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(test), 2)

    def test_purged_walk_forward_cv_flat(self):
        """Test CV on 2D flat data."""
        X, y = make_synthetic_data(300, 3, noise=0.1, seed=42)

        def train_fn(X_tr, y_tr):
            class DummyModel:
                def predict(self, X_te):
                    return np.ones(X_te.shape[0], dtype=np.float64)
            return DummyModel()

        result = purged_walk_forward_cv(
            X, y, train_fn, n_folds=3, train_size=0.6, gap_size=0, verbose=False,
        )
        self.assertIn("fold_results", result)
        self.assertIn("overall_metric", result)
        self.assertIn("all_preds", result)
        self.assertIn("all_true", result)
        self.assertEqual(len(result["fold_results"]), 3)

    def test_purged_walk_forward_cv_panel(self):
        """Test CV on 3D panel data."""
        # Panel: [T=50, N=10, D=3]
        X = np.random.default_rng(42).standard_normal((50, 10, 3)).astype(np.float32)
        y = np.random.default_rng(43).standard_normal((50, 10)).astype(np.float32)

        def train_fn(X_tr, y_tr):
            class DummyModel:
                def predict(self, X_te):
                    return np.zeros(X_te.shape[0] * X_te.shape[1], dtype=np.float64)
            return DummyModel()

        result = purged_walk_forward_cv(
            X, y, train_fn, n_folds=3, train_size=0.5, gap_size=1, verbose=False,
        )
        self.assertEqual(len(result["fold_results"]), 3)
        self.assertTrue(result["cv_config"]["is_panel"])


# ──────────────────────────────────────────────
#  Integration tests  (require C++ module)
# ──────────────────────────────────────────────

@unittest.skipIf(not _CPP_MODULE_AVAILABLE, "C++ gbdt_python module not compiled")
class TestGBDTIntegration(unittest.TestCase):
    """End-to-end GBDT training and inference (requires ``gbdt_python`` .so)."""

    @classmethod
    def setUpClass(cls):
        cls._GBDT, cls._GBDTConfig = _import_cpp()
        cls.X, cls.y = make_synthetic_data(1000, 5, noise=0.1, seed=42)

    def test_config_defaults(self):
        GBDTConfig = self._GBDTConfig
        config = GBDTConfig()
        self.assertEqual(config.num_trees, 100)
        self.assertEqual(config.max_depth, 6)
        self.assertAlmostEqual(config.learning_rate, 0.1, places=5)
        self.assertEqual(config.loss_type, "mse")

    def test_config_custom_values(self):
        GBDTConfig = self._GBDTConfig
        config = GBDTConfig()
        config.num_trees = 50
        config.max_depth = 4
        config.learning_rate = 0.05
        config.min_samples_leaf = 20
        config.lambda_l2 = 0.5
        config.loss_type = "huber"
        config.num_threads = 2
        config.random_seed = 123

        self.assertEqual(config.num_trees, 50)
        self.assertEqual(config.max_depth, 4)
        self.assertAlmostEqual(config.learning_rate, 0.05, places=5)
        self.assertEqual(config.min_samples_leaf, 20)
        self.assertEqual(config.lambda_l2, 0.5)
        self.assertEqual(config.loss_type, "huber")
        self.assertEqual(config.num_threads, 2)
        self.assertEqual(config.random_seed, 123)

    def test_config_repr(self):
        GBDTConfig = self._GBDTConfig
        config = GBDTConfig()
        r = repr(config)
        self.assertIn("GBDTConfig", r)
        self.assertIn("num_trees", r)

    def test_model_create_and_fit(self):
        GBDT, GBDTConfig = self._GBDT, self._GBDTConfig
        config = GBDTConfig()
        config.num_trees = 10
        config.max_depth = 4

        model = GBDT(config)
        model.fit(self.X, self.y)

        self.assertGreater(model.num_trees(), 0)

    def test_model_predict_shape(self):
        GBDT, GBDTConfig = self._GBDT, self._GBDTConfig
        config = GBDTConfig()
        config.num_trees = 10
        config.max_depth = 4

        model = GBDT(config)
        model.fit(self.X, self.y)

        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (1000,))
        self.assertEqual(preds.dtype, np.float32)

    def test_model_predict_ic(self):
        GBDT, GBDTConfig = self._GBDT, self._GBDTConfig
        config = GBDTConfig()
        config.num_trees = 20
        config.max_depth = 5

        model = GBDT(config)
        model.fit(self.X, self.y)

        preds = model.predict(self.X)
        ic = compute_ic(self.y, preds)
        self.assertGreater(ic, 0.3, "IC should be positive for a fitted model")

    def test_model_with_validation_data(self):
        GBDT, GBDTConfig = self._GBDT, self._GBDTConfig
        config = GBDTConfig()
        config.num_trees = 10
        config.max_depth = 4
        config.early_stopping_rounds = 5

        split = int(0.8 * self.X.shape[0])
        X_tr, X_va = self.X[:split], self.X[split:]
        y_tr, y_va = self.y[:split], self.y[split:]

        model = GBDT(config)
        model.fit(X_tr, y_tr, X_va, y_va)

        preds = model.predict(X_va)
        ic = compute_ic(y_va, preds)
        self.assertGreater(ic, 0.2)

    def test_feature_importance(self):
        GBDT, GBDTConfig = self._GBDT, self._GBDTConfig
        config = GBDTConfig()
        config.num_trees = 10
        config.max_depth = 4

        model = GBDT(config)
        model.fit(self.X, self.y)

        imp = model.get_feature_importance(5)
        self.assertEqual(len(imp), 5)
        self.assertGreater(sum(imp), 0.0, "Feature importance split counts should be > 0")

    def test_model_serialization_json(self):
        GBDT, GBDTConfig = self._GBDT, self._GBDTConfig
        config = GBDTConfig()
        config.num_trees = 10
        config.max_depth = 4

        model = GBDT(config)
        model.fit(self.X, self.y)
        preds_before = model.predict(self.X)

        json_str = model.to_json()
        self.assertGreater(len(json_str), 0)
        self.assertIn("trees", json_str)

        model2 = GBDT(config)
        model2.from_json(json_str)
        preds_after = model2.predict(self.X)

        np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=5)


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
