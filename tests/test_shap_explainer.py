"""Tests for TreeSHAPExplainer and helper functions (shap_explainer module)."""

import json

import numpy as np
import pytest

from src.utils.shap_explainer import (
    TreeSHAPExplainer,
    _is_leaf_node,
    _resolve_child_ids,
    _resolve_leaf_value,
)


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def synthetic_tree_json():
    """A simple 3-node tree: root → left leaf, root → right leaf (flat list format)."""
    return [
        {
            "nodeid": 0,
            "depth": 0,
            "split": 0,
            "split_condition": 0.5,
            "yes": 1,
            "no": 2,
            "missing": 1,
        },
        {"nodeid": 1, "depth": 1, "leaf": 0.3},
        {"nodeid": 2, "depth": 1, "leaf": -0.2},
    ]


class MockGBDTModel:
    """Mock GBDT model exposing _trees, _feature_names, to_json(), and __call__."""

    def __init__(self, n_features=5, n_trees=2):
        self._feature_names = [f"f{i}" for i in range(n_features)]
        self._n_features = n_features
        self._trees = [self._make_flat_tree() for _ in range(n_trees)]
        self.n_trees = n_trees

    @staticmethod
    def _make_flat_tree():
        return [
            {
                "nodeid": 0,
                "depth": 0,
                "split": 0,
                "split_condition": 0.0,
                "yes": 1,
                "no": 2,
                "missing": 1,
            },
            {"nodeid": 1, "depth": 1, "leaf": 0.1},
            {"nodeid": 2, "depth": 1, "leaf": -0.1},
        ]

    def __call__(self, X):
        return np.zeros(len(X))

    def to_json(self):
        return json.dumps({"trees": self._trees})


# ═══════════════════════════════════════════════════════════════════
# TreeSHAPExplainer tests
# ═══════════════════════════════════════════════════════════════════


class TestTreeSHAPExplainer:
    """Tests for TreeSHAPExplainer with mock GBDT model and synthetic trees."""

    def test_init(self):
        """Creates instance with mock model and num_features."""
        model = MockGBDTModel(n_features=5, n_trees=2)
        explainer = TreeSHAPExplainer(model, num_features=5)
        assert explainer.num_features == 5
        assert explainer.learning_rate == 0.1
        assert len(explainer._trees) == 2, f"Expected 2 trees, got {len(explainer._trees)}"

    def test_extract_trees_simple(self):
        """Parses 3-node flat tree correctly via to_json()."""
        model = MockGBDTModel(n_features=3, n_trees=1)
        explainer = TreeSHAPExplainer(model, num_features=3)
        assert len(explainer._trees) == 1
        tree_nodes = explainer._trees[0]
        assert len(tree_nodes) == 3, f"Expected 3 nodes (root + 2 leaves), got {len(tree_nodes)}"

    def test_get_feature_names(self):
        """Mock model exposes _feature_names list."""
        model = MockGBDTModel(n_features=4, n_trees=1)
        assert model._feature_names == ["f0", "f1", "f2", "f3"]

    def test_is_leaf_node(self):
        """_is_leaf_node correctly identifies leaf vs internal nodes."""
        leaf_node = {"nodeid": 1, "depth": 1, "leaf": 0.3}
        internal_node = {
            "nodeid": 0,
            "depth": 0,
            "split": 0,
            "split_condition": 0.5,
            "yes": 1,
            "no": 2,
            "missing": 1,
        }
        assert _is_leaf_node(leaf_node) is True
        assert _is_leaf_node(internal_node) is False

    def test_tree_shap_single_shape(self):
        """_tree_shap returns (N, D) for single-tree explainer."""
        model = MockGBDTModel(n_features=3, n_trees=1)
        explainer = TreeSHAPExplainer(model, num_features=3)
        X = np.array([[0.1, 0.2, 0.3], [-1.0, -2.0, -3.0]], dtype=np.float64)
        shap = explainer._tree_shap(explainer._trees[0], X)
        assert shap.shape == (2, 3), f"Expected (2, 3), got {shap.shape}"
        assert np.isfinite(shap).all()

    def test_explain_shape(self):
        """explain returns (N, D) array for multi-tree ensemble."""
        model = MockGBDTModel(n_features=5, n_trees=2)
        explainer = TreeSHAPExplainer(model, num_features=5)
        X = np.random.RandomState(42).randn(10, 5).astype(np.float64)
        shap = explainer.explain(X)
        assert shap.shape == (10, 5), f"Expected (10, 5), got {shap.shape}"
        assert shap.dtype == np.float32
        assert np.isfinite(shap).all()

    def test_explain_no_trees(self):
        """explain with empty trees returns zeros."""
        model = MockGBDTModel(n_features=3, n_trees=0)
        model._trees = []
        explainer = TreeSHAPExplainer(model, num_features=3)
        X = np.random.RandomState(42).randn(4, 3).astype(np.float64)
        shap = explainer.explain(X)
        assert shap.shape == (4, 3)
        assert np.allclose(shap, 0.0), "Expected all-zero SHAP with no trees"

    def test_explain_feature_mismatch_warns(self):
        """explain with mismatched feature count emits warning and adapts."""
        model = MockGBDTModel(n_features=5, n_trees=1)
        explainer = TreeSHAPExplainer(model, num_features=5)
        X = np.random.RandomState(42).randn(4, 3).astype(np.float64)
        with pytest.warns(UserWarning, match="Expected 5 features"):
            shap = explainer.explain(X)
        assert shap.shape == (4, 3)

    def test_feature_importance_shap(self):
        """feature_importance_shap returns (D,) array with finite values."""
        model = MockGBDTModel(n_features=5, n_trees=2)
        explainer = TreeSHAPExplainer(model, num_features=5)
        X = np.random.RandomState(42).randn(8, 5).astype(np.float64)
        importance = explainer.feature_importance_shap(X)
        assert importance.shape == (5,), f"Expected (5,), got {importance.shape}"
        assert np.isfinite(importance).all()
        assert np.all(importance >= 0), "Importance should be non-negative (mean |SHAP|)"

    def test_resolve_child_ids(self):
        """_resolve_child_ids extracts left/right child IDs from node dict."""
        node = {"nodeid": 0, "yes": 5, "no": 7}
        left, right = _resolve_child_ids(node)
        assert left == 5
        assert right == 7

    def test_resolve_child_ids_default(self):
        """_resolve_child_ids returns (-1, -1) for missing keys."""
        node = {"nodeid": 0}
        left, right = _resolve_child_ids(node)
        assert left == -1
        assert right == -1

    def test_resolve_leaf_value(self):
        """_resolve_leaf_value extracts leaf prediction value."""
        node = {"leaf": 0.42}
        assert _resolve_leaf_value(node) == pytest.approx(0.42)

    def test_resolve_leaf_value_default(self):
        """_resolve_leaf_value returns 0.0 for missing keys."""
        node = {"nodeid": 0}
        assert _resolve_leaf_value(node) == 0.0
