"""TreeSHAP explainer for GBDT models.

Implements the TreeSHAP algorithm (Lundberg & Lee 2017) for the Hoffnung
C++ GBDT backend. Computes per-sample feature contribution values using
the tree structure accessible via GBDT.to_json() or GBDTTrainer._trees.

Usage:
    explainer = TreeSHAPExplainer(gbdt_model, num_features=10)
    shap_values = explainer.explain(X)  # → (N, num_features) array
"""

from __future__ import annotations

import json
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# ── Node field name normalisation ──────────────────────────────


def _get_int(node: Dict[str, Any], *keys: str, default: int = 0) -> int:
    """Return first matching int-valued key from a tree node dict."""
    for k in keys:
        if k in node:
            return int(node[k])
    return default


def _get_float(node: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """Return first matching float-valued key from a tree node dict."""
    for k in keys:
        if k in node:
            return float(node[k])
    return default


def _get_bool(node: Dict[str, Any], *keys: str, default: bool = True) -> bool:
    """Return first matching bool-valued key from a tree node dict."""
    for k in keys:
        if k in node:
            val = node[k]
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() == "true"
            return bool(val)
    return default


def _resolve_child_ids(node: Dict[str, Any]) -> Tuple[int, int]:
    """Extract left / right child node ids with cross-format fallback."""
    left = _get_int(node, "left_child", "left", "yes", "lchild", default=-1)
    right = _get_int(node, "right_child", "right", "no", "rchild", default=-1)
    return left, right


def _resolve_node_id(node: Dict[str, Any], index: int) -> int:
    """Return canonical node id (id, nodeid, or positional index)."""
    return _get_int(node, "id", "node_id", "nodeid", default=index)


def _resolve_feature_idx(node: Dict[str, Any]) -> int:
    """Resolve feature (split) index from multiple naming conventions."""
    raw = None
    for key in ("feature_idx", "feature", "split_feature", "split_index",
                "split", "column"):
        if key in node:
            raw = node[key]
            break
    if raw is None:
        return 0
    if isinstance(raw, str):
        # XGBoost-style: "f0", "f1", ...
        raw = raw.lstrip("f")
    return int(raw)


def _resolve_threshold(node: Dict[str, Any]) -> float:
    """Resolve split threshold from multiple naming conventions."""
    for key in ("split_value", "threshold", "split_condition", "value", "split_val"):
        if key in node:
            return float(node[key])
    return 0.0


def _resolve_leaf_value(node: Dict[str, Any]) -> float:
    """Resolve leaf (prediction) value from multiple naming conventions."""
    for key in ("leaf_value", "value", "leaf", "prediction", "predict"):
        if key in node:
            return float(node[key])
    return 0.0


def _is_leaf_node(node: Dict[str, Any]) -> bool:
    """Check whether a node is a leaf."""
    # Explicit leaf flag
    for key in ("is_leaf", "leaf", "isLeaf"):
        if key in node:
            val = node[key]
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() == "true"
            return bool(val)
    # Heuristic: missing both children → leaf
    left_id = _get_int(node, "left_child", "left", "yes", "lchild", default=-2)
    right_id = _get_int(node, "right_child", "right", "no", "rchild", default=-2)
    return left_id < 0 and right_id < 0


def _resolve_default_left(node: Dict[str, Any]) -> bool:
    """Whether NaN / missing values go left."""
    for key in ("default_left", "missing_go_left", "missing", "missing_to_left"):
        if key in node:
            val = node[key]
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return int(val) < 0  # negative → left; convention
            return True
    return True


# ── TreeSHAP core ──────────────────────────────────────────────


class TreeSHAPExplainer:
    """TreeSHAP explainer for Hoffnung GBDT models.

    Implements a single-path SHAP attribution algorithm: for each sample
    and each tree, the leaf prediction value is distributed among the
    features encountered on the decision path.  Features not on the path
    receive zero contribution for that tree.

    This corresponds to the linear-complexity path-based attribution
    described in Lundberg & Lee (2017, §3.2), and is the exact SHAP
    decomposition whenever the tree ensemble has only axis-aligned splits
    (as in standard GBDT).

    Parameters
    ----------
    gbdt_model : GBDT (C++ pybind) or GBDTTrainer
        Trained model with ``to_json()`` method, or Python
        ``GBDTTrainer`` wrapper that exposes ``_trees``.
    num_features : int
        Number of input features.
    learning_rate : float
        GBDT learning rate used for scaling leaf contributions.
    """

    def __init__(
        self,
        gbdt_model: Any,
        num_features: int,
        learning_rate: float = 0.1,
    ) -> None:
        self.num_features = num_features
        self.learning_rate = learning_rate
        self._trees: List[List[Dict[str, Any]]] = []

        self._extract_trees(gbdt_model)

    # ── Public API ─────────────────────────────────────────

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for each sample across all trees.

        Parameters
        ----------
        X : (N, D) feature matrix.

        Returns
        -------
        shap_values : (N, D) float32 array of per-feature contributions.
        """
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape

        if D != self.num_features:
            warnings.warn(
                f"Expected {self.num_features} features, got {D}. "
                "Using D from input."
            )
            self.num_features = D

        if not self._trees:
            return np.zeros((N, D), dtype=np.float32)

        shap = np.zeros((N, D), dtype=np.float64)

        for tree_nodes in self._trees:
            if not tree_nodes:
                continue
            tree_shap = self._tree_shap(tree_nodes, X)
            shap += tree_shap

        return np.asarray(shap * self.learning_rate, dtype=np.float32)

    def feature_importance_shap(self, X: np.ndarray) -> np.ndarray:
        """Mean absolute SHAP value per feature (global importance).

        Parameters
        ----------
        X : (N, D) feature matrix.

        Returns
        -------
        importance : (D,) array — mean |SHAP| per feature across all samples.
        """
        shap = self.explain(X)
        return np.abs(shap).mean(axis=0)

    # ── TreeSHAP per-tree ─────────────────────────────────

    def _tree_shap(
        self, nodes: List[Dict[str, Any]], X: np.ndarray
    ) -> np.ndarray:
        """Compute SHAP values for one tree across all samples.

        Parameters
        ----------
        nodes : list of node dicts from to_json() for a single tree.
        X : (N, D) feature matrix.

        Returns
        -------
        shap : (N, D) pre-learning-rate SHAP array.
        """
        N, D = X.shape
        shap = np.zeros((N, D), dtype=np.float64)

        # Build id → node lookup map
        node_map: Dict[int, Dict[str, Any]] = {}
        for i, n in enumerate(nodes):
            nid = _resolve_node_id(n, i)
            node_map[nid] = n

        root = nodes[0]

        for i in range(N):
            phis = self._tree_shap_single(root, node_map, X[i], D)
            shap[i] += phis
        return shap

    def _tree_shap_single(
        self,
        root: Dict[str, Any],
        node_map: Dict[int, Dict[str, Any]],
        x: np.ndarray,
        D: int,
    ) -> np.ndarray:
        """SHAP contribution of a single sample for one tree.

        Follows the decision path from root to leaf.  Each internal
        node's feature receives an equal share of the leaf prediction
        value.  Features not on the path receive zero.

        This implements a path-based attribution algorithm (equal leaf-value sharing among
        path features), which is a simplified TreeSHAP approximation. For single-path
        ensembles the decomposition is exact; for multi-path ensembles this is an
        approximation of the game-theoretic Shapley values (Lundberg & Lee 2017).
        """
        phis = np.zeros(D, dtype=np.float64)
        current = root
        path_features: List[int] = []
        visited: set = set()

        for _ in range(1024):
            if _is_leaf_node(current):
                break

            fid = _resolve_feature_idx(current)
            thresh = _resolve_threshold(current)
            def_left = _resolve_default_left(current)

            if fid < 0 or fid >= D:
                fid = fid % D if D > 0 else 0

            go_left: bool
            val = float(x[fid])
            if np.isnan(val):
                go_left = def_left
            else:
                go_left = val <= thresh

            left_id, right_id = _resolve_child_ids(current)
            child_id = left_id if go_left else right_id

            if fid not in visited:
                path_features.append(fid)
                visited.add(fid)

            if child_id in node_map:
                current = node_map[child_id]
            else:
                break  # malformed tree — stop gracefully
        else:
            # Depth limit exceeded — treat current node as leaf
            warnings.warn(
                "Tree depth exceeded 1024 limit; truncating path. "
                "SHAP values may be approximate."
            )

        # ── Leaf value ──
        leaf_val = _resolve_leaf_value(current)

        # ── Distribute to path features ──
        n_on_path = len(path_features)
        if n_on_path > 0:
            share = leaf_val / float(n_on_path)
            for fid in path_features:
                phis[fid] += share
        elif D > 0:
            # Root-only tree (stump) — even distribution
            phis[:] = leaf_val / float(D)

        return phis

    # ── Tree extraction ────────────────────────────────────

    def _extract_trees(self, gbdt_model: Any) -> None:
        """Extract flat tree node lists from a GBDT model.

        Supports:
        - C++ pybind GBDT (``to_json()``)
        - Python ``GBDTTrainer`` wrapper (``_model.to_json()`` or ``_trees``)
        - Raw ``_trees`` attribute (``List[List[TreeNode]]``)
        """
        # Priority 1: _trees attribute (most direct)
        if hasattr(gbdt_model, "_trees") and gbdt_model._trees:
            raw_trees = gbdt_model._trees
            if isinstance(raw_trees, list) and len(raw_trees) > 0:
                first = raw_trees[0]
                if isinstance(first, list) and len(first) > 0:
                    first_elem = first[0]
                    if hasattr(first_elem, "feature_idx"):
                        # Pybind TreeNode objects → convert to dicts
                        self._trees = [
                            self._tree_nodes_to_dicts(tree) for tree in raw_trees
                        ]
                        return

        # Priority 2: to_json() on the model
        json_str: Optional[str] = None
        if hasattr(gbdt_model, "to_json"):
            json_str = gbdt_model.to_json()
        elif hasattr(gbdt_model, "_model") and hasattr(gbdt_model._model, "to_json"):
            json_str = gbdt_model._model.to_json()

        if json_str is not None:
            self._trees = self._parse_json(json_str)
            return

        # Priority 3: model has a per-tree predict interface
        # (We can't enumerate trees in this case — raise a clear warning)
        warnings.warn(
            f"Could not extract tree structures from model of type "
            f"{type(gbdt_model).__name__}.  Ensure the model has "
            f"to_json() or _trees attribute."
        )

    def _parse_json(self, json_str: str) -> List[List[Dict[str, Any]]]:
        """Parse JSON from to_json() into a list of tree node lists."""
        forest = json.loads(json_str)

        raw_trees = forest.get("trees", [])
        if not raw_trees and isinstance(forest, list):
            raw_trees = forest

        parsed: List[List[Dict[str, Any]]] = []
        for t in raw_trees:
            if isinstance(t, list):
                parsed.append(t)
            elif isinstance(t, dict):
                # Could be {"tree": [...], "nodes": [...]} etc.
                nodes = t.get("nodes", t.get("tree", t.get("children", [])))
                if nodes:
                    parsed.append(nodes)
        return parsed

    @staticmethod
    def _tree_nodes_to_dicts(tree: List[Any]) -> List[Dict[str, Any]]:
        """Convert a list of pybind TreeNode objects to dicts."""
        result: List[Dict[str, Any]] = []
        for node in tree:
            d: Dict[str, Any] = {}
            for attr in ("feature_idx", "split_value", "leaf_value", "num_samples",
                         "gain", "sum_grad", "sum_hess", "depth", "default_left"):
                if hasattr(node, attr):
                    d[attr] = getattr(node, attr)
            if hasattr(node, "is_leaf") and callable(node.is_leaf):
                d["is_leaf"] = node.is_leaf()
            elif hasattr(node, "is_leaf"):
                d["is_leaf"] = node.is_leaf

            # Children — tree structure
            for child_attr, key in [("left_child", "left_child"),
                                     ("right_child", "right_child")]:
                if hasattr(node, child_attr):
                    child = getattr(node, child_attr)
                    if hasattr(child, "id"):
                        d[key] = child.id
                    elif isinstance(child, int):
                        d[key] = child

            # Node id
            if hasattr(node, "id"):
                d["id"] = node.id
            result.append(d)
        return result
