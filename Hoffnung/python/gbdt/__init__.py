"""GBDT for Quantitative Investment — Python Package.

Provides:
    - C++ core bindings (GBDT, GBDTConfig, TreeNode) [requires compiled ``gbdt_python``]
    - Differentiable quant loss functions (MSE, MAE, Huber, RankIC, Sharpe,
      Composite, Quantile)
    - GBDTTrainer: high-level class bridging Python loss functions to the C++ core
    - Data pipeline utilities (synthetic data, train/val split, IC computation)
    - Convenience ``train_test_split`` for panel data
"""

__version__ = "0.1.0"

import numpy as np
import torch
from typing import Optional, Tuple, Callable, List, Union

# ── C++ core bindings (lazy — safe to import without compiled module) ──
_CPP_AVAILABLE = False
_CPP_ERROR = ""
try:
    from gbdt_python import GBDT, GBDTConfig, MonotoneConstraint, MonotoneDirection, TreeNode  # noqa: F401
    _CPP_AVAILABLE = True
except ImportError as e:
    _CPP_ERROR = str(e)
    GBDT = None  # type: ignore
    GBDTConfig = None  # type: ignore
    MonotoneConstraint = None  # type: ignore
    MonotoneDirection = None  # type: ignore
    TreeNode = None  # type: ignore

# ── Loss functions ───────────────────────────
from .losses import (
    mse_loss,
    mae_loss,
    huber_loss,
    quantile_loss,
    log_loss,
    rankic_loss,
    sharpe_loss,
    composite_quant_loss,
    CompositeQuantLoss,
    compute_gradients,
)

# ── Pipeline utilities ───────────────────────
from .pipeline import (
    make_synthetic_data,
    train_val_split as _pipeline_split,
    compute_ic,
    purged_walk_forward_split,
    purged_walk_forward_cv,
)

# ──────────────────────────────────────────────
#  GBDTTrainer — high-level Python→C++ bridge
# ──────────────────────────────────────────────


class GBDTTrainer:
    """High-level trainer bridging Python loss functions to the C++ GBDT core.

    The C++ core builds trees from pre-computed gradients and Hessians.
    This class orchestrates the training loop in Python: at each boosting
    round it computes gradients via the user-supplied loss function, passes
    them to ``GBDT.fit_one_tree()``, then updates predictions.

    This enables using any PyTorch loss (RankIC, Composite, Sharpe, custom)
    with the C++ tree builder — the core limitation that the original
    ``GBDT.fit()`` only supported MSE/MAE/Huber.

    Example::

        from gbdt import GBDTConfig, GBDTTrainer
        from gbdt.losses import huber_loss, rankic_loss

        config = GBDTConfig()
        config.num_trees = 100
        config.max_depth = 6
        config.learning_rate = 0.05

        trainer = GBDTTrainer(config, loss_fn=huber_loss)
        trainer.fit(X_train, y_train, X_val, y_val)
        preds = trainer.predict(X_test)
    """

    def __init__(
        self,
        config: object,
        loss_fn: Optional[Callable] = None,
    ):
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                f"C++ module 'gbdt_python' not available: {_CPP_ERROR}\n"
                "Build it first with: cd build && cmake .. && make"
            )
        self._model = GBDT(config)
        assert self._model is not None
        self.config = config
        self._loss_fn = loss_fn if loss_fn is not None else mse_loss
        self._init_pred: float = 0.0
        self._trees: List = []
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        self._tree_step_sizes: List[float] = []
        self._num_features: int = 0
        self._constraints: List = []

    @property
    def train_losses(self) -> List[float]:
        return self._train_losses

    @property
    def val_losses(self) -> List[float]:
        return self._val_losses

    @property
    def num_trees(self) -> int:
        return len(self._trees)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        constraints: Optional[List] = None,
    ) -> "GBDTTrainer":
        """Train the model using the configured loss function.

        At each boosting round:
        1. Compute current predictions.
        2. Evaluate loss_fn → (loss, gradients, hessians).
        3. Call ``GBDT.fit_one_tree(gradients, hessians)`` to build a tree.
        4. Update predictions: ``y_pred += lr * tree_pred``.
        5. Optionally perform line search for step size.
        6. Record train/val loss for monitoring.

        Args:
            X: Feature matrix, shape [N, D], float32.
            y: Target vector, shape [N], float32.
            X_val: Optional validation features, shape [N_val, D].
            y_val: Optional validation targets, shape [N_val].
            constraints: Optional list of MonotoneConstraint objects.
                Each MonotoneConstraint specifies a feature index and
                a monotone direction (INCREASING, DECREASING, or NONE).

        Returns:
            self (for chaining).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("Non-finite values detected in X at start of fit().")
        if not np.isfinite(y).all():
            raise ValueError("Non-finite values detected in y at start of fit().")

        self._num_features = X.shape[1]
        self._constraints = constraints if constraints is not None else []

        N = X.shape[0]
        self._init_pred = float(y.mean())
        y_pred = np.full(N, self._init_pred, dtype=np.float32)

        has_val = X_val is not None and y_val is not None
        y_pred_val: Optional[np.ndarray] = None
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            y_pred_val = np.full(X_val.shape[0], self._init_pred, dtype=np.float32)

        best_val_loss = float("inf")
        patience_counter = 0
        best_num_trees = 0  # track best model state for rollback

        for round_idx in range(self.config.num_trees):
            y_true_t = torch.from_numpy(y)
            y_pred_t = torch.from_numpy(y_pred)

            loss, grads, hess = self._loss_fn(y_true_t, y_pred_t)
            train_loss = float(loss.item())

            if grads.shape != y_pred_t.shape:
                raise ValueError(
                    f"Gradient shape {grads.shape} != prediction shape {y_pred_t.shape}. "
                    "Custom loss must return per-sample gradients."
                )
            if hess.shape != y_pred_t.shape:
                raise ValueError(
                    f"Hessian shape {hess.shape} != prediction shape {y_pred_t.shape}. "
                    "Custom loss must return per-sample Hessians."
                )

            grads_np = grads.numpy()
            hess_np = hess.numpy()

            tree = self._model.fit_one_tree(X, grads_np, hess_np, self._constraints)
            self._trees.append(tree)

            tree_pred = self._model.predict_tree(tree, X)

            if self.config.use_line_search:
                g_t = (grads_np * tree_pred).sum()
                h_t2 = (hess_np * tree_pred * tree_pred).sum()
                rho = -g_t / h_t2 if h_t2 > 1e-10 else 1.0
            else:
                rho = 1.0

            step = self.config.learning_rate * rho
            self._tree_step_sizes.append(step)
            y_pred = y_pred + step * tree_pred

            if has_val and y_pred_val is not None:
                tree_pred_val = self._model.predict_tree(tree, X_val)
                y_pred_val = y_pred_val + step * tree_pred_val
                y_val_t = torch.from_numpy(y_val)
                y_pred_val_t = torch.from_numpy(y_pred_val)
                val_loss_val, _, _ = self._loss_fn(y_val_t, y_pred_val_t)
                val_loss = float(val_loss_val.item())
            else:
                val_loss = train_loss

            self._train_losses.append(train_loss)
            self._val_losses.append(val_loss)

            if has_val and self.config.early_stopping_rounds > 0:
                if val_loss < best_val_loss - self.config.early_stopping_tol:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_num_trees = len(self._trees)
                else:
                    patience_counter += 1
                if patience_counter >= self.config.early_stopping_rounds:
                    self._trees = self._trees[:best_num_trees]
                    self._tree_step_sizes = self._tree_step_sizes[:best_num_trees]
                    self._train_losses = self._train_losses[:best_num_trees]
                    self._val_losses = self._val_losses[:best_num_trees]
                    break

        self._model.set_state(self._init_pred, self._trees, self._tree_step_sizes)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        self._model.set_state(self._init_pred, self._trees, self._tree_step_sizes)
        return self._model.predict(X).astype(np.float32)

    def get_feature_importance(self) -> np.ndarray:
        """Aggregate split-count feature importance across all trees.

        Returns:
            Importance scores, shape [D], summing to 1.0.
        """
        D = self._num_features
        if D == 0 and self._trees:
            # Fallback: infer from maximum feature index in tree nodes
            D = 1 + max(
                n.feature_idx
                for tree in self._trees
                for n in tree
                if n.feature_idx >= 0
            )
        if D == 0:
            return np.array([], dtype=np.float32)

        importance = np.zeros(D, dtype=np.float32)
        for tree in self._trees:
            for node in tree:
                if not node.is_leaf() and 0 <= node.feature_idx < D:
                    importance[node.feature_idx] += 1.0

        total = importance.sum()
        if total > 0:
            importance /= total
        return importance

    def __repr__(self) -> str:
        return (
            f"<GBDTTrainer trees={self.num_trees} "
            f"loss={getattr(self._loss_fn, '__name__', type(self._loss_fn).__name__)} "
            f"lr={self.config.learning_rate}>"
        )


def train_test_split(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    by_time: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """Convenience split for panel (time-series cross-section) data.

    Two modes:

    **Panel mode** (``by_time=True``, default):
        Prevents lookahead bias by splitting along the time axis.
        ``X`` should have shape ``[T, N, F]`` (time × assets × features).
        The first ``(1 - test_size)`` time-steps go to training, the rest to test.

        If ``y`` is provided it should have shape ``[T, N]``.

        Returns:
            ``(X_train, X_test, y_train, y_test)`` (panel) or
            ``(X_train, X_test)`` (if ``y`` is None).

    **Flat mode** (``by_time=False``):
        Performs a standard random shuffle split.
        ``X`` shape ``[N, F]``, ``y`` shape ``[N]``.

    Args:
        X: Feature array — shape ``[T, N, F]`` (panel) or ``[N, F]`` (flat).
        y: Target array — shape ``[T, N]`` (panel) or ``[N]`` (flat).
           Optional — if omitted only X splits are returned.
        test_size: Fraction of the dataset for testing (0 < test_size < 1).
        by_time: If True (default), split along time axis (panel mode).
                 If False, random shuffle split.
        random_state: Seed for reproducibility (only used when
                      ``by_time=False``).

    Returns:
        - With ``y``: ``(X_train, X_test, y_train, y_test)``
        - Without ``y``: ``(X_train, X_test)``
    """
    if by_time:
        # ── Panel mode: split along time axis ────────────────────
        T = X.shape[0]
        split_idx = int(T * (1.0 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]

        if y is not None:
            y_train, y_test = y[:split_idx], y[split_idx:]
            return X_train, X_test, y_train, y_test
        return X_train, X_test

    # ── Flat mode: random split ───────────────────────────────────
    rng = np.random.default_rng(random_state)
    N = X.shape[0]
    indices = np.arange(N)
    rng.shuffle(indices)
    split_idx = int(N * (1.0 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = X[train_idx], X[test_idx]

    if y is not None:
        y_train, y_test = y[train_idx], y[test_idx]
        return X_train, X_test, y_train, y_test
    return X_train, X_test


# Convenience re-export
train_val_split = _pipeline_split

__all__ = [
    # Version
    "__version__",
    # C++ core
    "GBDT",
    "GBDTConfig",
    "GBDTTrainer",
    "MonotoneConstraint",
    "MonotoneDirection",
    "TreeNode",
    # Losses
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "quantile_loss",
    "log_loss",
    "rankic_loss",
    "sharpe_loss",
    "composite_quant_loss",
    "CompositeQuantLoss",
    "compute_gradients",
    # Pipeline
    "make_synthetic_data",
    "train_test_split",
    "train_val_split",
    "compute_ic",
    "purged_walk_forward_split",
    "purged_walk_forward_cv",
]
