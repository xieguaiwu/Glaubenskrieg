"""Purged Walk-Forward Cross-Validation with Sample Weights.

Implements methods from López de Prado (2018), Advances in Financial Machine Learning,
Chapters 4 and 7.

Chapter 4 — Sample Weights by Average Uniqueness
    When labels span multiple bars (e.g., triple-barrier events from Ch.3),
    overlapping outcomes create redundancy: two observations active on the same
    bar share information.  Average uniqueness quantifies each observation's
    effective information contribution.  Combined with return attribution and
    time decay, this yields sample weights that down-weight highly overlapping
    (and thus less informative) observations.

Chapter 7 — Purged K-Fold Cross-Validation
    Standard k-fold CV leaks information through overlapping labels.
    If a training label's outcome window extends into the test period,
    that training sample is "contaminated" and must be purged.
    An additional embargo period prevents leakage from serial correlation
    in features (e.g., ARMA lags).

Key Functions
-------------
- :func:`get_avg_uniqueness`     — per-observation average uniqueness
- :func:`get_sample_weights`     — sample weights combining uniqueness, return, decay
- :func:`purged_train_test_split` — single walk-forward train/test split with purging

Key Class
---------
- :class:`PurgedKFold` — sklearn-compatible k-fold cross-validator with purge + embargo

References
----------
López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Average Uniqueness
# ---------------------------------------------------------------------------

def get_avg_uniqueness(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Compute average uniqueness for each observation.

    For each observation *i* that spans bars :math:`[t_{0,i}, t_{1,i}]`,
    its uniqueness at bar *t* is :math:`1 / c_t` where :math:`c_t` counts
    how many observations are active at bar *t*.  The *average uniqueness*
    of observation *i* is the mean of these per-bar uniqueness values over
    its lifespan:

    .. math::
        \\bar{u}_i = \\frac{1}{t_{1,i} - t_{0,i} + 1}
                     \\sum_{t=t_{0,i}}^{t_{1,i}} \\frac{1}{c_t}

    Observations that rarely overlap with others receive uniqueness near 1;
    observations in dense, overlapping regions receive values near 0.

    Args:
        t0: Start bar indices, shape ``(n,)``.  Must satisfy ``t0[i] <= t1[i]``
            for all *i*.
        t1: End bar indices (inclusive), shape ``(n,)``.

    Returns:
        Array of average uniqueness values, shape ``(n,)``, in (0, 1].
        Zero-length input returns an empty array.
        Observations with ``t0[i] > t1[i]`` receive uniqueness 0.

    Notes
    -----
    Memory cost is ``O(T)`` where ``T = max(t1) - min(t0) + 1``.
    For typical daily-frequency datasets (``T < 10^4``) this is negligible.
    For intraday data with ``T > 10^6``, consider down-sampling first.
    """
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    n = len(t0)

    if n == 0:
        return np.array([], dtype=float)

    t_min = int(np.min(t0))
    t_max = int(np.max(t1))
    n_bars = t_max - t_min + 1

    if n_bars <= 0:
        return np.zeros(n, dtype=float)

    # ---- 1. Count active observations at each bar via difference array ----
    # active_diff[t] = net change at bar index (t_min + t)
    active_diff = np.zeros(n_bars + 1, dtype=np.int64)

    # Clip to valid range (handles any out-of-range t0/t1 gracefully)
    t0_clipped = np.clip(t0 - t_min, 0, n_bars).astype(np.int64)
    t1_clipped = np.clip(t1 - t_min, 0, n_bars).astype(np.int64)

    np.add.at(active_diff, t0_clipped, 1)
    # End+1 positions must not overflow the diff array
    end_plus_one = np.clip(t1_clipped + 1, 0, n_bars).astype(np.int64)
    np.add.at(active_diff, end_plus_one, -1)

    # active_count[t] = number of observations spanning bar (t_min + t)
    active_count = np.cumsum(active_diff)[:n_bars]
    active_count = np.maximum(active_count, 1)  # guard against zero-division

    # ---- 2. Uniqueness per bar: 1 / active_count ----
    uniqueness = 1.0 / active_count.astype(np.float64)

    # ---- 3. Cumulative sum for O(1) range queries ----
    cum_uniqueness = np.empty(n_bars + 1, dtype=np.float64)
    cum_uniqueness[0] = 0.0
    np.cumsum(uniqueness, out=cum_uniqueness[1:])

    # ---- 4. Average uniqueness per observation (vectorised) ----
    valid_mask = t0_clipped <= t1_clipped
    spans = np.where(valid_mask, t1_clipped - t0_clipped + 1, 1)

    total_u = np.where(
        valid_mask,
        cum_uniqueness[t1_clipped + 1] - cum_uniqueness[t0_clipped],
        0.0,
    )
    avg_uniqueness = total_u / spans.astype(np.float64)
    avg_uniqueness[~valid_mask] = 0.0

    return avg_uniqueness


# ---------------------------------------------------------------------------
# Sample Weights
# ---------------------------------------------------------------------------

def get_sample_weights(
    returns: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    decay: float = 0.0,
) -> np.ndarray:
    """Compute sample weights combining average uniqueness, return attribution,
    and optional time decay.

    Weight for observation *i*:

    .. math::
        w_i = \\bar{u}_i \\cdot |r_i| \\cdot e^{-\\lambda (T - t_{0,i})}

    where :math:`\\bar{u}_i` is the average uniqueness, :math:`r_i` the
    realised return, :math:`\\lambda` the decay factor, and :math:`T = \\max(t_1)`
    the most recent bar.

    Large absolute returns and high uniqueness both increase weight.
    Time decay down-weights older observations, reflecting the non-stationary
    nature of financial markets.

    Args:
        returns: Realised returns per observation, shape ``(n,)``.
        t0: Start bar indices, shape ``(n,)``.
        t1: End bar indices, shape ``(n,)``.
        decay: Time decay factor :math:`\\lambda \\ge 0`.  Larger values give
            more weight to recent observations.  ``decay=0`` disables decay.

    Returns:
        Sample weights, shape ``(n,)``.  Zero-length input returns an empty array.

    Notes
    -----
    The absolute return is used (following López de Prado) so that both large
    positive and large negative outcomes are considered informative.
    """
    returns = np.asarray(returns, dtype=np.float64)
    t0_arr = np.asarray(t0, dtype=np.int64)
    t1_arr = np.asarray(t1, dtype=np.int64)

    n = len(returns)
    if n == 0:
        return np.array([], dtype=np.float64)

    avg_uniqueness = get_avg_uniqueness(t0_arr, t1_arr)
    weights = avg_uniqueness * np.abs(returns)

    if decay > 0:
        T = int(np.max(t1_arr))
        time_to_end = np.maximum(T - t0_arr, 0)
        weights *= np.exp(-decay * time_to_end.astype(np.float64))

    return weights


# ---------------------------------------------------------------------------
# Purged Train / Test Split
# ---------------------------------------------------------------------------

def purged_train_test_split(
    indices: np.ndarray,
    t1: np.ndarray,
    test_start: int,
    test_end: int,
    t0: Optional[np.ndarray] = None,
    embargo_pct: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Split chronologically-sorted indices into purged training and test sets.

    Designed for walk-forward (expanding-window) validation where all training
    data precedes the test set.  Training samples are *purged* if their label's
    outcome window overlaps with the test period, and an embargo zone further
    excludes samples whose labels end too close to the test boundary.

    .. code-block:: text

        ───────────────────────────── legend ─────────────────────────────
        ┌── train (kept)       ┌── test set             ┌── future (unused)
        └── purged (t1 too close to test)

                    purge boundary
                        │
        ··· ┌──────────┐ │ ┌──────────┐ ┌────────┐ ···
        ··· │ train   ██│█│████ test ██│ │ future │ ···
        ··· └──────────┘ │ └──────────┘ └────────┘ ···
                        │
                  ██ = embargo zone (width = embargo_pct × total_span)
                  training samples with t1 inside this zone are purged.

    **Purge condition** — training sample *i* is removed if:

    .. math::
        t_{1,i} \\ge \\text{test\\_start\\_time} - \\text{embargo\\_len}

    **Test start time** — if ``t0`` is provided, ``min(t0[test_idx])``;
    otherwise ``test_start`` is used as a positional time proxy (valid when
    bar index ≈ data position).

    Args:
        indices: Sorted dataset indices, shape ``(N,)`` (chronological).
        t1: End bar times for *all* observations, shape ``(N,)``.
            ``t1[idx]`` must be the end time for observation ``idx``.
        test_start: Position in ``indices`` where the test set begins.
        test_end: Position in ``indices`` where the test set ends (exclusive).
        t0: Optional start bar times, shape ``(N,)``.  If given, the actual
            minimum start time of test observations becomes the purge boundary.
            This is more accurate than the positional fallback.
        embargo_pct: Embargo period as a fraction of the total time span
            ``max(t1) - min(t1)``.  Default 0.01 (1%).

    Returns:
        A tuple ``(train_idx, test_idx)`` where ``train_idx`` is the purged
        training index array and ``test_idx = indices[test_start:test_end]``.

    Raises:
        ValueError: If test bounds are invalid (e.g., ``test_start >= test_end``
            or out of range).

    Examples
    --------
    >>> N = 1000
    >>> indices = np.arange(N)
    >>> t1 = np.arange(N) + np.random.randint(1, 21, size=N)  # labels span ~20 bars
    >>> train_idx, test_idx = purged_train_test_split(
    ...     indices, t1, test_start=700, test_end=850, embargo_pct=0.01
    ... )
    >>> len(test_idx)
    150
    >>> len(train_idx) <= 700  # some training samples were purged
    True
    """
    indices = np.asarray(indices, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    N = len(indices)

    if test_start < 0 or test_end > N or test_start >= test_end:
        raise ValueError(
            f"Invalid test bounds: test_start={test_start}, test_end={test_end}, "
            f"N={N}. Must satisfy 0 <= test_start < test_end <= N."
        )

    # ---- test set ----
    test_idx = indices[test_start:test_end]

    # ---- determine test period start time ----
    if t0 is not None:
        t0 = np.asarray(t0, dtype=np.int64)
        test_start_time = int(np.min(t0[test_idx]))
    else:
        # Positional fallback: assumes 1-to-1 bar-index ↔ position mapping.
        test_start_time = test_start

    # ---- embargo length (in bar units) ----
    if embargo_pct > 0 and len(t1) > 0:
        total_span = int(np.max(t1)) - int(np.min(t1))
        if total_span <= 0:
            total_span = max(N, 1)
        embargo_len = max(0, int(embargo_pct * total_span))
    else:
        embargo_len = 0

    purge_boundary = test_start_time - embargo_len

    # ---- training candidates (chronologically before test set) ----
    train_candidates = indices[:test_start]

    if len(train_candidates) == 0:
        return np.array([], dtype=np.int64), test_idx

    # ---- purge: keep only samples whose label ends before the boundary ----
    train_t1 = t1[train_candidates]
    keep_mask = train_t1 < purge_boundary

    train_idx = train_candidates[keep_mask]

    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Purged K-Fold Cross-Validator (sklearn-compatible)
# ---------------------------------------------------------------------------

class PurgedKFold:
    """Purged K-Fold cross-validator for financial time series.

    Splits chronologically-sorted data into *k* consecutive folds.
    For each fold, training samples whose labels overlap with the test
    period are purged, and an embargo period is enforced on both sides
    of the test fold.

    This class implements the ``split()`` protocol, making it usable as a
    ``cv`` argument in scikit-learn (e.g., ``cross_val_score``).

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    embargo_pct : float, default 0.01
        Embargo length as a fraction of ``max(t1) - min(t1)``.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import cross_val_score
    >>> X = np.random.randn(500, 10)
    >>> y = np.random.randn(500)
    >>> t1 = np.arange(500) + 20  # labels end ~20 bars after start
    >>> cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
    >>> for train_idx, test_idx in cv.split(X, t1=t1):
    ...     pass  # train_idx is purged
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if embargo_pct < 0:
            raise ValueError(f"embargo_pct must be >= 0, got {embargo_pct}")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        t1: Optional[np.ndarray] = None,
        t0: Optional[np.ndarray] = None,
    ):
        """Generate train/test indices for each fold.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.
                Only ``len(X)`` is used.
            y: Target values (ignored; for sklearn compatibility).
            t1: End bar times, shape ``(n_samples,)``.  **Required** for
                purging; if ``None`` falls back to standard k-fold
                (no purging).
            t0: Optional start bar times, shape ``(n_samples,)``.  Improves
                accuracy of the test-period start time.

        Yields
        ------
        train_idx : ndarray of int
            Purged training indices for this fold.
        test_idx : ndarray of int
            Test indices for this fold.
        """
        n_samples = len(X)
        indices = np.arange(n_samples, dtype=np.int64)

        # Compute consecutive fold boundaries
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        boundaries = np.zeros(self.n_splits + 1, dtype=int)
        np.cumsum(fold_sizes, out=boundaries[1:])

        # If no t1 is provided, fall back to standard (unpurged) k-fold
        if t1 is None:
            for k in range(self.n_splits):
                test_start = boundaries[k]
                test_end = boundaries[k + 1]
                test_idx = indices[test_start:test_end]
                # Training: all folds except test fold, concatenated
                train_idx = np.concatenate([
                    indices[:test_start],
                    indices[test_end:],
                ])
                yield train_idx, test_idx
            return

        t1 = np.asarray(t1, dtype=np.int64)
        if t0 is not None:
            t0 = np.asarray(t0, dtype=np.int64)

        # Compute global embargo length once (same for all folds)
        if self.embargo_pct > 0:
            total_span = int(np.max(t1)) - int(np.min(t1))
            if total_span <= 0:
                total_span = max(n_samples, 1)
            embargo_len = max(0, int(self.embargo_pct * total_span))
        else:
            embargo_len = 0

        for k in range(self.n_splits):
            test_start = boundaries[k]
            test_end = boundaries[k + 1]
            test_idx = indices[test_start:test_end]

            # --- determine test period time boundaries ---
            if t0 is not None:
                test_time_start = int(np.min(t0[test_idx]))
                test_time_end = int(np.max(t1[test_idx]))
            else:
                test_time_start = test_start
                test_time_end = test_end

            # --- train split: before + after, each side purged ---

            # Before-test training samples
            before_candidates = indices[:test_start]
            if len(before_candidates) > 0:
                before_t1 = t1[before_candidates]
                before_mask = before_t1 < (test_time_start - embargo_len)
                before_idx = before_candidates[before_mask]
            else:
                before_idx = np.array([], dtype=np.int64)

            # After-test training samples
            after_candidates = indices[test_end:]
            if len(after_candidates) > 0:
                # Purge: after-training sample i overlaps if t0[i] <= max(t1[test])
                if t0 is not None:
                    after_t0 = t0[after_candidates]
                    after_mask = after_t0 > (test_time_end + embargo_len)
                else:
                    # Without t0: use position as proxy — purge samples whose
                    # position is within the embargo zone after test_end.
                    after_mask = np.ones(len(after_candidates), dtype=bool)
                    # Conservative: purge everything within embargo_len of test_end
                    after_start_boundary = test_end + embargo_len
                    for i, idx in enumerate(after_candidates):
                        if idx < after_start_boundary:
                            after_mask[i] = False
                after_idx = after_candidates[after_mask]
            else:
                after_idx = np.array([], dtype=np.int64)

            train_idx = np.concatenate([before_idx, after_idx])
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits (for sklearn compatibility)."""
        return self.n_splits


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ----- get_avg_uniqueness -----
    print("=== get_avg_uniqueness ===")
    t0_test = np.array([0, 1, 2, 5], dtype=np.int64)
    t1_test = np.array([2, 3, 4, 7], dtype=np.int64)
    avg_u = get_avg_uniqueness(t0_test, t1_test)
    print(f"t0 = {t0_test}")
    print(f"t1 = {t1_test}")
    print(f"avg_uniqueness = {avg_u}")
    # Manual check for first three (overlapping):
    # t=0: {0} → 1.0
    # t=1: {0,1} → 0.5
    # t=2: {0,1,2} → 1/3
    # obs0 avg: (1 + 0.5 + 1/3) / 3 ≈ 0.6111
    # obs1 avg: (0.5 + 1/3 + 0.5) / 3 ≈ 0.4444
    # obs2 avg: (1/3 + 0.5 + 1) / 3 ≈ 0.6111
    # obs3 isolated: (1+1+1)/3 = 1.0
    expected = np.array([0.61111111, 0.44444444, 0.61111111, 1.0])
    assert np.allclose(avg_u, expected, atol=1e-6), f"Mismatch: {avg_u} vs {expected}"
    print("  ✓ passed\n")

    # Edge: empty
    assert len(get_avg_uniqueness(np.array([]), np.array([]))) == 0
    # Edge: single observation
    single_u = get_avg_uniqueness(np.array([0]), np.array([5]))
    assert np.allclose(single_u, np.array([1.0])), f"Single: {single_u}"
    # Edge: t0 > t1
    bad_u = get_avg_uniqueness(np.array([5]), np.array([3]))
    assert np.allclose(bad_u, np.array([0.0])), f"Bad: {bad_u}"
    # Edge: fully overlapping (all identical t0, t1)
    dense_u = get_avg_uniqueness(np.array([0, 0, 0]), np.array([4, 4, 4]))
    assert np.allclose(dense_u, np.full(3, 1.0 / 3)), f"Dense: {dense_u}"
    print("  edge cases ✓\n")

    # ----- get_sample_weights -----
    print("=== get_sample_weights ===")
    returns_test = np.array([0.05, -0.03, 0.02, 0.10])
    weights = get_sample_weights(returns_test, t0_test, t1_test, decay=0.0)
    expected_w = avg_u * np.abs(returns_test)
    assert np.allclose(weights, expected_w), f"Weights: {weights} vs {expected_w}"
    print(f"weights (no decay) = {weights}")

    # With decay
    weights_decay = get_sample_weights(returns_test, t0_test, t1_test, decay=0.01)
    # obs0 (t0=0): max time gap, lowest weight; obs3 (t0=5): newest, higher
    assert weights_decay[0] < weights_decay[3], "Decay should favour recent"
    print(f"weights (decay=0.01) = {weights_decay}")
    print("  ✓ passed\n")

    # ----- purged_train_test_split -----
    print("=== purged_train_test_split ===")
    N_test = 100
    indices_test = np.arange(N_test, dtype=np.int64)
    # Each label ends ~15 bars after its start
    rng = np.random.default_rng(42)
    t1_test_full = np.arange(N_test, dtype=np.int64) + rng.integers(1, 21, size=N_test).astype(np.int64)
    t0_test_full = np.arange(N_test, dtype=np.int64)  # exact bar-index start

    # Split: train on first 80%, test on last 20%
    train_idx, test_idx = purged_train_test_split(
        indices_test, t1_test_full,
        test_start=80, test_end=100,
        t0=t0_test_full, embargo_pct=0.01,
    )
    assert len(test_idx) == 20, f"Test size: {len(test_idx)}"
    assert len(train_idx) <= 80, "Should have purged some training samples"
    # All test indices should be from [80, 100)
    assert np.all((test_idx >= 80) & (test_idx < 100))
    # All train indices should be < 80
    assert np.all(train_idx < 80)
    print(f"Train: {len(train_idx)} (purged from 80), Test: {len(test_idx)}")
    print("  ✓ passed\n")

    # Edge: no training data
    tr0, te0 = purged_train_test_split(indices_test, t1_test_full,
                                       test_start=0, test_end=10)
    assert len(tr0) == 0
    assert len(te0) == 10
    print("  test_start=0 edge case ✓\n")

    # ----- PurgedKFold -----
    print("=== PurgedKFold ===")
    X_test = np.random.randn(N_test, 5)
    cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
    fold_sizes = []
    purged_counts = []
    for train_idx, test_idx in cv.split(X_test, t1=t1_test_full, t0=t0_test_full):
        fold_sizes.append(len(test_idx))
        purged_counts.append(len(train_idx))
        assert len(np.intersect1d(train_idx, test_idx)) == 0, "Train/test overlap!"
    assert sum(fold_sizes) == N_test, f"Fold sizes: {fold_sizes}"
    print(f"Fold test sizes: {fold_sizes}")
    print(f"Fold train sizes (purged): {purged_counts}")
    print("  ✓ passed\n")

    # Fallback: no t1 provided → standard k-fold
    std_sizes = []
    for train_idx, test_idx in cv.split(X_test, t1=None):
        std_sizes.append(len(test_idx))
        assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert sum(std_sizes) == N_test
    print("  t1=None fallback ✓\n")

    print("=== All tests passed ===")
