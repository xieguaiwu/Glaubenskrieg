"""
Fractional Differentiation (FFD) per López de Prado (2018),
Advances in Financial Machine Learning, Chapter 5.

Fixed-width window FFD for transforming non-stationary financial time series into
stationary ones while preserving as much memory (predictive structure) as possible.

Key insight: integer differencing (d=1) destroys all memory; fractional orders
(0 < d < 1) find the minimum amount of differencing needed to achieve stationarity,
retaining ~99% of the correlation structure when d is properly tuned.

Reference:
    López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ffd_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Compute fractional differentiation weights recursively.

    Weights are ω_k = (-1)^k · binom(d, k), computed via the recurrence:
        ω_0 = 1
        ω_k = ω_{k-1} · (k − 1 − d) / k      for k ≥ 1

    The sequence terminates when |ω_k| < threshold, which bounds the effective
    window width and prevents the infinite memory of pure fractional differencing.

    Args:
        d: Fractional differentiation order in [0, 1].
           0 → identity (no differencing), 1 → first difference.
        threshold: Drop weights whose absolute value falls below this value.
            Smaller thresholds produce longer windows.

    Returns:
        1D numpy array of weights [ω_0, ω_1, …, ω_K] where K + 1 is the
        effective window width.

    Examples:
        >>> ffd_weights(0.0)
        array([1.])
        >>> ffd_weights(1.0)
        array([ 1., -1.])
        >>> w = ffd_weights(0.35, threshold=0.05)
        >>> len(w)  # window width grows with d and shrinks with larger threshold
        4
    """
    if not (0.0 <= d <= 1.0):
        raise ValueError(f"Order d must be in [0, 1], got {d}")

    weights = [1.0]
    k = 1
    while True:
        w = weights[-1] * (k - 1.0 - d) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    return np.array(weights, dtype=float)


def fractional_diff(
    series,
    d: float,
    threshold: float = 1e-5,
) -> np.ndarray:
    """Apply fixed-width window fractional differentiation to a time series.

    For each time t (where enough history is available), computes:
        X'_t = Σ_{k=0}^{K} ω_k · X_{t−k}

    where ω_k are the FFD weights and K is determined by *threshold*.
    Early observations (t < K) are set to NaN.

    Args:
        series: 1D array-like of time series values (e.g. log-prices).
        d: Fractional differentiation order in [0, 1].
            d=0 returns the series unchanged; d=1 returns the first difference.
        threshold: Minimum absolute weight retained; controls window width.

    Returns:
        1D numpy array of the same length as *series*.  The first (K−1)
        elements are NaN (insufficient history); subsequent elements carry the
        fractionally differenced values.

    Examples:
        >>> fractional_diff(np.array([1., 2., 3., 4., 5.]), d=0)
        array([1., 2., 3., 4., 5.])
        >>> fractional_diff(np.array([1., 2., 3., 4., 5.]), d=1)
        array([nan,  1.,  1.,  1.,  1.])
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> rw = np.cumsum(np.random.randn(100))  # random walk (non-stationary)
        >>> fd = fractional_diff(rw, d=0.5, threshold=0.01)
        >>> np.isnan(fd[:9]).all()  # start is NaN-padded (width-1 = 9)
        True
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    # --- Edge cases ---
    if d == 0.0:
        return series.copy()
    if n == 0:
        return np.array([], dtype=float)

    # --- Compute weights ---
    weights = ffd_weights(d, threshold)
    width = len(weights)

    if width > n:
        # Window wider than the series: not enough data anywhere
        return np.full(n, np.nan)

    # --- Vectorised convolution ---
    # result[t] = Σ_{k=0}^{width-1} ω_k · series[t−k]   for t ≥ width−1
    result = np.zeros(n, dtype=float)
    for k in range(width):
        start = width - 1
        result[start:] += weights[k] * series[start - k : n - k]

    # NaN-pad the initial segment where full window is unavailable
    result[: width - 1] = np.nan

    return result


def find_min_d(
    series,
    max_d: float = 1.0,
    step: float = 0.05,
    p_threshold: float = 0.05,
) -> float:
    """Find the minimum fractional order *d* that achieves stationarity.

    Searches over d ∈ {0, step, 2·step, …, max_d} in increasing order and
    returns the first value for which the Augmented Dickey-Fuller (ADF) test
    rejects the unit-root null at *p_threshold*.

    Args:
        series: 1D array-like of time series values (e.g. log-prices).
        max_d: Upper bound of the search grid.
        step: Grid spacing.
        p_threshold: p-value threshold for the ADF test (default 0.05).

    Returns:
        The smallest *d* (float) for which the fractionally differenced series
        is stationary according to the ADF test.  If no *d* in the grid achieves
        stationarity, *max_d* is returned (the series may still be non-stationary
        at that order).

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> rw = np.cumsum(np.random.randn(500))  # random walk
        >>> d_min = find_min_d(rw, max_d=1.0, step=0.1, p_threshold=0.05)
        >>> 0.0 < d_min <= 1.0
        True
        >>> # Stationary series already at d=0
        >>> stationary = np.random.randn(200)
        >>> find_min_d(stationary, max_d=0.5, step=0.05, p_threshold=0.05)
        0.0
    """
    series = np.asarray(series, dtype=float)

    # Grid of candidate orders
    n_steps = int(np.round(max_d / step)) + 1
    candidates = np.linspace(0.0, max_d, n_steps)

    for d in candidates:
        diff_series = fractional_diff(series, d, threshold=1e-5)
        # Drop NaN padding before testing
        valid = diff_series[~np.isnan(diff_series)]
        if len(valid) < 20:
            continue  # too few valid observations for a reliable test
        _, p_val = _adf_test(valid)
        if p_val < p_threshold:
            return float(d)

    # Fallback: no d in [0, max_d] achieves stationarity
    return float(max_d)


# ---------------------------------------------------------------------------
# Internal: Augmented Dickey-Fuller test (NumPy-only OLS)
# ---------------------------------------------------------------------------

def _adf_test(series, max_lag=None):
    """Augmented Dickey-Fuller test for a unit root (constant-only, no trend).

    Estimates the regression:
        Δy_t = α + γ · y_{t-1} + Σ_{i=1}^{p} δ_i · Δy_{t-i} + ε_t

    H₀: γ = 0  (unit root / non-stationary)
    H₁: γ < 0  (stationary)

    Uses OLS via np.linalg.lstsq and MacKinnon (1994) asymptotic critical
    values for the t-statistic of γ̂.

    Args:
        series: 1D float array, must be NaN-free.
        max_lag: Number of lagged-difference terms.  If None, defaults to
            Schwert's rule: floor((n−1)^{1/3}).

    Returns:
        (test_statistic: float, p_value: float)
    """
    y = np.asarray(series, dtype=float)
    n = len(y)

    if n < 10:
        return 0.0, 1.0

    # --- Lag selection ---
    if max_lag is None:
        max_lag = max(1, int(np.floor(np.cbrt(n - 1))))
    max_lag = min(max_lag, (n - 1) // 3)  # hard cap to avoid over-fitting

    delta_y = np.diff(y)          # length n−1
    n_diff = len(delta_y)
    T = n_diff - max_lag           # usable regression observations

    if T <= max_lag + 2:
        return 0.0, 1.0

    # --- Build design matrix and response ---
    n_cols = 2 + max_lag           # const, y_{t-1}, Δy_{t-1} … Δy_{t-p}
    X = np.ones((T, n_cols), dtype=float)
    Y = np.empty(T, dtype=float)

    for j in range(T):
        m = max_lag + j            # index into delta_y
        Y[j] = delta_y[m]
        X[j, 1] = y[m]             # lagged level
        for i_lag in range(max_lag):
            X[j, 2 + i_lag] = delta_y[m - 1 - i_lag]

    # --- OLS estimation ---
    try:
        beta, _residuals, _rank, _s = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    gamma_hat = beta[1]            # coefficient on y_{t-1}

    # Standard error of γ̂
    residuals = Y - X @ beta
    sigma2 = float(np.sum(residuals**2)) / (T - n_cols)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    se_gamma = np.sqrt(sigma2 * XtX_inv[1, 1])

    if se_gamma < 1e-15:
        return 0.0, 0.0          # degenerate — treat as stationary

    t_stat = float(gamma_hat / se_gamma)

    # --- Critical values (MacKinnon 1994, constant-only, no trend) ---
    cv = _adf_critical_values(T)

    # --- Approximate p-value via linear interpolation ---
    p_val = _pvalue_from_tstat(t_stat, cv)

    return t_stat, p_val


def _adf_critical_values(n_obs: int):
    """Return {1%, 5%, 10%} asymptotic critical values for the ADF test
    (constant only, no trend), interpolated by sample size."""
    # Tabulated asymptotic values for selected sample sizes
    # (MacKinnon 1994, Table 1 — constant, no trend)
    table = {
        25:   {1: -3.75, 5: -3.00, 10: -2.63},
        50:   {1: -3.58, 5: -2.93, 10: -2.60},
        100:  {1: -3.51, 5: -2.89, 10: -2.58},
        250:  {1: -3.46, 5: -2.88, 10: -2.57},
        500:  {1: -3.44, 5: -2.87, 10: -2.57},
        float("inf"): {1: -3.43, 5: -2.86, 10: -2.57},
    }
    sizes = sorted(table.keys())
    for size in sizes:
        if n_obs <= size:
            return table[size]
    return table[float("inf")]


def _pvalue_from_tstat(t_stat: float, cv: dict) -> float:
    """Approximate the ADF p-value by linear interpolation of critical values.

    The interpolation is exact at the 1 %, 5 %, and 10 % critical values (by
    construction).  Values between thresholds are linearly interpolated.
    t-statistics above the 10 % critical value (closer to zero) are assigned
    0.50 — clearly non-stationary and well above any conventional threshold.
    """
    c1, c5, c10 = cv[1], cv[5], cv[10]

    if t_stat <= c1:
        # Beyond 1 % — map to (0.001, 0.01]
        ratio = min(1.0, max(0.0, (c1 - t_stat) / (-c1)))
        return 0.001 + 0.009 * (1.0 - ratio)
    elif t_stat <= c5:
        # [1 %, 5 %] → [0.01, 0.05]
        return 0.01 + 0.04 * (c1 - t_stat) / (c1 - c5)
    elif t_stat <= c10:
        # [5 %, 10 %] → [0.05, 0.10]
        return 0.05 + 0.05 * (c5 - t_stat) / (c5 - c10)
    else:
        # Above 10 % critical value → clearly non-stationary
        # (p-value is > 0.10; returning 0.50 ensures correct binary decisions)
        return 0.50
