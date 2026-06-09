"""Triple-Barrier Labeling per López de Prado (2018) Advances in Financial ML, Ch.3.

Three barriers for each observation:
- Upper (profit-taking):  price * (1 + pt_factor * vol)
- Lower (stop-loss):      price * (1 - sl_factor * vol)
- Vertical (expiration):   t + max_hold

The label is determined by the first barrier touched:
    +1  → upper barrier hit first
    -1  → lower barrier hit first
     0  → vertical barrier expires without touching either side

The touch time t1 is stored for use in sample weighting (e.g., average uniqueness).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """Compute daily volatility as exponentially weighted moving standard deviation.

    Uses daily simple returns: r_t = (close_t / close_{t-1}) - 1.
    The first ``span`` observations will have NaN volatility until enough
    data accumulates.

    Parameters
    ----------
    close : pd.Series
        Price series (e.g., daily close). Index can be datetime or integer.
    span : int, default 100
        Span for the exponential moving average.  Larger values produce
        smoother estimates that adapt more slowly to regime changes.

    Returns
    -------
    vol : pd.Series
        Rolling daily volatility estimates, same index as ``close``.
        Values near the start are NaN.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    returns = close.pct_change()
    vol = returns.ewm(span=span).std()

    # Ensure no zero or negative vol that would break barrier construction
    vol = vol.clip(lower=1e-8)

    return vol


def get_events(
    close: pd.Series,
    vol: pd.Series,
    pt_factor: float = 2.0,
    sl_factor: float = 1.0,
    max_hold: int = 20,
    min_return: float = 0.0,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Generate triple-barrier events from a price series.

    For each observation at time t with a valid volatility estimate, three
    barriers are constructed:

        upper  = close_t * (1 + pt_factor * vol_t)
        lower  = close_t * (1 - sl_factor * vol_t)
        expiry = t + max_hold  (vertical barrier)

    The price path is tracked forward from t+1 to expiry.  The label is +1
    if the upper barrier is breached first, -1 if the lower barrier is
    breached first, and 0 if neither is breached by expiry.  The touch
    time t1 records the timestamp of the first barrier breach (or the
    expiry timestamp when no breach occurs).

    Parameters
    ----------
    close : pd.Series
        Price series.  Must share the same index as ``vol``.
    vol : pd.Series
        Rolling daily volatility estimates (see :func:`get_daily_vol`).
    pt_factor : float, default 2.0
        Profit-taking multiplier on volatility.  A value of 2.0 means the
        upper barrier is set at close * (1 + 2 * vol), i.e. a 2-sigma move.
    sl_factor : float, default 1.0
        Stop-loss multiplier on volatility.  A value of 1.0 means the
        lower barrier is set at close * (1 - 1 * vol), i.e. a 1-sigma move.
    max_hold : int, default 20
        Maximum holding period in time steps (days).  The vertical barrier
        is set at t + max_hold.
    min_return : float, default 0.0
        Minimum absolute return required for the vertical-barrier case to
        be labelled 0.  When |return| < min_return the observation may be
        considered uninformative and is excluded.
    side : pd.Series, optional
        Pre-determined side prediction (1 for long, -1 for short).  When
        provided, only the barrier in the direction of the side is active
        (asymmetric barriers), and labels become {0, 1} for meta-labeling.
        If None, symmetric triple-barrier is applied with labels {-1, 0, 1}.

    Returns
    -------
    events : pd.DataFrame
        Table with columns:

        - ``t0``  : event start timestamp (from ``close.index``)
        - ``t1``  : timestamp of first barrier touch (or expiry)
        - ``label``: +1 / -1 / 0 (or +1/0 when ``side`` is provided)
        - ``return``: close_{t1} / close_{t0} - 1
        - ``type`` : 'upper', 'lower', or 'vertical' indicating which
          barrier determined the outcome

    Notes
    -----
    - Observations where ``vol`` is NaN or non-positive are skipped.
    - Observations with fewer than 2 forward bars are skipped.
    - The ``t1`` column can be used with :func:`get_bins` to align labels
      to the original close index, and for sample-weighting schemes such as
      average uniqueness.
    """
    # --- Input validation ---------------------------------------------------
    if len(close) != len(vol):
        raise ValueError(
            f"close and vol must have the same length: "
            f"{len(close)} vs {len(vol)}"
        )

    close_vals = close.values.astype(np.float64)
    vol_vals = vol.values.astype(np.float64)
    idx = close.index
    n = len(close_vals)

    if side is not None:
        if len(side) != n:
            raise ValueError(
                f"side must have same length as close: {len(side)} vs {n}"
            )
        side_vals = side.values.astype(np.float64)
        meta_labeling = True
    else:
        meta_labeling = False

    # Pre-compute forward return matrix (n × max_hold)
    # ret_matrix[i, d] = close[i+1+d] / close[i] - 1  for d in 0..max_hold-1
    # This trades memory for speed: O(n × max_hold) memory.
    # For typical series (10k bars × 20 horizons) this is ~1.6 MB.
    ret_matrix = np.full((n, max_hold), np.nan, dtype=np.float64)
    for d in range(max_hold):
        fut = close_vals[d + 1 :]
        cur = close_vals[: n - d - 1]
        valid = (cur > 0) & ~np.isnan(cur) & ~np.isnan(fut)
        ret_vals = np.full(n, np.nan, dtype=np.float64)
        ret_vals[: n - d - 1] = np.where(valid, fut / cur - 1.0, np.nan)
        ret_matrix[:, d] = ret_vals

    # Barrier thresholds per observation
    upper_thresh = pt_factor * vol_vals  # return needed to hit upper barrier
    lower_thresh = -sl_factor * vol_vals  # return needed to hit lower barrier

    # --- Event detection loop ------------------------------------------------
    records: list[dict] = []

    for i in range(n):
        # Skip invalid entries
        if np.isnan(vol_vals[i]) or vol_vals[i] <= 0:
            continue
        if np.isnan(close_vals[i]) or close_vals[i] <= 0:
            continue

        # Available forward horizon (may be truncated near series end)
        remaining = n - i - 1
        if remaining < 1:
            continue
        horizon = min(max_hold, remaining)

        # ---- Meta-labeling mode: only one active barrier --------------------
        if meta_labeling:
            s = side_vals[i]
            if np.isnan(s) or s == 0:
                continue  # no side prediction → skip

            future_rets = ret_matrix[i, :horizon]  # shape (horizon,)

            if s > 0:  # long — only upper (PT) barrier matters
                upper_hit = np.where(
                    ~np.isnan(future_rets) & (future_rets >= upper_thresh[i])
                )[0]
                lower_hit = np.array([], dtype=np.int64)
            else:  # short — only lower (SL) barrier matters
                upper_hit = np.array([], dtype=np.int64)
                lower_hit = np.where(
                    ~np.isnan(future_rets) & (future_rets <= lower_thresh[i])
                )[0]
        # ---- Standard symmetric triple-barrier ------------------------------
        else:
            future_rets = ret_matrix[i, :horizon]

            upper_hit = np.where(
                ~np.isnan(future_rets) & (future_rets >= upper_thresh[i])
            )[0]
            lower_hit = np.where(
                ~np.isnan(future_rets) & (future_rets <= lower_thresh[i])
            )[0]

        # Determine first touch
        has_upper = len(upper_hit) > 0
        has_lower = len(lower_hit) > 0

        if not has_upper and not has_lower:
            # Vertical barrier — no side barrier breached
            t1_pos = i + horizon
            label = 0
            barrier_type = "vertical"
        elif has_upper and not has_lower:
            t1_pos = i + 1 + int(upper_hit[0])
            label = 1
            barrier_type = "upper"
        elif has_lower and not has_upper:
            t1_pos = i + 1 + int(lower_hit[0])
            label = -1
            barrier_type = "lower"
        else:
            # Both hit — take the one that occurs first
            if upper_hit[0] < lower_hit[0]:
                t1_pos = i + 1 + int(upper_hit[0])
                label = 1
                barrier_type = "upper"
            elif lower_hit[0] < upper_hit[0]:
                t1_pos = i + 1 + int(lower_hit[0])
                label = -1
                barrier_type = "lower"
            else:
                # Simultaneous hit (same time step) — extremely rare
                t1_pos = i + 1 + int(upper_hit[0])
                label = 0
                barrier_type = "vertical"

        # Compute actual return
        ret = close_vals[t1_pos] / close_vals[i] - 1.0

        # Apply minimum return filter for vertical barrier
        if barrier_type == "vertical" and abs(ret) < min_return:
            continue

        # Meta-labeling: convert {-1, 0, 1} → {0, 1}
        if meta_labeling:
            # label = 1 if the active barrier was hit before vertical expiry
            label = 1 if barrier_type != "vertical" else 0

        records.append({
            "t0": idx[i],
            "t1": idx[t1_pos],
            "label": label,
            "return": ret,
            "type": barrier_type,
        })

    if not records:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=["t0", "t1", "label", "return", "type"])

    events = pd.DataFrame(records)
    # Ensure correct dtypes
    events["label"] = events["label"].astype(np.int8)
    return events


def get_bins(events: pd.DataFrame, close: pd.Series) -> np.ndarray:
    """Convert triple-barrier events into label bins aligned to the close index.

    Produces a 1-D numpy array of the same length as ``close``, where each
    position is either the triple-barrier label (when an event was triggered
    at that timestamp) or NaN (for non-event indices).

    The output is suitable as a target vector for supervised learning:
    non-NaN entries are training samples; NaN entries are ignored.

    Parameters
    ----------
    events : pd.DataFrame
        Events table produced by :func:`get_events`.  Must contain at least
        the columns ``t0`` and ``label``.
    close : pd.Series
        Original price series used to generate the events.  Only its index
        is used for alignment.

    Returns
    -------
    bins : np.ndarray, shape (len(close),)
        Triple-barrier labels.  ``bins[i]`` contains the label for the
        event triggered at ``close.index[i]``, or NaN if no event was
        generated for that timestamp.

    Raises
    ------
    KeyError
        If ``events`` does not contain the required columns ``t0`` or ``label``.
    """
    if "t0" not in events.columns:
        raise KeyError("events DataFrame must contain a 't0' column")
    if "label" not in events.columns:
        raise KeyError("events DataFrame must contain a 'label' column")

    bins = np.full(len(close), np.nan, dtype=np.float64)

    # Build a fast lookup from timestamp → positional index
    index_map = {ts: i for i, ts in enumerate(close.index)}

    t0_vals = events["t0"].values
    label_vals = events["label"].values

    for j in range(len(events)):
        ts = t0_vals[j]
        pos = index_map.get(ts)
        if pos is not None:
            bins[pos] = label_vals[j]

    return bins
