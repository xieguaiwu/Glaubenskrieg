"""Shared walk-forward window boundary helper.

Extracts the common window-slicing logic duplicated in
:class:`WalkForwardTrainer` and :class:`WalkForwardTrainerAdvanced`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class TrainingResult:
    """Per-window result for CTM + GBDT ensemble walk-forward training.

    Supports both CTM-only (non-ensemble) and ensemble training paths.
    Ensemble-specific fields (ctm_sharpe, gbdt_ic, etc.) are None when
    not applicable.
    """
    window_start: int = 0
    window_end: int = 0
    best_sharpe: float = 0.0
    epochs_run: int = 0
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    # Ensemble-specific optional fields
    ctm_sharpe: Optional[float] = None
    ctm_ic: Optional[float] = None
    ctm_dir_acc: Optional[float] = None
    gbdt_sharpe: Optional[float] = None
    gbdt_ic: Optional[float] = None
    gbdt_dir_acc: Optional[float] = None
    ensemble_sharpe: Optional[float] = None
    ensemble_ic: Optional[float] = None
    ensemble_dir_acc: Optional[float] = None
    ctm_weight: Optional[float] = None
    gbdt_weight: Optional[float] = None
    ctm_metrics: Optional[List[Dict[str, Any]]] = None
    gbdt_metrics: Optional[Dict[str, Any]] = None
    gbdt_importance: Optional[Dict[str, Any]] = None
    # P3 three-stage fusion fields
    p3_sharpe: Optional[float] = None
    p3_ic: Optional[float] = None
    p3_ensemble_sharpe: Optional[float] = None


def walk_forward_windows(
    N: int,
    train_window: int,
    val_window: int,
    purge_period: int,
    step_size: int,
) -> Iterator[Tuple[int, int, int, int]]:
    """Yield ``(pos, train_end, purge_end, val_end)`` for each window.

    Parameters
    ----------
    N : total number of samples.
    train_window, val_window : fold sizes in samples.
    purge_period : gap between train and val.
    step_size : walk-forward stride.

    Yields
    ------
    (pos, train_end, purge_end, val_end) for each valid window.
    """
    if train_window <= 0:
        raise ValueError(f"train_window must be > 0, got {train_window}")
    if val_window <= 0:
        raise ValueError(f"val_window must be > 0, got {val_window}")
    if purge_period >= val_window:
        import warnings
        warnings.warn(f"purge_period ({purge_period}) >= val_window ({val_window}) — no valid windows may be produced")

    pos = 0
    while pos + train_window + purge_period + val_window <= N:
        train_end = pos + train_window
        purge_end = train_end + purge_period
        val_end = purge_end + val_window
        yield pos, train_end, purge_end, val_end
        pos += step_size
