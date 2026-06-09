"""Feature engineering for stock prediction.

Technical indicators computed from OHLCV data following the CTM Architecture Guide.
All computations are causal (no look-ahead bias).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def compute_returns(prices: torch.Tensor, log_returns: bool = False) -> torch.Tensor:
    """Compute simple or log returns.

    Parameters
    ----------
    prices : (N,) or (N, 1) close prices.
    log_returns : if True, compute log returns.

    Returns
    -------
    (N-1,) returns tensor. First element is padded with 0.
    """
    prices = prices.flatten()
    ret = prices[1:] / prices[:-1] - 1.0 if not log_returns else torch.log(prices[1:] / prices[:-1])
    return torch.cat([torch.zeros(1, device=prices.device, dtype=prices.dtype), ret])


def compute_forward_returns(close: torch.Tensor, periods: int = 1) -> torch.Tensor:
    """Compute forward returns: (close[t+periods] / close[t] - 1).

    For time step t, forward_return[t] = (close[t+periods] / close[t] - 1).
    The last ``periods`` elements are padded with 0 to maintain the same
    length as the input series.

    Parameters
    ----------
    close : (N,) or (N, 1) close prices.
    periods : forward horizon in time steps (default 1).

    Returns
    -------
    (N,) forward returns tensor for each input price.
    """
    close = close.flatten()
    ret = close[periods:] / close[:-periods] - 1.0
    return torch.cat([ret, torch.zeros(periods, device=close.device, dtype=close.dtype)])


def compute_sma(prices: torch.Tensor, period: int) -> torch.Tensor:
    """Simple moving average (causal)."""
    prices = prices.flatten()
    N = len(prices)
    kernel = torch.ones(1, 1, period, device=prices.device, dtype=prices.dtype) / period
    x = prices.view(1, 1, -1)
    padded = torch.nn.functional.pad(x, (period - 1, 0))
    sma = torch.nn.functional.conv1d(padded, kernel).flatten()
    cumsum = torch.cumsum(prices, dim=0)
    early = min(period - 1, N)
    sma[:early] = cumsum[:early] / torch.arange(1, early + 1, device=prices.device, dtype=prices.dtype)
    return sma


def compute_rsi(prices: torch.Tensor, period: int = 14) -> torch.Tensor:
    """Relative Strength Index (causal, Wilder's smoothing)."""
    prices = prices.flatten()
    n = len(prices)
    if n < period + 1:
        return torch.zeros(n)

    returns = prices[1:] - prices[:-1]
    gain = torch.clamp(returns, min=0)
    loss = torch.clamp(-returns, min=0)

    N = len(returns)
    alpha = 1.0 / period
    avg_gain = torch.zeros(N)
    avg_loss = torch.zeros(N)
    avg_gain[period - 1] = gain[:period].mean()
    avg_loss[period - 1] = loss[:period].mean()

    for i in range(period, N):
        avg_gain[i] = avg_gain[i - 1] + alpha * (gain[i] - avg_gain[i - 1])
        avg_loss[i] = avg_loss[i - 1] + alpha * (loss[i] - avg_loss[i - 1])

    rsi = torch.zeros(n)
    rs = avg_gain / (avg_loss + 1e-8)
    # First valid RSI computed after `period` returns → at price index `period`
    rsi[period:] = 100.0 - 100.0 / (rs[period - 1:] + 1.0)
    return rsi


def compute_bollinger_bands(
    prices: torch.Tensor, period: int = 20, std_dev: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bollinger Bands: middle (SMA), upper, lower."""
    sma = compute_sma(prices, period)
    prices = prices.flatten()
    N = len(prices)
    lower = torch.full((N,), float('nan'))
    upper = torch.full((N,), float('nan'))
    for t in range(N):
        start = max(0, t - period + 1)
        window = prices[start : t + 1]
        # Note: correction=0 (population std) is the conventional formula for Bollinger Bands
        std = window.std(correction=0)
        lower[t] = sma[t] - std_dev * std
        upper[t] = sma[t] + std_dev * std
    return sma, upper, lower


def compute_volume_ratio(volume: torch.Tensor, period: int = 20) -> torch.Tensor:
    """Volume ratio: V_t / SMA(V, period)."""
    vol_sma = compute_sma(volume, period)
    return volume.flatten() / (vol_sma + 1e-8)


def compute_realized_volatility(returns: torch.Tensor, period: int = 21) -> torch.Tensor:
    """Rolling standard deviation of returns (annualized not applied)."""
    ret = returns.flatten()
    N = len(ret)
    vol = torch.full((N,), float('nan'))
    for t in range(N):
        start = max(0, t - period + 1)
        window = ret[start : t + 1]
        vol[t] = window.std(correction=1 if len(window) > 1 else 0)
    return vol


def compute_all_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute all technical indicators from OHLCV DataFrame.

    Columns expected: close, volume (at minimum).

    Returns DataFrame with features indexed to match input:
      simple_return, log_return, sma_5, sma_20, rsi_14,
      bollinger_position, volume_ratio, realized_vol_21, bias.
    """
    if feature_cols is None:
        feature_cols = ["close", "volume"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    result = pd.DataFrame(index=df.index)
    close = torch.from_numpy(df[feature_cols[0]].values).float()

    result["simple_return"] = compute_returns(close, log_returns=False).numpy()
    result["log_return"] = compute_returns(close, log_returns=True).numpy()

    sma5 = compute_sma(close, 5)
    result["sma_5"] = (close / (sma5 + 1e-8) - 1.0).numpy()

    sma20 = compute_sma(close, 20)
    result["sma_20"] = (close / (sma20 + 1e-8) - 1.0).numpy()

    result["rsi_14"] = compute_rsi(close, 14).numpy()

    _, upper, lower = compute_bollinger_bands(close, 20, 2.0)
    bb_mid = sma20.numpy()
    bb_width = upper.numpy() - bb_mid  # = 2*std for valid entries
    # Explicit NaN guard: first (period-1) values are NaN (insufficient data);
    # also NaN for flat Bollinger bands (width ≈ 0 within window) — no meaningful position signal
    bb_std = np.where(np.isfinite(bb_width) & (np.abs(bb_width) > 1e-12), bb_width / 2.0, np.nan)
    result["bollinger_position"] = (close.numpy() - bb_mid) / (bb_std + 1e-8)

    if len(feature_cols) >= 2:
        volume = torch.from_numpy(df[feature_cols[1]].values).float()
        result["volume_ratio"] = compute_volume_ratio(volume, 20).numpy()

    result["realized_vol_21"] = compute_realized_volatility(
        torch.from_numpy(result["simple_return"].values).float(), 21
    ).numpy()

    result["bias"] = (close.numpy() - sma20.numpy()) / (sma20.numpy() + 1e-8)

    return result
