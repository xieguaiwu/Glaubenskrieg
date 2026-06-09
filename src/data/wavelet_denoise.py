"""Wavelet denoising for financial time series.

Implements the method from Bai et al. (Symmetry 2026):
- One-sided (causal) wavelet denoising on close prices
- db4 wavelet, level 3 decomposition
- Soft thresholding with universal threshold σ√(2logN)
- Applied before feature engineering to remove microstructure noise
"""

import numpy as np

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    threshold_mode: str = "soft",
) -> np.ndarray:
    """Apply wavelet denoising to a 1D signal.

    Parameters
    ----------
    signal : (T,) array
        Raw price signal (close prices).
    wavelet : str
        Wavelet name. db4 is recommended for financial data.
    level : int
        Decomposition level. 3 is recommended for daily data.
    threshold_mode : str
        "soft" or "hard" thresholding.

    Returns
    -------
    denoised : (T,) array, same length as input
    """
    if not HAS_PYWT:
        raise ImportError("pywt required. Install: pip install PyWavelets")

    T = len(signal)
    if T < 2 ** (level + 1):
        level = max(1, int(np.log2(T)) - 1)

    # Decompose
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs[0] = approximation, coeffs[1:] = detail coefficients per level

    # Universal threshold: σ * sqrt(2 * log(N))
    # Estimate noise std from the finest detail coefficients (highest frequency)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(T))

    # Apply threshold to all detail coefficients
    for i in range(1, len(coeffs)):
        if threshold_mode == "soft":
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode="soft")
        else:
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode="hard")

    # Reconstruct
    denoised = pywt.waverec(coeffs, wavelet)

    # Trim to original length (waverec may return slightly longer)
    return denoised[:T]


def causal_wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    min_window: int = 252,
) -> np.ndarray:
    """Causal (one-sided) wavelet denoising.

    Each point uses only past data for denoising — no look-ahead.
    For early timesteps, uses expanding window until min_window is reached.

    Parameters
    ----------
    signal : (T,) array
    wavelet : str
    level : int
    min_window : int
        Minimum lookback for stable wavelet decomposition.

    Returns
    -------
    denoised : (T,) array
    """
    T = len(signal)
    denoised = np.full(T, np.nan)

    for t in range(min_window, T):
        window = signal[: t + 1]
        try:
            denoised[t] = wavelet_denoise(window, wavelet, level)[-1]
        except Exception:
            denoised[t] = signal[t]

    # For early timesteps, use expanding-window EMA as fallback
    early_mask = np.isnan(denoised)
    if early_mask.any():
        alpha = 2 / (min_window + 1)
        ema = signal[0]
        for t in range(T):
            ema = alpha * signal[t] + (1 - alpha) * ema
            if early_mask[t]:
                denoised[t] = ema

    return denoised


def compute_denoised_returns(
    close_prices: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    min_window: int = 252,
) -> np.ndarray:
    """Compute forward returns using wavelet-denoised prices.

    Parameters
    ----------
    close_prices : (T,) array
    wavelet : str
    level : int
    min_window : int

    Returns
    -------
    returns : (T,) array — forward returns based on denoised prices
    """
    denoised = causal_wavelet_denoise(close_prices, wavelet, level, min_window)

    # Forward return: (price[t+1] - price[t]) / price[t]
    # Use denoised prices for better signal-to-noise ratio
    returns = np.full(len(close_prices), np.nan)
    for t in range(len(close_prices) - 1):
        if denoised[t] > 0:
            returns[t] = (denoised[t + 1] - denoised[t]) / denoised[t]

    return returns
