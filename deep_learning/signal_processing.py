"""
EEG Signal Processing Module
-----------------------------
Implements:
  1. Band-pass filter (0.5–40 Hz) to retain relevant EEG frequency bands
  2. Notch filter (50/60 Hz) to remove powerline interference
  3. Combined filtering pipeline

Uses scipy.signal for Butterworth IIR filter design.
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from . import config


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """
    Design a Butterworth band-pass filter.

    Parameters
    ----------
    lowcut  : Lower cutoff frequency (Hz)
    highcut : Upper cutoff frequency (Hz)
    fs      : Sampling frequency (Hz)
    order   : Filter order (default 4)

    Returns
    -------
    b, a : ndarray
        Numerator and denominator polynomial coefficients of the IIR filter.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(signal: np.ndarray, fs: float = None,
                          lowcut: float = None, highcut: float = None,
                          order: int = None) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter to a 1-D or 2-D signal.

    Parameters
    ----------
    signal  : ndarray, shape (n_samples,) or (n_rows, n_samples)
    fs      : Sampling rate in Hz (default from config)
    lowcut  : Low cutoff Hz (default from config)
    highcut : High cutoff Hz (default from config)
    order   : Filter order (default from config)

    Returns
    -------
    filtered : ndarray, same shape as input
    """
    fs = fs or config.SAMPLING_RATE
    lowcut = lowcut or config.BANDPASS_LOW
    highcut = highcut or config.BANDPASS_HIGH
    order = order or config.BANDPASS_ORDER

    b, a = butter_bandpass(lowcut, highcut, fs, order)

    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        # Apply row-wise (each row is one EEG segment)
        return np.array([filtfilt(b, a, row) for row in signal])


def apply_notch_filter(signal: np.ndarray, fs: float = None,
                       freq: float = None, quality: float = None) -> np.ndarray:
    """
    Apply a notch (band-stop) filter to remove powerline noise.

    Parameters
    ----------
    signal  : ndarray, shape (n_samples,) or (n_rows, n_samples)
    fs      : Sampling rate in Hz
    freq    : Frequency to remove (50 or 60 Hz)
    quality : Quality factor Q (higher = narrower notch)

    Returns
    -------
    filtered : ndarray, same shape as input
    """
    fs = fs or config.SAMPLING_RATE
    freq = freq or config.NOTCH_FREQ
    quality = quality or config.NOTCH_QUALITY

    b, a = iirnotch(freq, quality, fs)

    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        return np.array([filtfilt(b, a, row) for row in signal])


def filter_eeg_signals(signals: np.ndarray, fs: float = None,
                       apply_notch: bool = True,
                       apply_bandpass: bool = True) -> np.ndarray:
    """
    Full filtering pipeline: notch → band-pass.

    Applying notch first removes the powerline spike before
    the band-pass filter shapes the overall frequency content.

    Parameters
    ----------
    signals       : ndarray, shape (n_rows, n_samples) or (n_samples,)
    fs            : Sampling rate
    apply_notch   : Whether to apply the notch filter
    apply_bandpass: Whether to apply the band-pass filter

    Returns
    -------
    filtered : ndarray, same shape as input
    """
    fs = fs or config.SAMPLING_RATE
    filtered = signals.copy().astype(np.float64)

    if apply_notch:
        filtered = apply_notch_filter(filtered, fs=fs)

    if apply_bandpass:
        filtered = apply_bandpass_filter(filtered, fs=fs)

    return filtered


# ─── Quick demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate a synthetic noisy EEG-like signal
    np.random.seed(42)
    fs = config.SAMPLING_RATE
    t = np.arange(0, 1, 1 / fs)                  # 1 second
    clean = 5 * np.sin(2 * np.pi * 10 * t)       # 10 Hz alpha wave
    noise_50 = 2 * np.sin(2 * np.pi * 50 * t)    # 50 Hz powerline
    high_noise = 0.5 * np.random.randn(len(t))    # Gaussian noise
    raw = clean + noise_50 + high_noise

    filtered = filter_eeg_signals(raw, fs=fs)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(t, raw, "k", alpha=0.7)
    axes[0].set_title("Raw Signal (with 50 Hz noise)")
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(t, filtered, "b", alpha=0.7)
    axes[1].set_title("Filtered Signal (notch + band-pass)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "signal_filtering_demo.png"), dpi=150)
    plt.show()
    print("Signal filtering demo complete.")
