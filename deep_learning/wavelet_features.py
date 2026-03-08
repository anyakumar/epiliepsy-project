"""
Wavelet Feature Extraction Module
-----------------------------------
Applies Discrete Wavelet Transform (DWT) using Daubechies-4 (db4)
and extracts statistical features from each decomposition level.

Features extracted per sub-band:
  - Mean, Std, Min, Max
  - Energy (sum of squares)
  - Entropy (Shannon)
  - Kurtosis, Skewness

This creates a rich feature vector capturing frequency-band information
that is highly relevant for EEG seizure detection.
"""

import numpy as np
import pywt
from scipy import stats as sp_stats

from . import config


def dwt_decompose(signal: np.ndarray, wavelet: str = None,
                  level: int = None) -> list:
    """
    Perform multi-level DWT decomposition on a 1-D signal.

    Parameters
    ----------
    signal  : 1-D array of EEG data points
    wavelet : Wavelet name (default: config.WAVELET_NAME = 'db4')
    level   : Decomposition level (default: config.WAVELET_LEVEL = 4)

    Returns
    -------
    coeffs : list of ndarray
        [cA_n, cD_n, cD_n-1, ..., cD_1]
        Approximation coefficients at level n, then detail coefficients
        from level n down to level 1.
    """
    wavelet = wavelet or config.WAVELET_NAME
    level = level or config.WAVELET_LEVEL
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs


def compute_subband_features(coeffs: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from a single sub-band (array of coefficients).

    Returns an array of 8 features:
      [mean, std, min, max, energy, entropy, kurtosis, skewness]
    """
    if len(coeffs) == 0:
        return np.zeros(8)

    mean = np.mean(coeffs)
    std = np.std(coeffs)
    minimum = np.min(coeffs)
    maximum = np.max(coeffs)
    energy = np.sum(coeffs ** 2)

    # Shannon entropy (using normalized squared coefficients as probabilities)
    sq = coeffs ** 2
    total = np.sum(sq)
    if total > 0:
        prob = sq / total
        prob = prob[prob > 0]  # Avoid log(0)
        entropy = -np.sum(prob * np.log2(prob))
    else:
        entropy = 0.0

    kurtosis = sp_stats.kurtosis(coeffs)
    skewness = sp_stats.skew(coeffs)

    return np.array([mean, std, minimum, maximum, energy, entropy, kurtosis, skewness])


def extract_wavelet_features(signal: np.ndarray, wavelet: str = None,
                             level: int = None) -> np.ndarray:
    """
    Full wavelet feature extraction for a single EEG segment.

    Steps:
      1. Decompose signal into sub-bands via DWT
      2. Extract 8 statistical features from each sub-band
      3. Concatenate into a single feature vector

    For level=4, DWT produces 5 sub-bands (1 approx + 4 detail),
    yielding 5 × 8 = 40 wavelet features.

    Parameters
    ----------
    signal  : 1-D array (178 data points)
    wavelet : Wavelet name
    level   : Decomposition level

    Returns
    -------
    features : 1-D array of shape (n_subbands * 8,)
    """
    coeffs = dwt_decompose(signal, wavelet, level)
    features = []
    for coeff in coeffs:
        subband_feat = compute_subband_features(coeff)
        features.append(subband_feat)
    return np.concatenate(features)


def extract_wavelet_features_batch(signals: np.ndarray, wavelet: str = None,
                                   level: int = None) -> np.ndarray:
    """
    Extract wavelet features for a batch of EEG segments.

    Parameters
    ----------
    signals : ndarray, shape (n_samples, n_features)
    wavelet : Wavelet name
    level   : Decomposition level

    Returns
    -------
    feature_matrix : ndarray, shape (n_samples, n_wavelet_features)
    """
    feature_list = []
    for i in range(signals.shape[0]):
        feat = extract_wavelet_features(signals[i], wavelet, level)
        feature_list.append(feat)
    return np.array(feature_list)


def get_wavelet_feature_names(level: int = None) -> list:
    """
    Generate descriptive feature names for the wavelet features.

    Returns
    -------
    names : list of str, e.g. ['cA4_mean', 'cA4_std', ..., 'cD1_skewness']
    """
    level = level or config.WAVELET_LEVEL
    stat_names = ["mean", "std", "min", "max", "energy", "entropy", "kurtosis", "skewness"]

    names = []
    # First element is approximation at level n
    for stat in stat_names:
        names.append(f"cA{level}_{stat}")
    # Remaining are detail coefficients from level n down to 1
    for lv in range(level, 0, -1):
        for stat in stat_names:
            names.append(f"cD{lv}_{stat}")
    return names


# ─── Quick demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Demo with random signal
    np.random.seed(42)
    sample_signal = np.random.randn(178)

    features = extract_wavelet_features(sample_signal)
    names = get_wavelet_feature_names()

    print(f"Number of wavelet features: {len(features)}")
    print(f"Feature names ({len(names)}):")
    for name, val in zip(names, features):
        print(f"  {name:20s} = {val:.4f}")
