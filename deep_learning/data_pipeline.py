"""
Data Pipeline Module
--------------------
Handles:
  1. Loading the UCI Epileptic Seizure Recognition dataset
  2. Binary & multi-class label encoding
  3. Signal filtering (optional)
  4. Wavelet feature extraction (optional)
  5. Standard scaling
  6. SMOTE oversampling for class imbalance
  7. Train / Validation / Test splitting
  8. Reshaping for CNN and LSTM inputs
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

from . import config
from .signal_processing import filter_eeg_signals
from .wavelet_features import extract_wavelet_features_batch


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_dataset(path: str = None) -> tuple:
    """
    Load the UCI Epileptic Seizure Recognition CSV.

    Returns
    -------
    X : ndarray, shape (11500, 178) — raw EEG signal values
    y : ndarray, shape (11500,) — labels 1–5
    """
    path = path or config.DATA_PATH
    print(f"[DATA] Loading dataset from {path} ...")
    df = pd.read_csv(path)

    # Drop unnamed index column if present
    if "Unnamed" in str(df.columns[0]) or df.columns[0].lower() in ("x", "unnamed: 0"):
        df = df.iloc[:, 1:]

    X = df.iloc[:, :-1].values.astype(np.float64)   # (11500, 178)
    y = df.iloc[:, -1].values.astype(int)           # (11500,)

    print(f"[DATA] Loaded: X={X.shape}, y={y.shape}")
    print(f"[DATA] Class distribution: { {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))} }")
    return X, y


# ─── Label Encoding ─────────────────────────────────────────────────────────

def binarize_labels(y: np.ndarray) -> np.ndarray:
    """Convert 5-class labels to binary: 1 → Seizure (1), 2-5 → Non-Seizure (0)."""
    return (y == 1).astype(int)


# ─── Full Pipeline ───────────────────────────────────────────────────────────

def prepare_data(
    apply_filter: bool = True,
    apply_wavelet: bool = False,
    apply_smote: bool = True,
    binary: bool = True,
    test_size: float = None,
    val_size: float = None,
    random_state: int = None,
) -> dict:
    """
    End-to-end data preparation pipeline.

    Parameters
    ----------
    apply_filter  : Apply EEG band-pass + notch filtering
    apply_wavelet : Extract wavelet features (replaces raw signal with wavelet features)
    apply_smote   : Apply SMOTE oversampling to training set
    binary        : Binary classification (seizure vs non-seizure)
    test_size     : Fraction for test set
    val_size      : Fraction of training set for validation
    random_state  : Random seed

    Returns
    -------
    data : dict with keys:
        'X_train', 'X_val', 'X_test',
        'y_train', 'y_val', 'y_test',
        'scaler', 'feature_type'
    """
    test_size = test_size or config.TEST_SIZE
    val_size = val_size or config.VAL_SIZE
    random_state = random_state or config.RANDOM_STATE

    # 1. Load
    X, y = load_dataset()

    # 2. Label encoding
    if binary:
        y = binarize_labels(y)
        print(f"[DATA] Binary labels: { {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))} }")

    # 3. Signal filtering
    if apply_filter:
        print("[DATA] Applying EEG signal filtering (notch + band-pass) ...")
        X = filter_eeg_signals(X)
        print("[DATA] Filtering complete.")

    # 4. Wavelet feature extraction (optional — used for ML models or ablation)
    feature_type = "raw"
    if apply_wavelet:
        print("[DATA] Extracting wavelet features ...")
        X = extract_wavelet_features_batch(X)
        feature_type = "wavelet"
        print(f"[DATA] Wavelet features shape: {X.shape}")

    # 5. Train / Test split (stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6. Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size,
        random_state=random_state, stratify=y_train_full
    )

    # 7. SMOTE — only on training set to avoid data leakage
    if apply_smote:
        print(f"[DATA] Before SMOTE — Train: { {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))} }")
        smote = SMOTE(random_state=config.SMOTE_RANDOM_STATE,
                      k_neighbors=config.SMOTE_K_NEIGHBORS)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"[DATA] After  SMOTE — Train: { {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))} }")

    # 8. Standard scaling
    print("[DATA] Scaling features ...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"[DATA] Final shapes — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_type": feature_type,
    }


# ─── Reshaping Utilities for DL ─────────────────────────────────────────────

def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    """
    Reshape to (samples, timesteps, channels=1) for Conv1D input.
    """
    return X.reshape(X.shape[0], X.shape[1], 1)


def reshape_for_lstm(X: np.ndarray, timesteps: int = None) -> np.ndarray:
    """
    Reshape to (samples, timesteps, features) for LSTM input.

    If the feature count is not divisible by timesteps, we pad with zeros.
    Default: treat the full signal as a single sequence of length n_features.
    """
    if timesteps is None:
        # Treat full signal as one sequence: (samples, n_features, 1)
        return X.reshape(X.shape[0], X.shape[1], 1)
    else:
        n_features = X.shape[1]
        feat_per_step = n_features // timesteps
        remainder = n_features % timesteps
        if remainder != 0:
            # Pad
            pad_width = timesteps - remainder
            X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant")
            feat_per_step = X.shape[1] // timesteps
        return X.reshape(X.shape[0], timesteps, feat_per_step)


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = prepare_data(apply_filter=True, apply_wavelet=False,
                        apply_smote=True, binary=True)
    print("\n✅ Data pipeline test passed!")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")
