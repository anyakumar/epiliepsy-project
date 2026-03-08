"""
Configuration and hyperparameters for the deep learning pipeline.
All tunable values are centralized here for easy experimentation.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Create directories if they don't exist
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────────────────────────
NUM_FEATURES = 178          # Number of EEG signal data points per sample
SAMPLING_RATE = 178         # Hz (178 data points recorded in 1 second)
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1              # Fraction of training set used for validation

# ─── Signal Processing ───────────────────────────────────────────────────────
BANDPASS_LOW = 0.5          # Hz - lower cutoff for band-pass filter
BANDPASS_HIGH = 40.0        # Hz - upper cutoff for band-pass filter
BANDPASS_ORDER = 4          # Butterworth filter order
NOTCH_FREQ = 50.0           # Hz - powerline noise frequency (50 Hz EU / 60 Hz US)
NOTCH_QUALITY = 30.0        # Quality factor for notch filter

# ─── Wavelet ─────────────────────────────────────────────────────────────────
WAVELET_NAME = "db4"        # Daubechies-4 wavelet
WAVELET_LEVEL = 4           # Decomposition level

# ─── SMOTE ───────────────────────────────────────────────────────────────────
SMOTE_RANDOM_STATE = 42
SMOTE_K_NEIGHBORS = 5

# ─── 1D CNN Hyperparameters ──────────────────────────────────────────────────
CNN_FILTERS = [64, 128, 64]        # Filters per conv block
CNN_KERNEL_SIZE = 5                 # Kernel size for Conv1D
CNN_POOL_SIZE = 2                   # MaxPooling size
CNN_DENSE_UNITS = 128               # Dense layer before output
CNN_DROPOUT = 0.5                   # Dropout rate
CNN_LEARNING_RATE = 1e-3
CNN_BATCH_SIZE = 64
CNN_EPOCHS = 10                     # Set higher (e.g. 50) for full training
CNN_PATIENCE = 5                    # Early stopping patience

# ─── LSTM / BiLSTM Hyperparameters ───────────────────────────────────────────
LSTM_UNITS = [64, 32]              # Units per LSTM layer
LSTM_DENSE_UNITS = 64
LSTM_DROPOUT = 0.3
LSTM_RECURRENT_DROPOUT = 0.2
LSTM_LEARNING_RATE = 1e-3
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 10                    # Set higher (e.g. 50) for full training
LSTM_PATIENCE = 5
LSTM_BIDIRECTIONAL = True           # Use BiLSTM if True

# ─── Sliding Window (Real-time Simulation) ───────────────────────────────────
WINDOW_SIZE = 178                   # Same as sample length
STRIDE = 50                         # Step size for sliding window

# ─── Number of classes ───────────────────────────────────────────────────────
NUM_CLASSES_BINARY = 2              # Seizure vs Non-Seizure
NUM_CLASSES_MULTI = 5               # All 5 original classes
