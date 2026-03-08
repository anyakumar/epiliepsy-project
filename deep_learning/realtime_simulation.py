"""
Real-Time Seizure Detection Simulation
---------------------------------------
Simulates real-time EEG monitoring using a sliding window approach.

Concept:
  - In a real clinical system, EEG data is streamed continuously.
  - We use a sliding window (e.g., 178 points = 1 second) with a
    configurable stride to simulate continuous monitoring.
  - Each window is preprocessed, fed to the model, and a
    prediction + confidence is produced.
  - Results are visualized as a timeline with seizure alerts.

This module works with any trained model (ML or DL).
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from . import config
from .signal_processing import filter_eeg_signals


def simulate_realtime_stream(signal: np.ndarray, window_size: int = None,
                             stride: int = None) -> list:
    """
    Generate sliding windows from a long EEG signal.

    Parameters
    ----------
    signal      : 1-D array (long continuous EEG stream, or concatenated segments)
    window_size : Number of samples per window (default 178)
    stride      : Step size between windows (default 50)

    Returns
    -------
    windows : list of 1-D arrays, each of length window_size
    """
    window_size = window_size or config.WINDOW_SIZE
    stride = stride or config.STRIDE

    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start:start + window_size])
    return windows


def run_realtime_simulation(model, scaler, signal: np.ndarray,
                            window_size: int = None, stride: int = None,
                            apply_filter: bool = True,
                            model_type: str = "dl",
                            simulate_delay: bool = True) -> dict:
    """
    Run sliding window inference to simulate real-time monitoring.

    Parameters
    ----------
    model          : Trained model (Keras for DL, sklearn for ML)
    scaler         : Fitted StandardScaler
    signal         : 1-D array - continuous EEG stream
    window_size    : Sliding window size
    stride         : Step size
    apply_filter   : Whether to filter each window
    model_type     : 'dl' (Keras) or 'ml' (sklearn)
    simulate_delay : Add small delay to mimic real-time

    Returns
    -------
    results : dict with 'predictions', 'confidences', 'timestamps', 'alerts'
    """
    window_size = window_size or config.WINDOW_SIZE
    stride = stride or config.STRIDE

    windows = simulate_realtime_stream(signal, window_size, stride)
    print(f"[RT-SIM] Generated {len(windows)} windows from signal of length {len(signal)}")

    predictions = []
    confidences = []
    timestamps = []
    alerts = []
    latencies = []

    for i, window in enumerate(windows):
        start_time = time.time()

        # Time stamp (center of window)
        t_sec = (i * stride + window_size / 2) / config.SAMPLING_RATE
        timestamps.append(t_sec)

        # Preprocess
        w = window.copy().reshape(1, -1)
        if apply_filter:
            w = filter_eeg_signals(w)
        w_scaled = scaler.transform(w)

        # Predict
        if model_type == "dl":
            w_input = w_scaled.reshape(1, window_size, 1)
            prob = model.predict(w_input, verbose=0)
            if prob.shape[-1] == 1:
                seizure_prob = float(prob[0, 0])
            else:
                seizure_prob = float(prob[0, 1])
            pred = 1 if seizure_prob > 0.5 else 0
            conf = seizure_prob if pred == 1 else 1.0 - seizure_prob
        else:
            pred = model.predict(w_scaled)[0]
            try:
                prob = model.predict_proba(w_scaled)
                seizure_prob = float(prob[0, 1])
                conf = seizure_prob if pred == 1 else 1.0 - seizure_prob
            except AttributeError:
                conf = 1.0
                seizure_prob = float(pred)

        elapsed = time.time() - start_time
        latencies.append(elapsed)

        predictions.append(int(pred))
        confidences.append(float(conf))

        if pred == 1:
            alerts.append({
                "window_idx": i,
                "time_sec": round(t_sec, 3),
                "confidence": round(conf, 4),
            })

        if simulate_delay:
            time.sleep(0.01)  # 10ms delay to simulate real-time

        # Progress update every 50 windows
        if (i + 1) % 50 == 0:
            print(f"  [RT-SIM] Processed {i + 1}/{len(windows)} windows "
                  f"(avg latency: {np.mean(latencies[-50:])*1000:.1f} ms)")

    avg_latency = np.mean(latencies) * 1000
    print(f"\n[RT-SIM] Done! {len(windows)} windows processed.")
    print(f"[RT-SIM] Seizure alerts: {len(alerts)}/{len(windows)} windows")
    print(f"[RT-SIM] Avg inference latency: {avg_latency:.2f} ms/window")

    return {
        "predictions": predictions,
        "confidences": confidences,
        "timestamps": timestamps,
        "alerts": alerts,
        "latencies": latencies,
        "avg_latency_ms": avg_latency,
    }


def plot_realtime_results(results: dict, title: str = "Real-Time Seizure Detection"):
    """
    Visualize the sliding window simulation results as a timeline.
    """
    timestamps = np.array(results["timestamps"])
    predictions = np.array(results["predictions"])
    confidences = np.array(results["confidences"])

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

    # Prediction timeline
    colors = ["green" if p == 0 else "red" for p in predictions]
    axes[0].scatter(timestamps, predictions, c=colors, s=15, alpha=0.7)
    axes[0].set_ylabel("Prediction")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Non-Seizure", "Seizure"])
    axes[0].set_title(f"{title} — Predictions")
    axes[0].grid(True, alpha=0.3)

    # Highlight seizure regions
    for alert in results["alerts"]:
        axes[0].axvline(x=alert["time_sec"], color="red", alpha=0.3, linewidth=0.5)

    # Confidence timeline
    axes[1].plot(timestamps, confidences, color="blue", linewidth=0.8, alpha=0.7)
    axes[1].fill_between(timestamps, 0, confidences, alpha=0.2, color="blue")
    axes[1].axhline(y=0.5, color="gray", linestyle="--", label="Decision Threshold")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Confidence")
    axes[1].set_title(f"{title} — Confidence Scores")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "realtime_simulation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Real-time simulation plot saved to {path}")
