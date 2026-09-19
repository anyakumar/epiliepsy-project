"""
Model Explainability Module
---------------------------
Provides:
  1. SHAP (SHapley Additive exPlanations) for ML and DL models
  2. Grad-CAM for the 1D CNN model

These help interpret *why* the model made a particular prediction,
which is critical for medical/clinical AI systems.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP Explainability
# ═══════════════════════════════════════════════════════════════════════════════

def explain_with_shap(model, X_sample: np.ndarray, feature_names: list = None,
                      model_name: str = "model", max_display: int = 20,
                      model_type: str = "tree"):
    """
    Generate SHAP explanations for a trained model.

    Parameters
    ----------
    model         : Trained sklearn/xgboost model or keras model
    X_sample      : Sample data for explanation, shape (n_samples, n_features)
    feature_names : List of feature names
    model_name    : For saving plots
    max_display   : Max features to show in summary plot
    model_type    : 'tree' for tree-based models (RF, XGBoost),
                    'kernel' for any model (slower),
                    'deep' for neural networks (Keras/TF)

    Returns
    -------
    shap_values : SHAP values array
    """
    import shap  # Import here to make it optional

    print(f"[SHAP] Computing SHAP values for {model_name} (type={model_type}) ...")

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    elif model_type == "kernel":
        # Use a small background dataset for KernelExplainer
        background = shap.kmeans(X_sample, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample[:100])  # Limit for speed
    elif model_type == "deep":
        # For Keras models — expects 3D input for CNN/LSTM
        explainer = shap.GradientExplainer(model, X_sample[:200])
        shap_values = explainer.shap_values(X_sample[:100])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Summary plot (bar)
    fig = plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list):
        # Multi-class: use class 1 (seizure)
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    if sv.ndim == 3:
        # CNN/LSTM: (samples, timesteps, channels) → flatten last dims
        sv = sv.reshape(sv.shape[0], -1)
        X_plot = X_sample[:sv.shape[0]].reshape(sv.shape[0], -1)
    else:
        X_plot = X_sample[:sv.shape[0]]

    shap.summary_plot(sv, X_plot, feature_names=feature_names,
                      max_display=max_display, show=False)
    path = os.path.join(config.PLOTS_DIR, f"shap_summary_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Summary plot saved to {path}")

    # Bar plot (mean absolute SHAP)
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(sv, X_plot, feature_names=feature_names,
                      max_display=max_display, plot_type="bar", show=False)
    path_bar = os.path.join(config.PLOTS_DIR, f"shap_bar_{model_name}.png")
    plt.savefig(path_bar, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Bar plot saved to {path_bar}")

    return shap_values


# ═══════════════════════════════════════════════════════════════════════════════
# Grad-CAM for 1D CNN
# ═══════════════════════════════════════════════════════════════════════════════

def grad_cam_1d(model, input_signal: np.ndarray, target_class: int = None,
                last_conv_layer_name: str = None) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a 1D CNN model.

    This highlights which temporal regions of the EEG signal
    are most important for the model's prediction.

    Parameters
    ----------
    model               : Trained Keras 1D CNN model
    input_signal        : Single sample, shape (1, timesteps, 1) or (timesteps, 1) or (timesteps,)
    target_class        : Class to explain (default: predicted class)
    last_conv_layer_name: Name of the last Conv1D layer

    Returns
    -------
    heatmap : 1D array of same length as input, normalized [0, 1]
    """
    import tensorflow as tf

    # Ensure 3D tensor: (1, timesteps, 1)
    if not isinstance(input_signal, np.ndarray):
        input_signal = np.array(input_signal)
    if input_signal.ndim == 1:
        input_signal = input_signal.reshape(1, -1, 1)
    elif input_signal.ndim == 2:
        if input_signal.shape[0] == 1:
            input_signal = input_signal.reshape(1, -1, 1)
        else:
            input_signal = input_signal.reshape(input_signal.shape[0], input_signal.shape[1], 1)

    input_tensor = tf.convert_to_tensor(input_signal.astype(np.float32))

    # Auto-detect last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No Conv1D layer found in model.")

    # Try functional model construction first; if that fails (e.g. Keras 3 Sequential),
    # use layer-by-layer forward pass with GradientTape
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(input_tensor)
            if target_class is None:
                if predictions.shape[-1] == 1:
                    target_class = int(predictions.numpy()[0, 0] > 0.5)
                    loss = predictions[0, 0] if target_class == 1 else (1 - predictions[0, 0])
                else:
                    target_class = tf.argmax(predictions[0])
                    loss = predictions[0, target_class]
            else:
                if predictions.shape[-1] == 1:
                    loss = predictions[0, 0] if target_class == 1 else (1 - predictions[0, 0])
                else:
                    loss = predictions[0, target_class]
        grads = tape.gradient(loss, conv_output)
    except Exception:
        # Layer-by-layer sequential execution under GradientTape (Keras 3 robust)
        x = input_tensor
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                break
        conv_output = x

        with tf.GradientTape() as tape:
            tape.watch(conv_output)
            x_post = conv_output
            passed = False
            for layer in model.layers:
                if passed:
                    x_post = layer(x_post)
                elif layer.name == last_conv_layer_name:
                    passed = True
            predictions = x_post
            if target_class is None:
                if predictions.shape[-1] == 1:
                    target_class = int(predictions.numpy()[0, 0] > 0.5)
                    loss = predictions[0, 0] if target_class == 1 else (1 - predictions[0, 0])
                else:
                    target_class = tf.argmax(predictions[0])
                    loss = predictions[0, target_class]
            else:
                if predictions.shape[-1] == 1:
                    loss = predictions[0, 0] if target_class == 1 else (1 - predictions[0, 0])
                else:
                    loss = predictions[0, target_class]
        grads = tape.gradient(loss, conv_output)

    # Global average pooling over the spatial/temporal dimension
    weights = tf.reduce_mean(grads, axis=1)  # shape: (1, n_filters)

    # Weighted combination of feature maps
    cam = tf.reduce_sum(conv_output[0] * weights[0], axis=-1)  # shape: (temporal_dim,)
    cam = tf.nn.relu(cam)  # Keep only positive contributions
    cam = cam.numpy()

    # Resize to input length via interpolation
    input_length = input_signal.shape[1]
    if len(cam) != input_length:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(cam))
        x_new = np.linspace(0, 1, input_length)
        cam = interp1d(x_old, cam, kind="linear")(x_new)

    # Normalize to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam


def plot_grad_cam(signal: np.ndarray, heatmap: np.ndarray,
                  prediction: str = "", sample_idx: int = 0):
    """
    Overlay Grad-CAM heatmap on the EEG signal.

    Parameters
    ----------
    signal   : Original EEG signal, 1D array (178,)
    heatmap  : Grad-CAM heatmap, 1D array (178,), values in [0, 1]
    prediction : Prediction label for title
    sample_idx : Sample index for filename
    """
    t = np.arange(len(signal)) / config.SAMPLING_RATE

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, signal, color="black", linewidth=0.8, alpha=0.8, label="EEG Signal")

    # Color the background based on heatmap intensity
    for i in range(len(t) - 1):
        ax.axvspan(t[i], t[i + 1], alpha=heatmap[i] * 0.6,
                   color="red", linewidth=0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Grad-CAM Explanation — Sample #{sample_idx} ({prediction})")
    ax.legend(loc="upper right")
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, f"gradcam_sample_{sample_idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GRADCAM] Plot saved to {path}")

    return path


def generate_gradcam_payload(model, signal: np.ndarray, prediction_label: str = "",
                            confidence_pct: float = 0.0) -> dict:
    """
    Generate Grad-CAM heatmap vector and base64 PNG plot for API responses and web UI.

    Parameters
    ----------
    model            : Trained 1D CNN model
    signal           : 1D array of 178 EEG signal points
    prediction_label : "Seizure" or "Non-Seizure"
    confidence_pct   : Confidence percentage (e.g. 98.5)

    Returns
    -------
    dict with:
        'heatmap': list of 178 floats in [0, 1]
        'signal': list of 178 floats
        'image_base64': base64-encoded PNG data URI ('data:image/png;base64,...')
        'key_regions': list of dicts describing top attention segments
    """
    import base64
    import io

    sig_1d = np.array(signal, dtype=np.float32).flatten()
    input_3d = sig_1d.reshape(1, -1, 1)

    heatmap = grad_cam_1d(model, input_3d)

    # Render figure to in-memory buffer
    t = np.arange(len(sig_1d)) / config.SAMPLING_RATE

    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fcfcfd")

    # Plot EEG signal
    ax.plot(t, sig_1d, color="#1e293b", linewidth=1.2, label="EEG Signal (178 Hz)")

    # Color span according to heatmap
    for i in range(len(t) - 1):
        intensity = float(heatmap[i])
        if intensity > 0.05:
            ax.axvspan(t[i], t[i + 1], alpha=min(intensity * 0.75, 0.8),
                       color="#ef4444", linewidth=0)

    is_seizure = "seizure" in prediction_label.lower() and "non" not in prediction_label.lower()
    status_text = f"Prediction: {prediction_label} ({confidence_pct:.1f}% confidence)"
    title_color = "#dc2626" if is_seizure else "#16a34a"

    ax.set_title(f"Grad-CAM Temporal Attention Map — {status_text}",
                 fontsize=11, fontweight="bold", color=title_color, pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=9, color="#475569")
    ax.set_ylabel("Amplitude (Scaled)", fontsize=9, color="#475569")
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")

    import matplotlib.patches as mpatches
    signal_line = ax.lines[0]
    high_att_patch = mpatches.Patch(color="#ef4444", alpha=0.7, label="High Seizure Focus")
    ax.legend(handles=[signal_line, high_att_patch], loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    img_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    # Compute key temporal regions (segments where heatmap > 0.6)
    threshold = 0.6
    regions = []
    in_region = False
    start_idx = 0
    for i, val in enumerate(heatmap):
        if val >= threshold and not in_region:
            in_region = True
            start_idx = i
        elif val < threshold and in_region:
            in_region = False
            regions.append({
                "start_time_s": round(start_idx / config.SAMPLING_RATE, 3),
                "end_time_s": round(i / config.SAMPLING_RATE, 3),
                "max_attention": round(float(np.max(heatmap[start_idx:i])), 3)
            })
    if in_region:
        regions.append({
            "start_time_s": round(start_idx / config.SAMPLING_RATE, 3),
            "end_time_s": round((len(heatmap) - 1) / config.SAMPLING_RATE, 3),
            "max_attention": round(float(np.max(heatmap[start_idx:])), 3)
        })

    return {
        "heatmap": [round(float(x), 4) for x in heatmap],
        "signal": [round(float(x), 4) for x in sig_1d],
        "image_base64": img_b64,
        "key_regions": regions
    }

