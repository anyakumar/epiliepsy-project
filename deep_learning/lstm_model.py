"""
LSTM / BiLSTM Model for EEG Seizure Detection
----------------------------------------------
Architecture:
  Input (178, 1)
    → BiLSTM(64, return_sequences=True)
    → Dropout(0.3)
    → BiLSTM(32)
    → Dropout(0.3)
    → Dense(64, relu)
    → Dropout(0.3)
    → Dense(num_classes, softmax/sigmoid)

Captures temporal dependencies in EEG time-series data.
Uses TensorFlow / Keras.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

from . import config


def build_lstm_model(input_shape: tuple = None, num_classes: int = 2,
                     lstm_units: list = None, dense_units: int = None,
                     dropout: float = None, recurrent_dropout: float = None,
                     bidirectional: bool = None,
                     learning_rate: float = None) -> keras.Model:
    """
    Build and compile an LSTM or BiLSTM model for EEG classification.

    Parameters
    ----------
    input_shape       : (timesteps, features), default (178, 1)
    num_classes       : 2 (binary) or 5 (multi-class)
    lstm_units        : List of units per LSTM layer
    dense_units       : Dense layer units before output
    dropout           : Dropout rate
    recurrent_dropout : Recurrent dropout within LSTM cells
    bidirectional     : If True, wraps LSTM in Bidirectional wrapper
    learning_rate     : Adam learning rate

    Returns
    -------
    model : keras.Model (compiled)
    """
    input_shape = input_shape or (config.NUM_FEATURES, 1)
    lstm_units = lstm_units or config.LSTM_UNITS
    dense_units = dense_units or config.LSTM_DENSE_UNITS
    dropout = dropout or config.LSTM_DROPOUT
    recurrent_dropout = recurrent_dropout or config.LSTM_RECURRENT_DROPOUT
    bidirectional = bidirectional if bidirectional is not None else config.LSTM_BIDIRECTIONAL
    learning_rate = learning_rate or config.LSTM_LEARNING_RATE

    model = keras.Sequential(name="EEG_BiLSTM" if bidirectional else "EEG_LSTM")

    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)  # Return sequences for all but last LSTM

        lstm_layer = layers.LSTM(
            units,
            return_sequences=return_seq,
            recurrent_dropout=recurrent_dropout,
            name=f"lstm_{i+1}"
        )

        if i == 0:
            # First layer needs input_shape
            if bidirectional:
                model.add(layers.Bidirectional(
                    lstm_layer, input_shape=input_shape, name=f"bilstm_{i+1}"
                ))
            else:
                lstm_layer = layers.LSTM(
                    units,
                    return_sequences=return_seq,
                    recurrent_dropout=recurrent_dropout,
                    input_shape=input_shape,
                    name=f"lstm_{i+1}"
                )
                model.add(lstm_layer)
        else:
            if bidirectional:
                model.add(layers.Bidirectional(lstm_layer, name=f"bilstm_{i+1}"))
            else:
                model.add(lstm_layer)

        model.add(layers.Dropout(dropout, name=f"dropout_lstm_{i+1}"))

    # Dense classifier
    model.add(layers.Dense(dense_units, activation="relu", name="dense_1"))
    model.add(layers.Dropout(dropout, name="dropout_dense"))

    # Output
    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid", name="output"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        model.add(layers.Dense(num_classes, activation="softmax", name="output"))
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )

    return model


def get_lstm_callbacks(model_name: str = "lstm_best") -> list:
    """
    Standard callbacks for LSTM training.
    """
    checkpoint_path = os.path.join(config.MODELS_DIR, f"{model_name}.keras")
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.LSTM_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               num_classes: int = 2,
               epochs: int = None, batch_size: int = None,
               bidirectional: bool = None) -> tuple:
    """
    Build, train, and return the LSTM/BiLSTM model + history.

    Parameters
    ----------
    X_train, y_train : Training data (shaped for LSTM)
    X_val, y_val     : Validation data
    num_classes      : 2 or 5
    epochs, batch_size : Training params
    bidirectional    : Use BiLSTM

    Returns
    -------
    model   : Trained keras.Model
    history : keras History object
    """
    epochs = epochs or config.LSTM_EPOCHS
    batch_size = batch_size or config.LSTM_BATCH_SIZE
    bidirectional = bidirectional if bidirectional is not None else config.LSTM_BIDIRECTIONAL

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape=input_shape, num_classes=num_classes,
                             bidirectional=bidirectional)
    model.summary()

    name = "bilstm_best" if bidirectional else "lstm_best"
    cbs = get_lstm_callbacks(name)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1,
    )

    return model, history


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_lstm_model()
    model.summary()
    print("\n✅ BiLSTM model built successfully.")
