################################ NEURAL NETWORK TRAINER ################################

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, GRU, Dense, Dropout, RepeatVector
from tensorflow.keras.optimizers import Adam  # AdamW no longer required
import joblib
import time
from RALLY_global_variables import global_variables as gv

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def NN_train():
    start_time = time.time()

    # Load historical data (CSV columns: timestamp, openPrice, highPrice, lowPrice, lastPrice, volume)
    data = pd.read_csv(gv.TRAINING_DATA)
    # We use only OHLCV data; the order of rows encodes time.
    features = data[["openPrice", "highPrice", "lowPrice", "lastPrice", "volume"]].values
    labels = data["lastPrice"].values  # Target: next close price (lastPrice)

    def create_sequences(features, labels, window_size, horizon):
        X, y = [], []
        max_i = len(features) - window_size - (horizon - 1)
        for i in range(max_i):
            X.append(features[i : i + window_size])
            y.append(labels[i + window_size + (horizon - 1)])
        return np.array(X), np.array(y)

    X, y = create_sequences(features, labels, gv.WINDOW_SIZE, gv.HORIZON)
    if len(X) == 0 or len(y) == 0:
        raise ValueError("❌ Not enough data to train! Collect more historical data.")

    # Scale the features and labels using separate scalers.
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # LEARNING RATE SCHEDULE — One‑Cycle Policy
    steps_per_epoch = len(X_scaled) // gv.BATCH_SIZE
    total_steps = steps_per_epoch * gv.EPOCHS  # full cycle length
    warmup_steps = int(gv.WARMUP_STEPS * total_steps)  # raise phase duration

    @tf.keras.utils.register_keras_serializable(package="Custom")
    class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
        """Implements Leslie Smith's One‑Cycle policy with a linear warm‑up followed by a cosine decay back to min_lr."""

        def __init__(self, max_lr, min_lr, warm_steps, total_steps, name=None):
            super().__init__()
            self.max_lr = max_lr
            self.min_lr = min_lr
            self.warm_steps = warm_steps
            self.total_steps = total_steps
            self.name = name

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            # Phase 1: linear warm‑up from min_lr → max_lr
            lr_increase = self.min_lr + (self.max_lr - self.min_lr) * (step / tf.maximum(1.0, self.warm_steps))
            # Phase 2: cosine decay back to min_lr
            progress = (step - self.warm_steps) / tf.maximum(1.0, self.total_steps - self.warm_steps)
            cosine_decay = 0.5 * (1 + tf.cos(np.pi * tf.clip_by_value(progress, 0.0, 1.0)))
            lr_decay = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            # Use warm‑up LR until warm_steps are finished
            return tf.where(step < self.warm_steps, lr_increase, lr_decay)

        def get_config(self):
            return {
                "max_lr": self.max_lr,
                "min_lr": self.min_lr,
                "warm_steps": self.warm_steps,
                "total_steps": self.total_steps,
                "name": self.name,
            }

    if gv.SCHEDULER:
        lr_schedule = OneCycleLR(gv.MAX_LR, gv.MIN_LR, warmup_steps, total_steps)
        optimizer = Adam(learning_rate=lr_schedule)  # Adam used instead of AdamW
    else:
        optimizer = Adam(learning_rate=gv.LEARNING_RATE)

    # ------------------------------------------------------------------
    # GRU‑based Seq2Seq network remains unchanged
    # ------------------------------------------------------------------
    gru_hidden = gv.LAYERS_SIZE
    gru_layers = gv.LAYERS
    gru_dropout = gv.DROPOUT

    # Build a GRU-based Seq2Seq network.
    encoder_inputs = tf.keras.Input(shape=(gv.WINDOW_SIZE, X.shape[-1]), name="encoder_inputs")
    x = encoder_inputs

    # Encoder: stacked GRUs (gru_layers-1 with return_sequences) and final GRU returning state.
    for i in range(gru_layers - 1):
        x = GRU(
            gru_hidden,
            return_sequences=True,
            dropout=gru_dropout,
            name=f"encoder_gru_{i + 1}",
        )(x)

    _, state = GRU(
        gru_hidden,
        return_state=True,
        dropout=gru_dropout,
        name=f"encoder_gru_{gru_layers}",
    )(x)

    # Decoder: repeat the encoder state and run through another GRU stack.
    decoder_inputs = RepeatVector(1, name="decoder_repeat")(state)  # sequence length=1 for single‑step forecast
    y_dec = decoder_inputs
    for i in range(gru_layers - 1):
        y_dec = GRU(
            gru_hidden,
            return_sequences=True,
            dropout=gru_dropout,
            name=f"decoder_gru_{i + 1}",
        )(y_dec)

    y_dec = GRU(
        gru_hidden,
        return_sequences=False,
        dropout=gru_dropout,
        name=f"decoder_gru_{gru_layers}",
    )(y_dec)

    outputs = Dense(1, name="prediction")(y_dec)  # Single price prediction

    model = tf.keras.Model(encoder_inputs, outputs, name="GRU_Seq2Seq")

    model.compile(optimizer=optimizer, loss="mse")
    model.summary()

    # Train the model using validation split.
    model.fit(
        X_scaled,
        y_scaled,
        epochs=gv.EPOCHS,
        batch_size=gv.BATCH_SIZE,
        validation_split=gv.VALIDATION_SPLIT,
    )

    # Save the trained model.
    model.save(gv.NEURAL_NETWORK, include_optimizer=False)
    print("✅ Model saved to", gv.NEURAL_NETWORK)

    # Save scalers for future use.
    joblib.dump(scaler_X, "scaler_X.save")
    joblib.dump(scaler_y, "scaler_y.save")
    print("✅ Scalers saved: scaler_X.save and scaler_y.save")

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Training time: {elapsed_time:.2f} minutes")


if __name__ == "__main__":
    NN_train()
