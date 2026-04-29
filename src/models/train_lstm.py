import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def train_lstm(x_train_scaled, y_train_scaled):
    x_train_lstm = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))

    model = Sequential(
        [
            Input(shape=(x_train_lstm.shape[1], x_train_lstm.shape[2])),
            LSTM(64, activation="relu", return_sequences=True),
            Dropout(0.3),
            LSTM(32, activation="relu"),
            Dropout(0.3),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train_lstm, y_train_scaled, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    return model
