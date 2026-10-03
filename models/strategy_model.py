import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class StrategyModel:
    def __init__(self, input_dim, state_dim=64, output_dim=32):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.model = self._build()
        self.state = np.zeros(state_dim, dtype=np.float32)

    def _build(self):
        sequence = keras.Input(shape=(None, self.input_dim))
        x = layers.GRU(
            self.state_dim,
            return_sequences=False,
            name="strategy_memory"
        )(sequence)
        x = layers.Dense(64, activation="relu")(x)
        output = layers.Dense(self.output_dim, activation="tanh")(x)

        model = keras.Model(sequence, output)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="mse"
        )
        return model

    def reset(self):
        self.state.fill(0.0)

    def update(self, features):
        x = np.asarray(features, dtype=np.float32)

        if x.ndim == 1:
            x = x[None, :]

        sequence = x[None, :, :]
        self.state = self.model.predict(sequence, verbose=0)[0]
        return self.state.copy()
