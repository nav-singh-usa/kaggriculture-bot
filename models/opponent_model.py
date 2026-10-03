import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class OpponentModel:
    def __init__(self, input_dim, output_dim=13):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._build()

    def _build(self):
        inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dense(96, activation="relu")(x)
        x = layers.Dropout(0.15)(x)
        outputs = layers.Dense(self.output_dim, activation="sigmoid")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="mse"
        )
        return model

    def predict(self, features):
        x = np.asarray(features, dtype=np.float32)[None, :]
        return self.model.predict(x, verbose=0)[0]
