import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

ACTIONS = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST",
    "PLANT_WHEAT", "PLANT_CARROT", "PLANT_TOMATO",
    "PLANT_STRAWBERRY", "PLANT_MELON",
    "WATER", "HARVEST", "FERTILIZE", "FEED",
    "CARE", "COLLECT_FERTILIZER", "DIG"
]

class MoveModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build()

    def _build(self):
        inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(len(ACTIONS), activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def action(self, features):
        x = np.asarray(features, dtype=np.float32)[None, :]
        probabilities = self.model.predict(x, verbose=0)[0]
        return ACTIONS[int(np.argmax(probabilities))]
