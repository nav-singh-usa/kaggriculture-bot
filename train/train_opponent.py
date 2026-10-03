from models.opponent_model import OpponentModel

def train(x_train, y_train, epochs=20):
    model = OpponentModel(x_train.shape[1], y_train.shape[1])
    model.model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    model.model.save("opponent_model.keras")
    return model

if __name__ == "__main__":
    print("Supply public observations and hidden opponent-state labels.")
