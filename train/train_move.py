from models.move_model import MoveModel

def train(x_train, y_train, epochs=20):
    model = MoveModel(x_train.shape[1])
    model.model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    model.model.save("move_model.keras")
    return model

if __name__ == "__main__":
    print("Supply state/action examples for policy training.")
