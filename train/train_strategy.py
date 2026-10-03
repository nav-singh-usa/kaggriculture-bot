from models.strategy_model import StrategyModel

def train(sequence_x, sequence_y, epochs=20):
    model = StrategyModel(
        sequence_x.shape[-1],
        output_dim=sequence_y.shape[-1]
    )
    model.model.fit(
        sequence_x,
        sequence_y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    model.model.save("strategy_model.keras")
    return model

if __name__ == "__main__":
    print("Supply sequential observation windows and strategy targets.")
