from config import Config
from engine import train


if __name__ == "__main__":
    cfg = Config(
        block_size=63, 
        batch_size=16, 
        num_heads=4, 
        num_layers=4, 
        learning_rate=1e-3
    )
    train(config=cfg)
