from config import Config
from engine import train


if __name__ == "__main__":
    cfg = Config()
    train(config=cfg)
