from config import Config
from core.engine import train


if __name__ == "__main__":
    cfg = Config()
    train(config=cfg)
