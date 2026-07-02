from config import RoFormerConfig
from core.engine import train


if __name__ == "__main__":
    cfg = RoFormerConfig()
    train(config=cfg)
