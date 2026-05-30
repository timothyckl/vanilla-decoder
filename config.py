from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int = 16
    vocab_size: int = 50257
    embed_dim: int = 768
    block_size: int = 63
    num_heads: int = 4  # embed_dim must be divisible by num_heads
    num_layers: int = 4
    epochs: int = 1
    learning_rate: float = 1e-3
    dropout: float = 0.1
    max_grad_norm: float | None = 1.0
    ignore_index: int = -100
    checkpoint_path: str = "./weights/model_weights.pt"

    def __post_init__(self):
        if self.block_size < 1:
            raise ValueError("block_size must be at least 1")
        if self.num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")
