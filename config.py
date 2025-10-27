from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int = 8
    vocab_size: int = 50257
    embed_dim: int = 768
    block_size: int = 127 
    num_heads: int = 2  # must be divisible by embed_dim!
    num_layers: int = 1
    epochs: int = 1
    learning_rate: float = 0.001
