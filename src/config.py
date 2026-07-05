from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int = 16
    vocab_size: int = 50257
    embed_dim: int = 768
    block_size: int = 128
    num_heads: int = 8  # embed_dim must be divisible by num_heads
    num_layers: int = 6
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int = 500
    min_lr: float = 1e-5
    gradient_accumulation_steps: int = 4
    dropout: float = 0.1
    max_grad_norm: float | None = 1.0
    ignore_index: int = -100
    resume_training: bool = True
    checkpoint_path: str | None = "./weights/checkpoint_epoch_002_step_549207.pt"
    checkpoint_dir: str = "./weights"
    checkpoint_prefix: str = "checkpoint"
    model_type: str = "decoder"
    seed: int = 42
    num_workers: int = 4
    prefetch_factor: int = 2

    def __post_init__(self):
        if self.block_size < 1:
            raise ValueError("block_size must be at least 1")
        if self.num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")


@dataclass
class RoFormerConfig(Config):
    batch_size: int = 8
    block_size: int = 256
    model_type: str = "roformer"
    checkpoint_prefix: str = "roformer_checkpoint"
    checkpoint_path: str | None = "roformer_checkpoint_epoch_000_step_008335.pt" 
    resume_training: bool = False
