from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


@dataclass
class OptimizerConfig:
    max_lr: float = 6e-4
    min_lr: float = 6e-4 * 0.1
    warmup_steps: int = 715
    max_steps: int = 50
    weight_decay: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.95
    eps: float = 1e-8


@dataclass
class GenerationConfig:
    max_length: int = 32
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
