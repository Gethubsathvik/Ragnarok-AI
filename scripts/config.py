"""Shared paths and model configuration."""
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
TOKENIZER_DIR = DATA_DIR / "tokenizer"

GPT2_VOCAB = 50257  # real GPT-2 vocabulary size


@dataclass
class GPTConfig:
    block_size: int = 512        # context length
    vocab_size: int = 50304      # 50257 padded to a multiple of 64 (faster GPU kernels)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = False

    def to_dict(self):
        return asdict(self)
