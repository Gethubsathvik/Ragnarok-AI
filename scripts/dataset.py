"""Memory-mapped token dataset + DataLoaders."""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import DATA_DIR


class TokenBlocks(Dataset):
    """Non-overlapping blocks of `block_size` tokens from a uint16 .bin file.
    Item = (x, y) where y is x shifted left by one (next-token targets)."""

    def __init__(self, path, block_size):
        self.path = str(path)
        self.block = block_size
        n_tokens = os.path.getsize(self.path) // 2
        self.n = (n_tokens - 1) // block_size
        self.data = None  # opened lazily (safe with DataLoader workers)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.data is None:
            self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        s = i * self.block
        chunk = torch.from_numpy(self.data[s : s + self.block + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def make_loaders(batch_size, block_size, num_workers=4, data_dir=DATA_DIR):
    train_ds = TokenBlocks(data_dir / "train.bin", block_size)
    val_ds = TokenBlocks(data_dir / "val.bin", block_size)
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=num_workers > 0)
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, **kw)
    return train, val
