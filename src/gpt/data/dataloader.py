import os
from typing import Dict, List, Tuple

import lightning as L
import numpy as np
import tiktoken
import torch
import torch.utils.data as tud
from torch.utils.data import Dataset


class ShakespeareDataset(tud.Dataset):
    def __init__(self, data_path, block_size):
        self.block_size = block_size
        self.data = self.load_data(data_path)
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.tokens = self.tokenizer.encode(self.data)
        self.tokens = torch.tensor(self.tokens)
        self.n_tokens = len(self.tokens)
        self.n_samples = self.n_tokens // self.block_size

    @staticmethod
    def load_data(data_path):
        with open(data_path, "r") as f:
            data = f.read()
        return data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = (idx + 1) * self.block_size + 1
        chunk = self.tokens[start_idx:end_idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class ShakespeareDataModule(L.LightningDataModule):
    def __init__(self, data_path, block_size, batch_size):
        super().__init__()
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = ShakespeareDataset(self.data_path, self.block_size)

    def train_dataloader(self):
        return tud.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> torch.Any:
        return self.train_dataloader()


class FineWebEduDataset(Dataset):
    def __init__(
        self, data_path: str, block_size: int, split: str, dtype: np.dtype = np.uint16
    ):
        """Dataset class for loading tokenized text sequences from pre-tokenized shards.

        Args:
            data_path: Path to the directory containing the pre-tokenized shards.
            block_size: The size of each sequence.
            split: The split of the dataset to load (e.g. "train" or "val").
            dtype: The numpy data type of the tokens.
        """
        shards = [
            os.path.join(data_path, f) for f in os.listdir(data_path) if split in f
        ]
        self.shards = shards
        self.block_size = block_size  # Add 1 for target
        self.dtype: np.dtype = dtype
        self.mmaps: Dict[int, np.memmap] = {}
        self.buffers: Dict[int, memoryview] = {}
        self.chunk_sizes = self._load_chunk_sizes()
        self.total_sequences = sum(size // (block_size + 1) for size in self.chunk_sizes)

    def _load_chunk_sizes(self) -> List[int]:
        """Load sizes of each shard for determining chunk splits."""
        sizes = []
        for shard_path in self.shards:
            size, _ = self._compute_size_excluding_header(shard_path)
            sizes.append(size)
        return sizes

    def _compute_size_excluding_header(self, path: str) -> Tuple[int, int]:
        """Compute the size of the .npy file excluding the header."""
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            header = np.lib.format._read_array_header(f, version)
            offset = f.tell()  # Get the position after the header
        if header[-1] != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype} but got {header[-1]}")
        size = os.path.getsize(path) - offset
        size = size // np.dtype(self.dtype).itemsize
        return size, offset

    def _load_shard(self, shard_idx: int):
        """Load a shard file into memory using np.memmap."""
        if shard_idx not in self.mmaps:
            shard_path = self.shards[shard_idx]
            _, offset = self._compute_size_excluding_header(shard_path)
            self.mmaps[shard_idx] = np.memmap(
                shard_path, dtype=self.dtype, mode="r", offset=offset
            )
            self.buffers[shard_idx] = memoryview(self.mmaps[shard_idx])

    def _get_shard_and_offset(self, index: int) -> Tuple[int, int]:
        """Convert a global sequence index to a specific shard and offset within the shard."""
        sequence_idx = index
        for i, chunk_size in enumerate(self.chunk_sizes):
            num_sequences = chunk_size // (self.block_size + 1)
            if sequence_idx < num_sequences:
                return i, sequence_idx * self.block_size
            sequence_idx -= num_sequences
        raise IndexError("Index out of range")

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, index: int) -> torch.Tensor:
        """Retrieve a single sequence of tokens by global index."""
        shard_idx, offset = self._get_shard_and_offset(index)
        self._load_shard(shard_idx)
        buffer = self.buffers[shard_idx]
        start = offset * np.dtype(self.dtype).itemsize
        data = np.frombuffer(
            buffer, dtype=self.dtype, count=self.block_size + 1, offset=start
        )
        data = data.astype(np.int64)
        x = torch.tensor(data[:-1], dtype=torch.long)
        y = torch.tensor(data[1:], dtype=torch.long)
        return x, y


class FineWebEduDataModule(L.LightningDataModule):
    def __init__(
        self, data_path: str, block_size: int, batch_size: int, num_workers: int
    ):
        super().__init__()
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = FineWebEduDataset(self.data_path, self.block_size, "train")
        self.val_dataset = FineWebEduDataset(self.data_path, self.block_size, "val")

    def train_dataloader(self):
        return tud.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> torch.Any:
        return tud.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
