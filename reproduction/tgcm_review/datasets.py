"""Evaluation-only sequence datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class RaggedSequenceDataset(Dataset):
    """Memory-mappable compact replacement for a validation shard."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        payload = np.load(self.path, mmap_mode="r", allow_pickle=False)
        self.tokens = payload["tokens"]
        self.labels = payload["labels"]
        self.offsets = payload["offsets"]
        self.timesteps = payload["timesteps"]

    def __len__(self) -> int:
        return len(self.timesteps)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return {
            "x": torch.from_numpy(np.asarray(self.tokens[start:end], dtype=np.int64)),
            "y_z": torch.from_numpy(np.asarray(self.labels[start:end], dtype=np.int64)),
            "t": torch.tensor(int(self.timesteps[index]), dtype=torch.long),
        }


def collate_sequences(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    tokens = pad_sequence([item["x"] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item["y_z"] for item in batch], batch_first=True, padding_value=0)
    return {
        "x": tokens,
        "y_z": labels,
        "t": torch.stack([item["t"] for item in batch]),
        "mask": tokens.ne(0),
    }


def tokenize(techniques: Sequence[str], vocabulary: dict[str, int]) -> list[int]:
    return [int(vocabulary.get(str(value).replace(".", ""), 1)) for value in techniques]
