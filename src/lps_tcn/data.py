from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .utils import seed_worker


class SequentialMNIST(Dataset):
    def __init__(self, root: str, train: bool, permute: bool = False, seed: int = 1111):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.permutation = rng.permutation(28 * 28) if permute else None

        try:
            from torchvision import datasets, transforms

            base = datasets.MNIST(
                root=root,
                train=train,
                download=True,
                transform=transforms.ToTensor(),
            )
            xs, ys = [], []
            for img, target in base:
                xs.append(img.view(-1).numpy())
                ys.append(int(target))
            self.x = np.stack(xs).astype(np.float32)
            self.y = np.asarray(ys, dtype=np.int64)
        except Exception:
            from sklearn.datasets import fetch_openml

            mnist = fetch_openml("mnist_784", version=1, as_frame=False)
            x = mnist.data.astype(np.float32) / 255.0
            y = mnist.target.astype(np.int64)
            if train:
                self.x, self.y = x[:60000], y[:60000]
            else:
                self.x, self.y = x[60000:], y[60000:]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        seq = torch.from_numpy(self.x[idx])
        if self.permutation is not None:
            seq = seq[self.permutation]
        seq = seq.unsqueeze(0)  # [1, 784]
        target = int(self.y[idx])
        return seq, target


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_channels: int
    n_classes: int
    seq_len: int


def build_seqmnist_loaders(
    data_root: str,
    batch_size: int,
    permute: bool,
    seed: int,
    val_ratio: float = 0.1,
    num_workers: int = 2,
) -> DataBundle:
    train_full = SequentialMNIST(data_root, train=True, permute=permute, seed=seed)
    test_ds = SequentialMNIST(data_root, train=False, permute=permute, seed=seed)

    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=generator)

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_channels=1,
        n_classes=10,
        seq_len=784,
    )
