from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .utils import seed_worker


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_channels: int
    n_classes: int
    seq_len: int
    source: str


class SequentialVisionDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        train: bool,
        permute: bool = False,
        seed: int = 1111,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name.lower()
        rng = np.random.RandomState(seed)
        self.permutation = None

        self.x, self.y, self.spec = self._load_dataset(root=root, train=train)
        if permute:
            self.permutation = rng.permutation(self.spec.seq_len)

    def _load_dataset(self, root: str, train: bool):
        try:
            from torchvision import datasets, transforms
        except Exception as e:  # pragma: no cover - import-time environment issue
            raise ImportError(
                'torchvision is required for dataset loading. Install torchvision to use this project.'
            ) from e

        to_tensor = transforms.ToTensor()

        def materialize(base_ds, spec: DatasetSpec, target_transform: Callable[[int], int] | None = None):
            xs, ys = [], []
            for img, target in base_ds:
                if img.ndim == 2:
                    img = img.unsqueeze(0)
                xs.append(img.reshape(-1).numpy())
                target = int(target)
                if target_transform is not None:
                    target = target_transform(target)
                ys.append(target)
            x = np.stack(xs).astype(np.float32)
            y = np.asarray(ys, dtype=np.int64)
            return x, y, spec

        if self.dataset_name == 'seqmnist':
            try:
                base = datasets.MNIST(root=root, train=train, download=True, transform=to_tensor)
                spec = DatasetSpec(name='seqmnist', input_channels=1, n_classes=10, seq_len=28 * 28, source='torchvision')
                return materialize(base, spec)
            except Exception:
                from sklearn.datasets import fetch_openml

                mnist = fetch_openml('mnist_784', version=1, as_frame=False)
                x = mnist.data.astype(np.float32) / 255.0
                y = mnist.target.astype(np.int64)
                if train:
                    x, y = x[:60000], y[:60000]
                else:
                    x, y = x[60000:], y[60000:]
                spec = DatasetSpec(name='seqmnist', input_channels=1, n_classes=10, seq_len=28 * 28, source='openml')
                return x, y, spec

        if self.dataset_name == 'fashion_mnist':
            base = datasets.FashionMNIST(root=root, train=train, download=True, transform=to_tensor)
            spec = DatasetSpec(name='fashion_mnist', input_channels=1, n_classes=10, seq_len=28 * 28, source='torchvision')
            return materialize(base, spec)

        if self.dataset_name == 'kmnist':
            base = datasets.KMNIST(root=root, train=train, download=True, transform=to_tensor)
            spec = DatasetSpec(name='kmnist', input_channels=1, n_classes=10, seq_len=28 * 28, source='torchvision')
            return materialize(base, spec)

        if self.dataset_name == 'emnist_digits':
            base = datasets.EMNIST(root=root, split='digits', train=train, download=True, transform=to_tensor)
            spec = DatasetSpec(name='emnist_digits', input_channels=1, n_classes=10, seq_len=28 * 28, source='torchvision')
            return materialize(base, spec)

        if self.dataset_name == 'cifar10_gray':
            grayscale = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
            base = datasets.CIFAR10(root=root, train=train, download=True, transform=grayscale)
            spec = DatasetSpec(name='cifar10_gray', input_channels=1, n_classes=10, seq_len=32 * 32, source='torchvision')
            return materialize(base, spec)

        raise ValueError(
            f"Unsupported dataset '{self.dataset_name}'. Supported datasets: "
            "seqmnist, fashion_mnist, kmnist, emnist_digits, cifar10_gray"
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        seq = torch.from_numpy(self.x[idx])
        if self.permutation is not None:
            seq = seq[self.permutation]
        seq = seq.unsqueeze(0)
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
    dataset_name: str
    train_size: int
    val_size: int
    test_size: int


def build_sequence_loaders(
    data_root: str,
    batch_size: int,
    dataset_name: str,
    permute: bool,
    seed: int,
    val_ratio: float = 0.1,
    num_workers: int = 2,
) -> DataBundle:
    train_full = SequentialVisionDataset(data_root, dataset_name=dataset_name, train=True, permute=permute, seed=seed)
    test_ds = SequentialVisionDataset(data_root, dataset_name=dataset_name, train=False, permute=permute, seed=seed)

    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=generator)

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'worker_init_fn': seed_worker,
        'generator': generator,
        'persistent_workers': num_workers > 0,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_channels=train_full.spec.input_channels,
        n_classes=train_full.spec.n_classes,
        seq_len=train_full.spec.seq_len,
        dataset_name=train_full.spec.name,
        train_size=train_size,
        val_size=val_size,
        test_size=len(test_ds),
    )


def build_seqmnist_loaders(
    data_root: str,
    batch_size: int,
    permute: bool,
    seed: int,
    val_ratio: float = 0.1,
    num_workers: int = 2,
) -> DataBundle:
    return build_sequence_loaders(
        data_root=data_root,
        batch_size=batch_size,
        dataset_name='seqmnist',
        permute=permute,
        seed=seed,
        val_ratio=val_ratio,
        num_workers=num_workers,
    )
