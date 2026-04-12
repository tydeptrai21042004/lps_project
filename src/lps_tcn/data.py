from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .utils import seed_worker


VISION_DATASETS = (
    'seqmnist',
    'fashion_mnist',
    'kmnist',
    'emnist_digits',
    'cifar10_gray',
)

ARCHIVE_DATASETS: dict[str, str] = {
    'ecg5000': 'ECG5000',
    'electricdevices': 'ElectricDevices',
    'forda': 'FordA',
    'wafer': 'Wafer',
    'two_patterns': 'TwoPatterns',
    'basicmotions': 'BasicMotions',
}

SYNTHETIC_DATASETS = (
    'synthetic_sines',
    'synthetic_shiftmix',
    'synthetic_multiscale',
)

DATASET_CHOICES = tuple(sorted((*VISION_DATASETS, *ARCHIVE_DATASETS.keys(), *SYNTHETIC_DATASETS)))


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_channels: int
    n_classes: int
    seq_len: int
    source: str


class InMemorySequenceDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        spec: DatasetSpec,
        *,
        permute: bool = False,
        seed: int = 1111,
    ) -> None:
        super().__init__()
        if x.ndim != 3:
            raise ValueError(f'Expected x with shape [N, C, T], got {x.shape}')
        if len(x) != len(y):
            raise ValueError('x and y must have the same number of samples')

        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.spec = spec

        self.permutation = None
        if permute:
            rng = np.random.RandomState(seed)
            self.permutation = rng.permutation(self.spec.seq_len)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        seq = torch.from_numpy(self.x[idx])
        if self.permutation is not None:
            seq = seq[:, self.permutation]
        target = int(self.y[idx])
        return seq, target


class SequentialVisionDataset(InMemorySequenceDataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        train: bool,
        permute: bool = False,
        seed: int = 1111,
    ) -> None:
        x, y, spec = self._load_dataset(root=root, dataset_name=dataset_name, train=train)
        super().__init__(x=x, y=y, spec=spec, permute=permute, seed=seed)

    @staticmethod
    def _load_dataset(root: str, dataset_name: str, train: bool):
        dataset_name = dataset_name.lower()
        try:
            from torchvision import datasets, transforms
        except Exception as e:  # pragma: no cover - import-time environment issue
            raise ImportError(
                'torchvision is required for vision dataset loading. Install torchvision to use these datasets.'
            ) from e

        to_tensor = transforms.ToTensor()

        def materialize(
            base_ds,
            spec: DatasetSpec,
            target_transform: Callable[[int], int] | None = None,
        ):
            xs, ys = [], []
            for img, target in base_ds:
                if img.ndim == 2:
                    img = img.unsqueeze(0)
                xs.append(img.reshape(1, -1).numpy())
                target = int(target)
                if target_transform is not None:
                    target = target_transform(target)
                ys.append(target)
            x = np.stack(xs).astype(np.float32)
            y = np.asarray(ys, dtype=np.int64)
            return x, y, spec

        if dataset_name == 'seqmnist':
            try:
                base = datasets.MNIST(root=root, train=train, download=True, transform=to_tensor)
                spec = DatasetSpec(
                    name='seqmnist',
                    input_channels=1,
                    n_classes=10,
                    seq_len=28 * 28,
                    source='torchvision',
                )
                return materialize(base, spec)
            except Exception:
                from sklearn.datasets import fetch_openml

                mnist = fetch_openml('mnist_784', version=1, as_frame=False)
                x = (mnist.data.astype(np.float32) / 255.0).reshape(-1, 1, 28 * 28)
                y = mnist.target.astype(np.int64)
                if train:
                    x, y = x[:60000], y[:60000]
                else:
                    x, y = x[60000:], y[60000:]
                spec = DatasetSpec(
                    name='seqmnist',
                    input_channels=1,
                    n_classes=10,
                    seq_len=28 * 28,
                    source='openml',
                )
                return x, y, spec

        if dataset_name == 'fashion_mnist':
            base = datasets.FashionMNIST(root=root, train=train, download=True, transform=to_tensor)
            spec = DatasetSpec('fashion_mnist', 1, 10, 28 * 28, 'torchvision')
            return materialize(base, spec)

        if dataset_name == 'kmnist':
            base = datasets.KMNIST(root=root, train=train, download=True, transform=to_tensor)
            spec = DatasetSpec('kmnist', 1, 10, 28 * 28, 'torchvision')
            return materialize(base, spec)

        if dataset_name == 'emnist_digits':
            base = datasets.EMNIST(root=root, split='digits', train=train, download=True, transform=to_tensor)
            spec = DatasetSpec('emnist_digits', 1, 10, 28 * 28, 'torchvision')
            return materialize(base, spec)

        if dataset_name == 'cifar10_gray':
            grayscale = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
            base = datasets.CIFAR10(root=root, train=train, download=True, transform=grayscale)
            spec = DatasetSpec('cifar10_gray', 1, 10, 32 * 32, 'torchvision')
            return materialize(base, spec)

        raise ValueError(
            f"Unsupported vision dataset '{dataset_name}'. Supported datasets: {', '.join(VISION_DATASETS)}"
        )


def _resample_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError(f'Expected 1D signal, got shape {signal.shape}')
    if len(signal) == target_len:
        return signal.astype(np.float32, copy=False)
    if len(signal) == 1:
        return np.full(target_len, float(signal[0]), dtype=np.float32)

    src_idx = np.linspace(0.0, 1.0, num=len(signal), dtype=np.float32)
    tgt_idx = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(tgt_idx, src_idx, signal).astype(np.float32)


def _ensure_nct_array(X, *, target_len: int | None = None) -> np.ndarray:
    if isinstance(X, np.ndarray):
        arr = X.astype(np.float32, copy=False)
        if arr.ndim == 2:
            arr = arr[:, None, :]
        elif arr.ndim != 3:
            raise ValueError(f'Unsupported ndarray shape: {arr.shape}')
        if target_len is not None and arr.shape[-1] != target_len:
            arr = np.stack(
                [_resample_1d(arr[i, c], target_len) for i in range(arr.shape[0]) for c in range(arr.shape[1])],
                axis=0,
            ).reshape(arr.shape[0], arr.shape[1], target_len)
        return arr

    if not isinstance(X, (list, tuple)) or len(X) == 0:
        raise ValueError('Expected X to be a non-empty ndarray, list, or tuple')

    processed_cases = []
    inferred_target_len = target_len
    if inferred_target_len is None:
        lengths = []
        for case in X:
            case_arr = np.asarray(case, dtype=np.float32)
            if case_arr.ndim == 1:
                case_arr = case_arr[None, :]
            elif case_arr.ndim != 2:
                raise ValueError(f'Unsupported case shape: {case_arr.shape}')
            lengths.append(case_arr.shape[-1])
        inferred_target_len = int(max(lengths))

    for case in X:
        case_arr = np.asarray(case, dtype=np.float32)
        if case_arr.ndim == 1:
            case_arr = case_arr[None, :]
        elif case_arr.ndim != 2:
            raise ValueError(f'Unsupported case shape: {case_arr.shape}')

        if case_arr.shape[-1] != inferred_target_len:
            case_arr = np.stack([_resample_1d(channel, inferred_target_len) for channel in case_arr], axis=0)
        processed_cases.append(case_arr.astype(np.float32, copy=False))

    return np.stack(processed_cases, axis=0).astype(np.float32)


def _encode_labels(train_y, test_y) -> tuple[np.ndarray, np.ndarray]:
    train_arr = np.asarray(train_y)
    test_arr = np.asarray(test_y)
    all_labels = np.concatenate([train_arr.astype(str), test_arr.astype(str)], axis=0)
    classes = sorted(set(all_labels.tolist()))
    mapping = {label: idx for idx, label in enumerate(classes)}

    train_encoded = np.asarray([mapping[str(v)] for v in train_arr], dtype=np.int64)
    test_encoded = np.asarray([mapping[str(v)] for v in test_arr], dtype=np.int64)
    return train_encoded, test_encoded


def _standardize_train_test(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = train_x.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((test_x - mean) / std).astype(np.float32),
    )


def _load_archive_dataset(data_root: str, dataset_name: str):
    canonical_name = ARCHIVE_DATASETS[dataset_name]
    try:
        from aeon.datasets import load_classification
    except Exception as e:  # pragma: no cover - optional dependency path
        raise ImportError(
            'aeon is required for archive time-series datasets. Install aeon to use ECG5000, '
            'ElectricDevices, FordA, Wafer, TwoPatterns, or BasicMotions.'
        ) from e

    train_payload = load_classification(
        canonical_name,
        split='train',
        extract_path=data_root,
        return_metadata=True,
        load_equal_length=False,
        load_no_missing=False,
    )
    test_payload = load_classification(
        canonical_name,
        split='test',
        extract_path=data_root,
        return_metadata=True,
        load_equal_length=False,
        load_no_missing=False,
    )

    if len(train_payload) == 3:
        train_x_raw, train_y_raw, train_meta = train_payload
    else:  # pragma: no cover - compatibility fallback
        train_x_raw, train_y_raw = train_payload
        train_meta = {}

    if len(test_payload) == 3:
        test_x_raw, test_y_raw, test_meta = test_payload
    else:  # pragma: no cover - compatibility fallback
        test_x_raw, test_y_raw = test_payload
        test_meta = train_meta

    target_len = None
    if isinstance(train_x_raw, np.ndarray) and train_x_raw.ndim == 3:
        target_len = int(train_x_raw.shape[-1])

    train_x = _ensure_nct_array(train_x_raw, target_len=target_len)
    test_x = _ensure_nct_array(test_x_raw, target_len=train_x.shape[-1])

    train_y, test_y = _encode_labels(train_y_raw, test_y_raw)
    train_x, test_x = _standardize_train_test(train_x, test_x)

    n_classes = len(np.unique(np.concatenate([train_y, test_y], axis=0)))
    spec = DatasetSpec(
        name=dataset_name,
        input_channels=int(train_x.shape[1]),
        n_classes=int(n_classes),
        seq_len=int(train_x.shape[2]),
        source=f"aeon:{canonical_name}:{train_meta.get('problemname', canonical_name)}",
    )
    return train_x, train_y, test_x, test_y, spec


def _make_synthetic_sample(
    *,
    rng: np.random.RandomState,
    seq_len: int,
    n_channels: int,
    label: int,
    shifted: bool = False,
    multiscale: bool = False,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num=seq_len, dtype=np.float32)
    channels = []
    base_freq = [2.0, 5.0, 9.0, 13.0][label % 4]

    for c in range(n_channels):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.7, 1.3)
        sig = amp * np.sin(2.0 * np.pi * (base_freq + 0.35 * c) * t + phase)

        if multiscale:
            sig += 0.45 * np.sin(2.0 * np.pi * (base_freq * 0.5 + c) * t + 0.5 * phase)
            sig += 0.25 * np.cos(2.0 * np.pi * (base_freq * 1.8 + 0.2 * c) * t - phase)

        trend = 0.12 * (label - 1.5) * (t - 0.5)
        sig += trend

        if shifted:
            shift = rng.randint(0, seq_len // 10 + 1)
            sig = np.roll(sig, shift)
            sig[:shift] = 0.0
            burst_pos = rng.randint(seq_len // 8, seq_len - seq_len // 8)
            burst = np.exp(-((np.arange(seq_len) - burst_pos) ** 2) / (2.0 * (seq_len / 30.0) ** 2))
            sig += 0.15 * burst.astype(np.float32)

        sig += rng.normal(loc=0.0, scale=0.08 + 0.02 * c, size=seq_len).astype(np.float32)
        channels.append(sig.astype(np.float32))

    return np.stack(channels, axis=0).astype(np.float32)


def _load_synthetic_dataset(dataset_name: str):
    train_rng = np.random.RandomState(2026)
    test_rng = np.random.RandomState(3036)

    if dataset_name == 'synthetic_sines':
        n_train, n_test, seq_len, n_channels, n_classes = 512, 256, 192, 1, 4
        shifted = False
        multiscale = False
    elif dataset_name == 'synthetic_shiftmix':
        n_train, n_test, seq_len, n_channels, n_classes = 512, 256, 224, 1, 4
        shifted = True
        multiscale = True
    elif dataset_name == 'synthetic_multiscale':
        n_train, n_test, seq_len, n_channels, n_classes = 448, 224, 160, 3, 4
        shifted = True
        multiscale = True
    else:
        raise ValueError(f'Unknown synthetic dataset: {dataset_name}')

    train_labels = np.asarray([i % n_classes for i in range(n_train)], dtype=np.int64)
    test_labels = np.asarray([i % n_classes for i in range(n_test)], dtype=np.int64)
    train_rng.shuffle(train_labels)
    test_rng.shuffle(test_labels)

    train_x = np.stack(
        [
            _make_synthetic_sample(
                rng=train_rng,
                seq_len=seq_len,
                n_channels=n_channels,
                label=int(label),
                shifted=shifted,
                multiscale=multiscale,
            )
            for label in train_labels
        ],
        axis=0,
    )
    test_x = np.stack(
        [
            _make_synthetic_sample(
                rng=test_rng,
                seq_len=seq_len,
                n_channels=n_channels,
                label=int(label),
                shifted=shifted,
                multiscale=multiscale,
            )
            for label in test_labels
        ],
        axis=0,
    )

    train_x, test_x = _standardize_train_test(train_x.astype(np.float32), test_x.astype(np.float32))
    spec = DatasetSpec(
        name=dataset_name,
        input_channels=n_channels,
        n_classes=n_classes,
        seq_len=seq_len,
        source='synthetic',
    )
    return train_x, train_labels, test_x, test_labels, spec


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
    dataset_name = dataset_name.lower()
    if dataset_name in VISION_DATASETS:
        train_full = SequentialVisionDataset(data_root, dataset_name=dataset_name, train=True, permute=permute, seed=seed)
        test_ds = SequentialVisionDataset(data_root, dataset_name=dataset_name, train=False, permute=permute, seed=seed)
    elif dataset_name in ARCHIVE_DATASETS:
        train_x, train_y, test_x, test_y, spec = _load_archive_dataset(data_root, dataset_name)
        train_full = InMemorySequenceDataset(train_x, train_y, spec, permute=permute, seed=seed)
        test_ds = InMemorySequenceDataset(test_x, test_y, spec, permute=permute, seed=seed)
    elif dataset_name in SYNTHETIC_DATASETS:
        train_x, train_y, test_x, test_y, spec = _load_synthetic_dataset(dataset_name)
        train_full = InMemorySequenceDataset(train_x, train_y, spec, permute=permute, seed=seed)
        test_ds = InMemorySequenceDataset(test_x, test_y, spec, permute=permute, seed=seed)
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported datasets: {', '.join(DATASET_CHOICES)}"
        )

    val_size = max(1, int(len(train_full) * val_ratio))
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
