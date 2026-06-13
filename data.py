"""Data loading and Dirichlet client partitioning for FashionMNIST."""

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = (0.2860,), (0.3530,)


def get_datasets(data_dir="./data"):
    """Download (if needed) and return FashionMNIST train/test datasets."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
    return train_ds, test_ds


def dirichlet_split(dataset, n_clients, beta, seed=42):
    """Partition `dataset` indices across `n_clients` via a Dirichlet(beta) split per class.

    Small beta (e.g. 0.1) -> highly heterogeneous (non-IID) client data.
    Large beta (e.g. 10) -> approximately IID client data.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    n_cls = int(labels.max()) + 1
    idx_per_client = [[] for _ in range(n_clients)]

    for c in range(n_cls):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet(np.repeat(beta, n_clients))
        props = (props * len(idx_c)).astype(int)
        props[props.argmin()] += len(idx_c) - props.sum()
        start = 0
        for i, p in enumerate(props):
            idx_per_client[i].extend(idx_c[start:start + p].tolist())
            start += p

    return idx_per_client


def build_client_loaders(train_ds, n_clients, beta, batch_size, seed=42):
    """Return one DataLoader per client using a Dirichlet(beta) partition."""
    idx = dirichlet_split(train_ds, n_clients, beta, seed)
    return [
        DataLoader(Subset(train_ds, i), batch_size=batch_size,
                   shuffle=True, drop_last=True)
        for i in idx
    ]


class CycleLoader:
    """Wraps a DataLoader so `.next()` cycles forever (re-shuffling on each pass)."""

    def __init__(self, loader):
        self.loader = loader
        self._iter = iter(loader)

    def next(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)
