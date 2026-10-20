"""Shared helpers for Lab 3: seeding and the Fashion-MNIST pipeline."""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 128
LAB_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = LAB_DIR.parent
DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = LAB_DIR / "results"

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_datasets():
    """55k train / 5k validation from the official 60k train set; official 10k test.
    Downloads into <repo>/data if missing."""
    full_train = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=TRANSFORM)
    test = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=TRANSFORM)
    train, val = random_split(full_train, [55_000, 5_000],
                              generator=torch.Generator().manual_seed(SEED))
    return train, val, test


def make_loaders(batch_size: int = BATCH_SIZE):
    train, val, test = load_datasets()
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(SEED)),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )
