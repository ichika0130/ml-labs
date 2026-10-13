"""Shared helpers for Lab 2: seeding and the digits data pipeline."""
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
N_CLASSES = 10
LAB_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_DIR / "results"


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    out = np.zeros((y.size, n_classes))
    out[np.arange(y.size), y] = 1.0
    return out


def load_data():
    """Digits scaled to [0, 1], one-hot labels, 70/30 stratified split."""
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    return X_train, X_test, one_hot(y_train), one_hot(y_test)
