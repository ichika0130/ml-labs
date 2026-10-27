"""Shared helpers for Lab 4: paths, config, seeding, hashing."""
import hashlib
import random
from pathlib import Path

import numpy as np
import yaml

LAB_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = LAB_DIR.parent
DATA_DIR = REPO_DIR / "data" / "lab04"          # git-ignored
RESULTS_DIR = LAB_DIR / "results"
CONFIG_PATH = LAB_DIR / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df, path: Path) -> None:
    """Deterministic CSV: fixed float format and line endings, so the hash is stable."""
    df.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")


def numeric_columns(cfg: dict) -> list:
    return [c for c, t in cfg["schema"]["types"].items() if t == "int"]


def categorical_columns(cfg: dict) -> list:
    target = cfg["dataset"]["target_column"]
    return [c for c, t in cfg["schema"]["types"].items() if t == "category" and c != target]
