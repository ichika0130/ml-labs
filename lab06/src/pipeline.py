"""Shared pieces of the Lab 6 pipeline: config loading, seeding, data, model, optimizer, scheduler."""
import copy
import importlib.util
import random
import resource
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

LAB_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = LAB_DIR.parent
DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = LAB_DIR / "results"
DEFAULT_CONFIG = LAB_DIR / "src" / "config.yaml"

# Same normalisation as Lab 3.
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])


# ---- config ----------------------------------------------------------------------------------
def load_config(path=DEFAULT_CONFIG, overrides=()):
    """Load the YAML config and apply 'a.b=value' overrides (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def flatten(cfg, prefix=""):
    """{'a': {'b': 1}} -> {'a.b': 1}, for logging params to MLflow."""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


# ---- reproducibility -------------------------------------------------------------------------
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


# ---- data ------------------------------------------------------------------------------------
def make_loaders(cfg):
    """55k train / 5k validation (seed 42, as in Lab 3) and the official 10k test set.
    Returns the loaders and the generator that controls the shuffle order (saved in checkpoints)."""
    full = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=TRANSFORM)
    test = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=TRANSFORM)
    val_size = cfg["data"]["val_size"]
    train, val = random_split(full, [len(full) - val_size, val_size],
                              generator=torch.Generator().manual_seed(cfg["seed"]))
    gen = torch.Generator().manual_seed(cfg["seed"])
    bs, nw = cfg["data"]["batch_size"], cfg["data"]["num_workers"]
    extra = {"persistent_workers": False} if nw > 0 else {}
    train_loader = DataLoader(train, batch_size=bs, shuffle=True, generator=gen, num_workers=nw, **extra)
    val_loader = DataLoader(val, batch_size=512, num_workers=0)
    test_loader = DataLoader(test, batch_size=512, num_workers=0)
    return train_loader, val_loader, test_loader, gen


# ---- model -----------------------------------------------------------------------------------
def build_model(cfg):
    """Import the model class from Lab 3 so that exactly the same architecture is used."""
    spec = importlib.util.spec_from_file_location("lab03_models", REPO_DIR / "lab03" / "src" / "models.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lab03_models"] = module
    spec.loader.exec_module(module)
    return module.MODEL_FACTORIES[cfg["model"]["name"]]()


def build_optimizer(cfg, model):
    o = cfg["optimizer"]
    if o["name"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    if o["name"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=o["momentum"],
                               weight_decay=o["weight_decay"])
    raise ValueError(o["name"])


def build_scheduler(cfg, optimizer):
    s, epochs = cfg["scheduler"], cfg["train"]["epochs"]
    if s["name"] == "none":
        return None
    if s["name"] == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=s["step_size"], gamma=s["gamma"])
    if s["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if s["name"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=s["factor"],
                                                          patience=s["patience"])
    raise ValueError(s["name"])


# ---- memory ----------------------------------------------------------------------------------
def peak_rss_mb():
    """Peak resident memory of this process (ru_maxrss is KB on Linux, bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 / 1024 if sys.platform == "darwin" else peak / 1024
