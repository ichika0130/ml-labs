"""Task 3: vary DataLoader num_workers in [0, 1, 2, 4, 8].

For each value we time (a) one pass over the training loader with no model ("data only")
and (b) one full training epoch. If the training epoch gets faster when workers are added,
data loading was a bottleneck; once it stops improving, compute is the bottleneck.
"""
import csv
import os
import time

import torch
from torch import nn

from pipeline import RESULTS_DIR, build_model, build_optimizer, load_config, make_loaders, set_seeds

WORKERS = [0, 1, 2, 4, 8]


def main():
    rows = []
    for nw in WORKERS:
        cfg = load_config(overrides=[f"data.num_workers={nw}"])
        set_seeds(cfg["seed"])
        loader, *_ = make_loaders(cfg)
        start = time.perf_counter()
        n = sum(x.size(0) for x, _ in loader)
        data_s = time.perf_counter() - start

        model = build_model(cfg)
        optimizer = build_optimizer(cfg, model)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        start = time.perf_counter()
        for x, y in loader:
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
        train_s = time.perf_counter() - start
        row = {"num_workers": nw, "data_only_s": round(data_s, 2), "train_epoch_s": round(train_s, 2),
               "train_samples_per_s": round(n / train_s, 1), "data_samples_per_s": round(n / data_s, 1)}
        rows.append(row)
        print(row, flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "num_workers_study.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CPU cores visible: {os.cpu_count()}, torch threads: {torch.get_num_threads()}")


if __name__ == "__main__":
    main()
