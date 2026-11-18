"""Task 4 (helper): controlled training-throughput benchmark per batch size.

The 5-epoch runs of the batch-size study were made one after another, and the speed of the
machine drifted during the session (the same baseline config ran at ~4,400 samples/s early
and ~5,600 samples/s later). To compare batch sizes fairly, this script measures training
throughput for all batch sizes back to back, in 3 interleaved rounds, and reports the median.
Each measurement: 5 warm-up steps, then training steps until at least 12,800 samples.
"""
import csv
import statistics
import time

from torch import nn

from pipeline import RESULTS_DIR, build_model, build_optimizer, load_config, make_loaders, set_seeds

BATCH_SIZES = [32, 64, 128, 256, 512]
ROUNDS = 3
MIN_SAMPLES = 12_800
WARMUP_STEPS = 5


def measure(bs):
    cfg = load_config(overrides=[f"data.batch_size={bs}"])
    set_seeds(cfg["seed"])
    loader, *_ = make_loaders(cfg)
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    n, start = 0, None
    for step, (x, y) in enumerate(loader):
        if step == WARMUP_STEPS:
            start = time.perf_counter()
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()
        if start is not None:
            n += y.size(0)
            if n >= MIN_SAMPLES:
                break
    return n / (time.perf_counter() - start)


def main():
    results = {bs: [] for bs in BATCH_SIZES}
    for r in range(ROUNDS):
        for bs in BATCH_SIZES:
            results[bs].append(measure(bs))
            print(f"round {r + 1} batch {bs}: {results[bs][-1]:.0f} samples/s", flush=True)
    rows = [{"batch_size": bs, "median_samples_per_s": round(statistics.median(v), 1),
             "rounds": " ".join(f"{x:.0f}" for x in v)} for bs, v in results.items()]
    with open(RESULTS_DIR / "throughput_bench.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
