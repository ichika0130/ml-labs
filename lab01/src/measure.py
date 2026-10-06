"""Task 3: measure training time, single-sample latency, model size, and peak memory.

Memory is measured in a fresh subprocess per (model, phase) so that earlier runs do not
raise the RSS high-water mark. That subprocess does not import torch: the models do not
use it, and importing it alone adds ~110 MB RSS that would be misattributed to the model.
"""
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import warnings

import joblib
import psutil
from memory_profiler import memory_usage
from sklearn.exceptions import ConvergenceWarning

from common import RESULTS_DIR, load_split, make_models, set_seeds

TRAIN_REPEATS = 5
INFER_REPEATS = 100
MEM_INTERVAL = 0.001  # seconds between RSS samples

# The spec fixes LogisticRegression(max_iter=1000) on unscaled data; baseline.py shows the warning once.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def time_training(factory, X, y):
    factory().fit(X, y)  # warm-up
    times = []
    for _ in range(TRAIN_REPEATS):
        model = factory()
        start = time.perf_counter()
        model.fit(X, y)
        times.append(time.perf_counter() - start)
    return statistics.median(times), times


def time_single_inference(model, x):
    model.predict(x)  # warm-up
    times = []
    for _ in range(INFER_REPEATS):
        start = time.perf_counter()
        model.predict(x)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def measure_memory_child(name: str, phase: str) -> None:
    """Run in a fresh process: print JSON with baseline and peak RSS for one phase."""
    set_seeds(include_torch=False)
    X_train, X_test, y_train, _ = load_split()
    factory = make_models()[name]
    if phase == "train":
        func = lambda: factory().fit(X_train, y_train)
    else:
        model = joblib.load(RESULTS_DIR / f"model_{name}.joblib")
        func = lambda: [model.predict(X_test[:1]) for _ in range(INFER_REPEATS)]
    baseline = rss_mb()
    peak = memory_usage((func, ()), interval=MEM_INTERVAL, max_usage=True)
    print(json.dumps({"baseline_mb": baseline, "peak_mb": max(peak, baseline)}))


def measure_memory(name: str, phase: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-W", "ignore", __file__, "--memory", name, phase],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    x_one = X_test[:1]
    RESULTS_DIR.mkdir(exist_ok=True)

    rows, models = [], {}
    for name, factory in make_models().items():
        set_seeds()
        train_median, train_runs = time_training(factory, X_train, y_train)

        set_seeds()
        model = factory().fit(X_train, y_train)
        models[name] = model
        infer_median = time_single_inference(model, x_one)

        path = RESULTS_DIR / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_bytes = path.stat().st_size

        train_mem = measure_memory(name, "train")
        infer_mem = measure_memory(name, "infer")

        rows.append({
            "model": name,
            "train_time_median_ms": round(train_median * 1e3, 3),
            "train_time_runs_ms": " ".join(f"{t * 1e3:.3f}" for t in train_runs),
            "infer_latency_median_ms": round(infer_median * 1e3, 4),
            "model_size_bytes": size_bytes,
            "model_size_kb": round(size_bytes / 1024, 2),
            "train_peak_rss_mb": round(train_mem["peak_mb"], 1),
            "train_rss_delta_mb": round(train_mem["peak_mb"] - train_mem["baseline_mb"], 2),
            "infer_peak_rss_mb": round(infer_mem["peak_mb"], 1),
            "infer_rss_delta_mb": round(infer_mem["peak_mb"] - infer_mem["baseline_mb"], 2),
        })
        print(rows[-1])

    joblib.dump(models, RESULTS_DIR / "model.joblib")

    with open(RESULTS_DIR / "system_cost.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--memory":
        measure_memory_child(sys.argv[2], sys.argv[3])
    else:
        main()
