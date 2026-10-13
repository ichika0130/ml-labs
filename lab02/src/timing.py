"""Task 7: time forward and backward passes separately (median of 100 runs after a warm-up).

Each pass is also timed with only its matrix multiplications (same shapes, nothing else), and
matmul FLOPs are counted analytically, so the forward/backward gap can be traced to matmuls.
"""
import csv
import statistics
import time

import numpy as np

from common import RESULTS_DIR, SEED, load_data, set_seeds
from mlp import backward, forward, init_params

WIDTHS = (16, 64)
BATCHES = (32, 1257)  # training mini-batch, and the whole training set
REPEATS = 100


def median_us(fn) -> float:
    fn()  # warm-up
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e6


def matmul_flops(batch, width, n_in=64, n_out=10):
    """2*m*n*k per matmul. Backward has two matmuls per layer (dW and dA) except the
    first layer, which needs no gradient w.r.t. the input."""
    fwd = 2 * batch * (n_in * width + width * width + width * n_out)
    bwd = 2 * batch * (2 * width * n_out + 2 * width * width + n_in * width)
    return fwd, bwd


def main() -> None:
    set_seeds()
    X_train, _, Y_train, _ = load_data()
    rows = []
    for width in WIDTHS:
        params = init_params(width, rng=np.random.default_rng(SEED))
        W1, W2, W3 = params["W1"], params["W2"], params["W3"]
        for batch in BATCHES:
            X, Y = X_train[:batch], Y_train[:batch]
            P, cache = forward(params, X)
            A1, A2 = cache["A1"], cache["A2"]
            G3 = P - Y                   # stand-ins with the backward pass's shapes
            G2 = G3 @ W3.T

            fwd_us = median_us(lambda: forward(params, X))
            bwd_us = median_us(lambda: backward(params, cache, P, Y))
            fwd_mm_us = median_us(lambda: (X @ W1, A1 @ W2, A2 @ W3))
            bwd_mm_us = median_us(lambda: (A2.T @ G3, G3 @ W3.T, A1.T @ G2, G2 @ W2.T, X.T @ G2))
            fwd_flops, bwd_flops = matmul_flops(batch, width)
            rows.append({
                "width": width, "batch": batch,
                "forward_median_us": round(fwd_us, 2), "backward_median_us": round(bwd_us, 2),
                "backward_over_forward": round(bwd_us / fwd_us, 2),
                "forward_matmul_only_us": round(fwd_mm_us, 2),
                "backward_matmul_only_us": round(bwd_mm_us, 2),
                "forward_matmul_mflops": round(fwd_flops / 1e6, 3),
                "backward_matmul_mflops": round(bwd_flops / 1e6, 3),
                "flop_ratio_bwd_over_fwd": round(bwd_flops / fwd_flops, 2),
                "forward_matmuls": 3, "backward_matmuls": 5,
            })
            print(rows[-1])

    with open(RESULTS_DIR / "timing.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
