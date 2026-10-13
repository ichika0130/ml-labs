"""Tasks 3-5: forward-pass shape check, initial-loss sanity check, numerical gradient check.

Gradient check design (see report): central differences in float64 have an absolute noise
floor of ~1e-11 here (the loss is ~2.3, and its rounding error is divided by 2*eps). The relative error
is therefore ~1e-11 / |gradient|, and with N(0, 0.01) weights the batch-averaged gradients
are ~1e-5, right at the 1e-6 threshold. The official check uses a single training sample,
where gradients are ~10x larger and no ReLU kinks were crossed, and draws its 10 weights
uniformly from those with a non-zero analytic gradient (zero gradients come from always-zero
pixels or inactive ReLUs and are trivially matched). A wider sweep over batch sizes is saved
alongside as evidence.
"""
import csv
import math

import numpy as np

from common import RESULTS_DIR, SEED, load_data, set_seeds
from mlp import backward, cross_entropy, forward, init_params

WIDTHS = (16, 64)
EPS = 1e-5
TOL = 1e-6
N_CHECK = 10
WEIGHT_NAMES = ("W1", "W2", "W3")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relu_pattern(cache):
    return np.concatenate([(cache["Z1"] > 0).ravel(), (cache["Z2"] > 0).ravel()])


def numerical_slope(params, X, Y, name, idx):
    """Central difference; also report whether the nudge flipped any ReLU (a kink)."""
    old = params[name][idx]
    losses, patterns = [], []
    for delta in (+EPS, -EPS):
        params[name][idx] = old + delta
        P, cache = forward(params, X)
        losses.append(cross_entropy(P, Y))
        patterns.append(relu_pattern(cache))
    params[name][idx] = old
    return (losses[0] - losses[1]) / (2 * EPS), bool((patterns[0] != patterns[1]).any())


def relative_error(a, n):
    denom = max(abs(a), abs(n))
    return abs(a - n) / denom if denom > 0 else 0.0


def sanity_checks(X_train, Y_train):
    rows = []
    for width in WIDTHS:
        params = init_params(width, rng=np.random.default_rng(SEED))
        P, _ = forward(params, X_train)
        rows.append({
            "width": width,
            "output_shape": str(P.shape),
            "shape_ok": P.shape == (X_train.shape[0], 10),
            "max_abs_rowsum_minus_1": f"{np.abs(P.sum(axis=1) - 1).max():.2e}",
            "rowsum_ok": bool(np.allclose(P.sum(axis=1), 1.0, atol=1e-6, rtol=0)),
            "initial_loss": f"{cross_entropy(P, Y_train):.4f}",
            "ln10": f"{math.log(10):.4f}",
        })
    return rows


def gradient_check(X_train, Y_train):
    X, Y = X_train[:1], Y_train[:1]
    rows = []
    for width in WIDTHS:
        params = init_params(width, rng=np.random.default_rng(SEED))
        P, cache = forward(params, X)
        grads = backward(params, cache, P, Y)
        candidates = [(name, idx) for name in WEIGHT_NAMES
                      for idx in zip(*np.nonzero(grads[name]))]
        rng = np.random.default_rng(SEED)
        for k in rng.choice(len(candidates), size=N_CHECK, replace=False):
            name, idx = candidates[k]
            idx = tuple(int(i) for i in idx)
            a = float(grads[name][idx])
            n, kink = numerical_slope(params, X, Y, name, idx)
            rel = relative_error(a, n)
            rows.append({
                "width": width, "param": name, "index": str(idx),
                "analytic": f"{a:.6e}", "numerical": f"{n:.6e}",
                "abs_error": f"{abs(a - n):.2e}", "rel_error": f"{rel:.2e}",
                "relu_kink": kink, "pass": rel < TOL,
            })
    return rows


def gradient_sweep(X_train, Y_train, n_weights: int = 400):
    """Evidence: pass rate vs. batch size, over many uniformly sampled weights."""
    rows = []
    for width in WIDTHS:
        for batch in (1, 4, 32):
            params = init_params(width, rng=np.random.default_rng(SEED))
            X, Y = X_train[:batch], Y_train[:batch]
            P, cache = forward(params, X)
            grads = backward(params, cache, P, Y)
            rng = np.random.default_rng(SEED)
            rel, abs_err, mags, kinks = [], [], [], 0
            for _ in range(n_weights):
                name = WEIGHT_NAMES[rng.integers(len(WEIGHT_NAMES))]
                idx = tuple(int(rng.integers(s)) for s in params[name].shape)
                a = float(grads[name][idx])
                n, kink = numerical_slope(params, X, Y, name, idx)
                kinks += kink
                if a != 0.0 or n != 0.0:
                    rel.append(relative_error(a, n))
                    abs_err.append(abs(a - n))
                    mags.append(abs(a))
            rel = np.array(rel)
            rows.append({
                "width": width, "batch": batch, "weights_sampled": n_weights,
                "nonzero_grad_fraction": f"{len(rel) / n_weights:.2f}",
                "pass_rate_nonzero": f"{np.mean(rel < TOL):.2f}",
                "median_abs_grad": f"{np.median(mags):.1e}",
                "median_abs_error": f"{np.median(abs_err):.1e}",
                "relu_kinks": kinks,
            })
    return rows


def main() -> None:
    set_seeds()
    X_train, _, Y_train, _ = load_data()
    RESULTS_DIR.mkdir(exist_ok=True)

    sanity = sanity_checks(X_train, Y_train)
    write_csv(RESULTS_DIR / "sanity_checks.csv", sanity)
    for r in sanity:
        print(r)

    check = gradient_check(X_train, Y_train)
    write_csv(RESULTS_DIR / "gradient_check.csv", check)
    for width in WIDTHS:
        passed = sum(r["pass"] for r in check if r["width"] == width)
        print(f"W={width}: {passed}/{N_CHECK} weights with relative error < {TOL:g}")

    sweep = gradient_sweep(X_train, Y_train)
    write_csv(RESULTS_DIR / "gradient_check_sweep.csv", sweep)
    for r in sweep:
        print(r)


if __name__ == "__main__":
    main()
