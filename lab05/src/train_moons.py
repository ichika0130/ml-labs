"""Task 5: train a 2 -> 8 -> 1 network on make_moons with the engine and with PyTorch.

Both versions use the same data, the same initial weights, full-batch gradient descent,
ReLU in the hidden layer, a sigmoid output and binary cross-entropy (BCE) loss, all in
float64. Their loss curves should therefore match, which is a second check of the engine.
Speed: the whole training run is timed 5 times for each, and the median is reported.
"""
import math
import statistics
import time

import numpy as np
import torch
from sklearn.datasets import make_moons

from common import SEED, set_seeds, write_csv
from engine import Value

N_SAMPLES = 100
NOISE = 0.1
HIDDEN = 8
STEPS = 400
LR = 1.0
RUNS = 5


def init_weights():
    """PyTorch-style uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) from a fixed seed."""
    rng = np.random.default_rng(SEED)
    b1, b2 = 1 / math.sqrt(2), 1 / math.sqrt(HIDDEN)
    return {"W1": rng.uniform(-b1, b1, (2, HIDDEN)), "c1": rng.uniform(-b1, b1, HIDDEN),
            "W2": rng.uniform(-b2, b2, (HIDDEN, 1)), "c2": rng.uniform(-b2, b2, 1)}


# ---- engine ----------------------------------------------------------------------------------
def train_engine(X, y, w0):
    W1 = [[Value(w0["W1"][i, j]) for j in range(HIDDEN)] for i in range(2)]
    c1 = [Value(v) for v in w0["c1"]]
    W2 = [Value(w0["W2"][j, 0]) for j in range(HIDDEN)]
    c2 = Value(w0["c2"][0])
    params = [w for row in W1 for w in row] + c1 + W2 + [c2]

    def forward(x):
        h = [(x[0] * W1[0][j] + x[1] * W1[1][j] + c1[j]).relu() for j in range(HIDDEN)]
        return (sum((h[j] * W2[j] for j in range(HIDDEN)), c2)).sigmoid()

    losses, nodes = [], 0
    for _ in range(STEPS):
        total = Value(0.0)
        for xi, yi in zip(X, y):
            p = forward(xi)
            total = total + (-(yi * p.log() + (1 - yi) * (1 - p).log()))
        loss = total / len(X)
        for p_ in params:
            p_.grad = 0.0
        nodes = len(loss.backward())
        for p_ in params:
            p_.data -= LR * p_.grad
        losses.append(loss.data)
    acc = np.mean([(forward(xi).data > 0.5) == yi for xi, yi in zip(X, y)])
    return losses, acc, nodes


# ---- PyTorch ---------------------------------------------------------------------------------
def train_torch(X, y, w0):
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    W1 = torch.tensor(w0["W1"], requires_grad=True)
    c1 = torch.tensor(w0["c1"], requires_grad=True)
    W2 = torch.tensor(w0["W2"], requires_grad=True)
    c2 = torch.tensor(w0["c2"], requires_grad=True)
    params = [W1, c1, W2, c2]

    def forward(x):
        return torch.sigmoid(torch.relu(x @ W1 + c1) @ W2 + c2)

    losses = []
    for _ in range(STEPS):
        p = forward(Xt)
        loss = -(yt * torch.log(p) + (1 - yt) * torch.log(1 - p)).mean()
        for p_ in params:
            p_.grad = None
        loss.backward()
        with torch.no_grad():
            for p_ in params:
                p_ -= LR * p_.grad
        losses.append(loss.item())
    with torch.no_grad():
        acc = ((forward(Xt) > 0.5).double() == yt).double().mean().item()
    return losses, acc


def timed(fn, *args):
    times, out = [], None
    for _ in range(RUNS):
        start = time.perf_counter()
        out = fn(*args)
        times.append(time.perf_counter() - start)
    return out, times


def main() -> None:
    set_seeds()
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    w0 = init_weights()
    (eng_losses, eng_acc, nodes), eng_t = timed(train_engine, X, y, w0)
    (tor_losses, tor_acc), tor_t = timed(train_torch, X, y, w0)

    curve_diff = max(abs(a - b) for a, b in zip(eng_losses, tor_losses))
    write_csv("moons_loss_curve.csv", [
        {"step": i + 1, "loss_engine": f"{a:.10f}", "loss_torch": f"{b:.10f}"}
        for i, (a, b) in enumerate(zip(eng_losses, tor_losses))])
    rows = []
    for name, losses, acc, times in (("engine", eng_losses, eng_acc, eng_t),
                                     ("pytorch", tor_losses, tor_acc, tor_t)):
        rows.append({"implementation": name, "samples": N_SAMPLES, "steps": STEPS,
                     "initial_loss": round(losses[0], 6), "final_loss": round(losses[-1], 6),
                     "train_accuracy": round(acc, 4),
                     "median_total_s": round(statistics.median(times), 4),
                     "median_per_step_ms": round(statistics.median(times) / STEPS * 1e3, 3),
                     "runs_s": " ".join(f"{t:.4f}" for t in times)})
    rows[0]["graph_nodes_per_step"] = nodes
    rows[1]["graph_nodes_per_step"] = ""
    for r in rows:
        r["max_loss_curve_diff"] = f"{curve_diff:.2e}"
        r["speedup_vs_engine"] = round(statistics.median(eng_t) / statistics.median(
            tor_t if r["implementation"] == "pytorch" else eng_t), 1)
        print(r)
    write_csv("moons_speed.csv", rows)


if __name__ == "__main__":
    main()
