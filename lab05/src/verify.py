"""Task 4: check the engine against PyTorch autograd and numerical gradients.

Expression from the manual: ((a*b + c)**2).relu().sum(). The engine works on scalars, so
a, b and c are vectors of N Values and .sum() adds the N results. PyTorch computes the same
thing on float64 tensors. Numerical gradients use central differences.
Two extra checks: a value used several times (tests gradient accumulation), and
(a*b + c).relu().sum(), where the ReLU actually cuts negative inputs (in the manual's
expression the square is never negative, so the ReLU never cuts).
"""
import numpy as np
import torch

from common import SEED, set_seeds, write_csv
from engine import Value, topo_size

N = 8
EPS = 1e-6
TOL = 1e-6


def manual_expr(a, b, c):
    return sum(((x * y + z) ** 2).relu() for x, y, z in zip(a, b, c))


def manual_expr_torch(a, b, c):
    return ((a * b + c) ** 2).relu().sum()


def relu_expr(a, b, c):
    return sum((x * y + z).relu() for x, y, z in zip(a, b, c))


def relu_expr_torch(a, b, c):
    return (a * b + c).relu().sum()


def reuse_expr(a, b, c):
    # every a[i] is used three times: a*a*b + a*c
    return sum(x * x * y + x * z for x, y, z in zip(a, b, c))


def reuse_expr_torch(a, b, c):
    return (a * a * b + a * c).sum()


def numerical_grads(f_np, arrays):
    grads = []
    for k, arr in enumerate(arrays):
        g = np.zeros_like(arr)
        for i in range(arr.size):
            plus = [x.copy() for x in arrays]
            minus = [x.copy() for x in arrays]
            plus[k][i] += EPS
            minus[k][i] -= EPS
            g[i] = (f_np(*plus) - f_np(*minus)) / (2 * EPS)
        grads.append(g)
    return grads


def check(name, f_engine, f_torch, arrays):
    vals = [[Value(x) for x in arr] for arr in arrays]
    out = f_engine(*vals)
    out.backward()
    eng = [np.array([v.grad for v in vs]) for vs in vals]

    tens = [torch.tensor(arr, dtype=torch.float64, requires_grad=True) for arr in arrays]
    t_out = f_torch(*tens)
    t_out.backward()
    tor = [t.grad.numpy() for t in tens]

    f_np = lambda *xs: f_torch(*[torch.tensor(x, dtype=torch.float64) for x in xs]).item()
    num = numerical_grads(f_np, [a.copy() for a in arrays])

    d_torch = max(np.abs(e - t).max() for e, t in zip(eng, tor))
    d_num = max(np.abs(e - n).max() for e, n in zip(eng, num))
    row = {"expression": name, "n_inputs": 3 * N, "graph_nodes": topo_size(out),
           "output_engine": out.data, "output_torch": t_out.item(),
           "max_abs_diff_vs_torch": f"{d_torch:.2e}", "max_abs_diff_vs_numerical": f"{d_num:.2e}",
           "pass": bool(d_torch < TOL and d_num < TOL)}
    print(row)
    return row


def main() -> None:
    set_seeds()
    rng = np.random.default_rng(SEED)
    arrays = [rng.normal(size=N) for _ in range(3)]
    rows = [
        check("((a*b + c)**2).relu().sum()  [manual]", manual_expr, manual_expr_torch, arrays),
        check("(a*b + c).relu().sum()  [ReLU cuts]", relu_expr, relu_expr_torch, arrays),
        check("(a*a*b + a*c).sum()  [value reused]", reuse_expr, reuse_expr_torch, arrays),
    ]
    write_csv("verify_gradients.csv", rows)


if __name__ == "__main__":
    main()
