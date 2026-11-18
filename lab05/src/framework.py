"""Task 6: inside PyTorch. Graph inspection, eager vs compiled/traced execution,
tensor shape/dtype/contiguity, and the speed cost of a non-contiguous view.

All timings: CPU, 1 thread (stable on a mixed-core laptop), median after warm-up.
"""
import statistics
import time
import traceback

import torch
from torch import nn

from common import RESULTS_DIR, set_seeds, write_csv

torch.set_num_threads(1)
WARMUP = 10


class SmallMLP(nn.Module):
    """The make_moons model from task 5: 2 -> 8 -> 1."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class PointwiseChain(nn.Module):
    """No weights, only a chain of element-wise operations: the case operator fusion targets.
    Eager runs each operation as its own kernel and writes every intermediate to memory."""
    def forward(self, x):
        y = x * 1.5 + 0.5
        y = torch.tanh(y) * torch.sigmoid(x)
        y = y - 0.1 * x * x
        return torch.relu(y) + torch.exp(-x.abs())


class LargerMLP(nn.Module):
    """A bigger model, to see whether the answer changes with size: 784 -> 512 -> 512 -> 10."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 10))

    def forward(self, x):
        return self.net(x)


def median_ms(fn, runs):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


# ---- 1. graph inspection ---------------------------------------------------------------------
def autograd_graph(fn, depth=0, seen=None, lines=None):
    seen = set() if seen is None else seen
    lines = [] if lines is None else lines
    if fn is None or fn in seen:
        return lines
    seen.add(fn)
    lines.append("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        autograd_graph(nxt, depth + 1, seen, lines)
    return lines


def inspect_graphs():
    model = SmallMLP()
    x = torch.randn(4, 2)
    loss = model(x).sum()
    text = ["# Autograd graph of SmallMLP(x).sum(), from loss.grad_fn back to the leaves",
            *autograd_graph(loss.grad_fn), "",
            "# torch.fx graph (the forward computation as recorded operations)",
            str(torch.fx.symbolic_trace(model).graph), "",
            "# torch.jit.trace graph (inlined)", str(torch.jit.trace(model, x).inlined_graph)]
    (RESULTS_DIR / "graph_inspection.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text[:20]))


# ---- 2. eager vs compiled / traced -----------------------------------------------------------
def try_compile(model, x):
    """Returns (compiled model or None, first-call seconds, error message)."""
    try:
        torch._dynamo.reset()
        compiled = torch.compile(model)
        start = time.perf_counter()
        with torch.inference_mode():
            compiled(x)
        return compiled, time.perf_counter() - start, ""
    except Exception as e:  # noqa: BLE001 - record why compilation is unavailable
        return None, None, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"


def execution_modes():
    rows = []
    for name, cls, in_dim in (("SmallMLP 2-8-1", SmallMLP, 2), ("LargerMLP 784-512-512-10", LargerMLP, 784),
                              ("PointwiseChain (4096 values per sample)", PointwiseChain, 4096)):
        for batch, runs in ((1, 500), (256, 100)):
            set_seeds()
            model = cls().eval()
            x = torch.randn(batch, in_dim)
            with torch.inference_mode():
                ref = model(x)
            start = time.perf_counter()
            traced = torch.jit.trace(model, x)
            trace_s = time.perf_counter() - start
            compiled, compile_s, err = try_compile(model, x)
            variants = [("eager", model, 0.0, ""), ("jit.trace", traced, trace_s, "")]
            variants.append(("torch.compile", compiled, compile_s, err))
            for mode, m, setup_s, note in variants:
                row = {"model": name, "batch": batch, "mode": mode, "runs": runs}
                if m is None:
                    row.update({"median_ms": "", "per_sample_us": "", "speedup_vs_eager": "",
                                "setup_or_first_call_s": "", "max_abs_diff_vs_eager": "",
                                "note": note or "unavailable"})
                else:
                    with torch.inference_mode():
                        out = m(x)
                        ms = median_ms(lambda: m(x), runs)
                    row.update({"median_ms": round(ms, 4), "per_sample_us": round(ms * 1e3 / batch, 3),
                                "speedup_vs_eager": "", "setup_or_first_call_s": round(setup_s, 3),
                                "max_abs_diff_vs_eager": f"{(out - ref).abs().max().item():.1e}",
                                "note": note})
                rows.append(row)
                print(row)
    for r in rows:   # speedup relative to eager for the same model and batch
        eager = next(e for e in rows if e["model"] == r["model"] and e["batch"] == r["batch"]
                     and e["mode"] == "eager")
        if r["median_ms"] != "":
            r["speedup_vs_eager"] = round(eager["median_ms"] / r["median_ms"], 2)
    write_csv("execution_modes.csv", rows)


# ---- 3. tensor representation and contiguity -------------------------------------------------
def describe(name, t):
    return {"tensor": name, "shape": tuple(t.shape), "dtype": str(t.dtype), "stride": t.stride(),
            "is_contiguous": t.is_contiguous(), "shares_memory_with_x": t.data_ptr() == X_PTR}


X_PTR = None


def contiguity():
    global X_PTR
    set_seeds()
    x = torch.randn(4096, 4096)
    X_PTR = x.data_ptr()
    xt = x.t()                      # a view: same memory, strides swapped
    xc = xt.contiguous()            # a real copy laid out row by row
    write_csv("tensor_inspection.csv", [describe("x", x), describe("x.t()", xt),
                                        describe("x.t().contiguous()", xc),
                                        describe("x[:, ::2]", x[:, ::2])])
    v = torch.randn(4096)
    ops = {
        "elementwise t*2 + 1": lambda t: t * 2 + 1,
        "sum over rows t.sum(dim=1)": lambda t: t.sum(dim=1),
        "flatten t.reshape(-1)": lambda t: t.reshape(-1),
        "matrix-vector t @ v": lambda t: t @ v,
    }
    rows = []
    for op, f in ops.items():
        c = median_ms(lambda: f(xc), 20)
        n = median_ms(lambda: f(xt), 20)
        rows.append({"operation": op, "contiguous_ms": round(c, 3), "non_contiguous_view_ms": round(n, 3),
                     "slowdown": round(n / c, 2)})
    rows.append({"operation": "making it contiguous: xt.contiguous()", "contiguous_ms": "",
                 "non_contiguous_view_ms": round(median_ms(lambda: xt.contiguous(), 20), 3), "slowdown": ""})
    for r in rows:
        print(r)
    write_csv("contiguity_timing.csv", rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    inspect_graphs()
    execution_modes()
    contiguity()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
