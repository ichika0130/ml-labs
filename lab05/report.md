# Lab 5: Frameworks and automatic differentiation

## 1. Goal

Build a small automatic-differentiation engine for scalars, check that its gradients are correct, train a tiny network with it, and compare it with PyTorch. Then look inside PyTorch: its computation graph, eager versus compiled or traced execution, and how tensor layout (contiguity) changes the speed of an operation.

## 2. Method

**Environment.** Python 3.10.12, PyTorch 2.3.1, NumPy 1.26.4, scikit-learn 1.5.0 (versions in `results/versions.txt`). These results were measured in a Linux ARM virtual machine running on the MacBook (4 vCPUs, 3.9 GB RAM), not directly in macOS as in Labs 1–4, so absolute times are not comparable with earlier labs. PyTorch uses 4 threads. Seed 42. All results are reproduced with `bash lab05/src/run_all.sh` (it uses the repo `.venv`; set `PY=...` to use another Python).

**Engine (`src/engine.py`, tasks 1–3).** Each `Value` stores:

| Field | Meaning |
|---|---|
| `data` | the number itself |
| `grad` | d(output)/d(this value), filled in by `backward()` |
| `_parents` | the Values it was computed from (the edges of the graph) |
| `_backward` | its local derivative rule: how to pass its gradient on to its parents |

Every operation (`+`, `*`, `**`, `relu`, plus `exp`, `log`, `sigmoid` and the derived `-`, `/` needed for training) creates a new Value and records its parents and rule, so the forward pass builds the graph automatically. `backward()` sorts the graph topologically, sets the output gradient to 1, visits the nodes in reverse order and calls each local rule. The rules use `+=`, so a value used several times collects the gradient from every use instead of overwriting it.

**Gradient check (`src/verify.py`, task 4).** a, b and c are vectors of 8 random values; the engine works on scalars, so `.sum()` adds the 8 results. The same expression is computed in PyTorch (float64) and with central differences (ε = 1e-6). Besides the manual's expression, two extra checks: `(a*b + c).relu().sum()`, where ReLU really cuts negative values (in the manual's expression the square is never negative, so ReLU never cuts), and `(a*a*b + a*c).sum()`, where the same value is used several times (tests accumulation).

**Training (`src/train_moons.py`, task 5).** `make_moons` (100 points, noise 0.1). Network 2 → 8 (ReLU) → 1 (sigmoid), BCE loss, full-batch gradient descent, lr 1.0, 400 steps, float64. The engine and PyTorch start from the same weights. Each full training run is timed 5 times; the median is reported.

**Inside PyTorch (`src/framework.py`, task 6).**
- Graph: the autograd graph of a small model (walking `grad_fn`), the `torch.fx` graph and the `torch.jit.trace` graph (`results/graph_inspection.txt`).
- Execution modes: eager, `torch.jit.trace` and `torch.compile` on three models (the 2-8-1 MLP, a 784-512-512-10 MLP and a chain of 8 pointwise operations), batch 1 and 256. Median of 500 (batch 1) or 100 (batch 256) calls after warm-up. The one-time cost (tracing or the first compiled call) is recorded separately.
- Tensors: shape, dtype, strides and contiguity of a 4096×4096 tensor, its transpose `x.t()` (a view) and `x[:, ::2]`. Then four operations timed on `x.t()` (non-contiguous) and on `x.t().contiguous()` (a real copy), median of 20.

## 3. Results

### 3.1 Gradient check (`results/verify_gradients.csv`)

| Expression | Graph nodes | Max \|engine − PyTorch\| | Max \|engine − numerical\| | < 1e-6 |
|---|---|---|---|---|
| `((a*b + c)**2).relu().sum()` (manual) | 65 | 0 | 1.01e-9 | yes |
| `(a*b + c).relu().sum()` (ReLU cuts) | 57 | 0 | 2.94e-10 | yes |
| `(a*a*b + a*c).sum()` (value reused) | 65 | 0 | 5.92e-10 | yes |

The engine and PyTorch agree exactly. The small difference to the numerical gradient comes from the finite-difference approximation itself.

### 3.2 Training on make_moons (`results/moons_speed.csv`, `results/moons_loss_curve.csv`)

| | Engine | PyTorch |
|---|---|---|
| Loss, step 1 → step 400 | 0.570161 → 0.052897 | 0.570161 → 0.052897 |
| Training accuracy | 100% | 100% |
| Median time for 400 steps | 27.65 s | 0.0554 s |
| Time per step | 69.1 ms | 0.138 ms |
| Graph nodes per step | 9,436 | (whole-tensor ops) |

The two loss curves differ by at most 5.0e-16 over all 400 steps. PyTorch is **about 500× faster** (27.65 s / 0.0554 s = 499).

### 3.3 Graph inspection (`results/graph_inspection.txt`)

The autograd graph of the 2-8-1 MLP goes `SumBackward0 → SigmoidBackward0 → AddmmBackward0 → ReluBackward0 → AddmmBackward0`, with an `AccumulateGrad` node for each weight and bias. These are the same steps as in my engine, only at tensor level: one `Addmm` node (matrix multiply plus bias) replaces all the scalar `*` and `+` nodes that my engine builds for a layer.

### 3.4 Eager vs compiled vs traced (`results/execution_modes.csv`)

Speed-up relative to eager for the same model and batch (> 1 means faster):

| Model | Batch | Eager (ms) | jit.trace | torch.compile | One-time cost of compile |
|---|---|---|---|---|---|
| SmallMLP 2-8-1 | 1 | 0.0424 | 1.18× | **3.09×** (0.0137 ms) | 1.75 s |
| SmallMLP 2-8-1 | 256 | 0.0500 | 1.10× | 0.87× | 1.06 s |
| MLP 784-512-512-10 | 1 | 0.472 | 1.02× | 0.83× | 0.98 s |
| MLP 784-512-512-10 | 256 | 1.912 | 1.05× | 0.86× | 0.88 s |
| PointwiseChain | 1 | 0.080 | 1.05× | 0.74× | 0.84 s |
| PointwiseChain | 256 | 24.04 | 0.95× | 0.54× | 1.10 s |

All modes give the same outputs (max difference 0). `torch.compile` was faster only in one case, the tiny model at batch 1. There it saves 0.0287 ms per call, so the 1.75 s compile cost pays back after about 61,000 calls.

### 3.5 Tensor layout and contiguity (`results/tensor_inspection.csv`, `results/contiguity_timing.csv`)

| Tensor | Shape | dtype | Strides | Contiguous | Shares memory with x |
|---|---|---|---|---|---|
| `x` | 4096×4096 | float32 | (4096, 1) | yes | yes |
| `x.t()` | 4096×4096 | float32 | (1, 4096) | no | yes |
| `x.t().contiguous()` | 4096×4096 | float32 | (4096, 1) | yes | no (copy) |
| `x[:, ::2]` | 4096×2048 | float32 | (4096, 2) | no | yes |

| Operation | Contiguous (ms) | Non-contiguous view (ms) | Slow-down |
|---|---|---|---|
| Elementwise `t*2 + 1` | 35.68 | 31.37 | 0.88× (none) |
| Sum over rows `t.sum(dim=1)` | 1.40 | 6.31 | **4.5×** |
| Matrix-vector `t @ v` | 7.25 | 33.53 | **4.6×** |
| Flatten `t.reshape(-1)` | 0.001 | 45.00 | must copy |
| Making it contiguous `xt.contiguous()` | – | 44.88 | – |

## 4. Conclusions

1. My engine computes exactly the same gradients as PyTorch, but it is about 500 times slower. The gradients match PyTorch with difference 0 and the numerical gradients within 1e-9, and the loss curves on make_moons are the same within 5e-16. But 400 training steps take 27.65 s with my engine and 0.055 s with PyTorch. The reason is not the math, because the math is the same. My engine makes one Python object for every single number, so one training step builds 9,436 graph nodes and calls 9,436 small Python functions. PyTorch does the same work with a few operations on whole tensors, and each operation runs in fast C++ code. So the speed of a framework comes from working on whole tensors, not from a different way of computing gradients.
2. Compilation does not always make a model faster. In my tests it only helped a very small model called many times. torch.compile was 3.1 times faster only for the 2-8-1 MLP at batch 1 (0.042 ms → 0.014 ms). In the other 5 cases it was 13–46% slower, and jit.trace changed the speed by only −5% to +18%. Compiling also has a one-time cost of about 1–1.7 s, so even in the best case it needs about 61,000 calls before it saves time. For a tiny model at batch 1, most of the time is Python overhead, and compilation removes this overhead. For bigger operations, the time is in the computation itself, which eager mode already does in optimized kernels. So compilation helps when a small model is called very many times, for example when serving a model, and not for short experiments. We should always measure it after warm-up and include the compile time.
3. The memory layout of a tensor changes the speed of an operation, even when the values are the same. A transposed tensor (`x.t()`) is only a view: it uses the same memory with swapped strides, so its rows are not stored next to each other. On this view, a sum over rows was 4.5 times slower and a matrix-vector product 4.6 times slower, and reshape had to make a copy (45 ms instead of almost 0). Only the elementwise operation did not slow down, because it can read the memory in any order. Calling `.contiguous()` once costs 44.9 ms, which is about the time lost by 2 matrix-vector products or 9 row sums on the view. So if a transposed tensor will be used many times, we should make it contiguous once, and if it is used only once, the copy is not worth it.
