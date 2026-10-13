"""Two-hidden-layer MLP in plain NumPy: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax.

Shapes (B = batch size):
    X  (B, 64)   W1 (64, W)  b1 (W,)   Z1, A1 (B, W)
                 W2 (W, W)   b2 (W,)   Z2, A2 (B, W)
                 W3 (W, 10)  b3 (10,)  Z3, P  (B, 10)
"""
import numpy as np

PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")


def init_params(width: int, n_in: int = 64, n_out: int = 10, std: float = 0.01, rng=None):
    """Weights ~ N(0, std^2) with std = 0.01; biases start at zero."""
    rng = rng if rng is not None else np.random.default_rng(42)
    return {
        "W1": rng.normal(0.0, std, (n_in, width)), "b1": np.zeros(width),
        "W2": rng.normal(0.0, std, (width, width)), "b2": np.zeros(width),
        "W3": rng.normal(0.0, std, (width, n_out)), "b3": np.zeros(n_out),
    }


def relu(z):
    return np.maximum(z, 0.0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability; does not change the result
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X):
    """Return class probabilities P (B, 10) and the cache needed by backward()."""
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)
    Z3 = A2 @ params["W3"] + params["b3"]
    P = softmax(Z3)
    return P, {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


def cross_entropy(P, Y, eps: float = 1e-12):
    """Mean cross-entropy over the batch; Y is one-hot."""
    return float(-np.sum(Y * np.log(P + eps)) / Y.shape[0])


def backward(params, cache, P, Y):
    """Gradients of the mean cross-entropy w.r.t. every parameter (chain rule, layer by layer)."""
    B = Y.shape[0]
    dZ3 = (P - Y) / B                       # softmax + cross-entropy combined
    grads = {"W3": cache["A2"].T @ dZ3, "b3": dZ3.sum(axis=0)}
    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (cache["Z2"] > 0)           # ReLU derivative
    grads["W2"] = cache["A1"].T @ dZ2
    grads["b2"] = dZ2.sum(axis=0)
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (cache["Z1"] > 0)
    grads["W1"] = cache["X"].T @ dZ1
    grads["b1"] = dZ1.sum(axis=0)
    return grads


def sgd_step(params, grads, lr: float):
    for name in PARAM_NAMES:
        params[name] -= lr * grads[name]


def accuracy(params, X, Y) -> float:
    P, _ = forward(params, X)
    return float(np.mean(P.argmax(axis=1) == Y.argmax(axis=1)))


def n_params(params) -> int:
    return sum(p.size for p in params.values())
