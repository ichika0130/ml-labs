"""Task 1-3: a scalar automatic-differentiation engine.

Every Value stores:
  data       the number itself
  grad       d(output)/d(this value), filled in by backward()
  _parents   the Values it was computed from (the edges of the computation graph)
  _backward  its local derivative rule: how to pass its grad on to its parents
Every operation creates a new Value and records its parents and rule, so running the
forward pass builds the computation graph automatically.
"""
import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._op = op                     # for printing the graph
        self._backward = lambda: None     # leaf nodes have no rule

    # ---- operations (task 2); each adds one node to the graph -------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():                  # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():                  # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():                  # d(a^k)/da = k * a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(self.data, 0.0), (self,), "relu")

        def _backward():                  # 1 if a > 0 else 0 (0 at the kink, like PyTorch)
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # exp and log are not in task 2, but the sigmoid and BCE loss of task 5 need them.
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():                  # d(e^a)/da = e^a
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():                  # d(ln a)/da = 1/a
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        return 1.0 / (1.0 + (-self).exp())

    # convenience operators, all built from the ones above
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Value(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    # ---- backward pass (task 3) -----------------------------------------------------------
    def backward(self):
        # 1. Topological sort: every node appears after all of its parents.
        order, visited = [], set()

        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                visit(parent)
            order.append(node)
        visit(self)
        # 2. Seed: d(output)/d(output) = 1.
        self.grad = 1.0
        # 3. Reverse order: a node passes its grad on only after it has received all of it.
        #    Each rule uses +=, so a value used several times accumulates every path.
        for node in reversed(order):
            node._backward()
        return order

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"


def topo_size(root):
    """Number of nodes in the graph below root (for reporting graph size)."""
    seen, stack = set(), [root]
    while stack:
        n = stack.pop()
        if id(n) not in seen:
            seen.add(id(n))
            stack.extend(n._parents)
    return len(seen)
