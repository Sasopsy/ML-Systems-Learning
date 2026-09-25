from __future__ import annotations


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.op = op

        self._backward = lambda: None


    def __add__(self, other: Value) -> Value:
        # Return a new Value holding the sum.
        # Record both input Values as its parents.
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other) -> Value:
        return self + (other if isinstance(other, Value) else Value(other))

    def __mul__(self, other: Value) -> Value:
        # Same idea, for multiplication.
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __sub__(self, other: Value) -> Value:
        return self + (-other)

    def __neg__(self):
        return self * Value(-1)

    def relu(self):
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node in visited:
                return
            visited.add(node)

            # recursively visit each parent
            for parent in node.parents:
                build_topo(parent)
            topo.append(node)

        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()