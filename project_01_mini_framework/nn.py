import random
from value import Value


class Neuron:
    def __init__(self, nin: int, nonlinear: bool = True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlinear = nonlinear

    def __call__(self, x: list[Value]) -> Value:
        assert len(x) == len(self.w)
        out = sum((wi * xi for wi, xi, in zip(self.w,x))) + self.b
        if self.nonlinear:
            out = out.relu()
        return out

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int, nonlinear: bool = True):
        self.neurons = [Neuron(nin, nonlinear) for _ in range(nout)]

    def __call__(self, x):
        # TODO: return a list containing each neuron's output.
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin: int, widths: list[int]):
        assert len(widths) > 0
        sizes = [nin] + list(widths)

        # All layers except the last should use ReLU.
        n_layers = len(sizes) - 1
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlinear=i != n_layers - 1)
            for i in range(n_layers)
        ]

    def __call__(self, x):
        # TODO: pass x through each layer in order.
        # Return the final list of outputs.
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]