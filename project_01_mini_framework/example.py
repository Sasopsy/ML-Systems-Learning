from value import Value

a = Value(2)
b = Value(-3)
c = Value(10)

e = a * b + c
loss = e * e

assert loss.data == 16
assert loss.parents[0] is e
assert loss.parents[1] is e

loss.grad = 1.0

loss.backward()

assert e.grad == 8.0
assert a.grad == -24.0
assert b.grad == 16.0
assert c.grad == 8.0

x = Value(2)
u = x * x
p = u * Value(3)
q = u * Value(4)
loss = p + q

loss.backward()

assert u.grad == 7.0
assert x.grad == 28.0

def f(x):
    u = x * x
    return u * (x + 3.0)

x0 = 2.0
eps = 1e-6

numerical_grad = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)

x = Value(x0)
u = x * x
loss = u * (x + Value(3.0))
loss.backward()

assert abs(x.grad - numerical_grad) < 1e-5


x = Value(2.0)
loss = x.relu() * Value(3.0)
loss.backward()

assert x.grad == 3.0


for input_value, expected_output, expected_grad in [
    (-2.0, 0.0, 0.0),
    ( 0.0, 0.0, 0.0),
    ( 2.0, 2.0, 1.0),
]:
    x = Value(input_value)
    y = x.relu()
    y.backward()

    assert y.data == expected_output
    assert x.grad == expected_grad


from nn import Neuron

n = Neuron(2)
n.w[0].data = 2.0
n.w[1].data = -1.0
n.b.data = 0.5

out = n([Value(3.0), Value(4.0)])
assert out.data == 2.5

out.backward()
assert n.w[0].grad == 3.0
assert n.w[1].grad == 4.0
assert n.b.grad == 1.0

params = n.parameters()
assert len(params) == 3
assert params[0] is n.w[0]
assert params[1] is n.w[1]
assert params[2] is n.b


from nn import Layer

layer = Layer(2, 2, nonlinear=False)

# Assuming you name the neuron list self.neurons:
for neuron in layer.neurons:
    neuron.w[0].data = 2.0
    neuron.w[1].data = -1.0
    neuron.b.data = 0.5

x = [Value(3.0), Value(4.0)]
outputs = layer(x)

assert [v.data for v in outputs] == [2.5, 2.5]
assert len(layer.parameters()) == 6
assert len({id(p) for p in layer.parameters()}) == 6

sum(outputs).backward()

assert x[0].grad == 4.0
assert x[1].grad == -2.0

from nn import MLP

model = MLP(2, [2, 1])

# Assuming you store layers as self.layers:
for p in model.parameters():
    p.data = 1.0

x = [Value(1.0), Value(2.0)]
out = model(x)

# Each hidden neuron: relu(1 + 2 + 1) = 4
# Output neuron: 4 + 4 + 1 = 9
assert len(out) == 1
assert out[0].data == 9.0

params = model.parameters()
assert len(params) == 9
assert len({id(p) for p in params}) == 9

out[0].backward()
assert x[0].grad == 2.0
assert x[1].grad == 2.0

# Verify that the output layer does not apply ReLU.
for p in params:
    p.data = 0.0
model.layers[-1].neurons[0].b.data = -2.0

assert model([Value(1.0), Value(2.0)])[0].data == -2.0
