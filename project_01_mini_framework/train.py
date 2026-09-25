from value import Value

w = Value(0.0)
learning_rate = 0.1

data = [
    (-2.0, -3.0),
    (-1.0, -1.0),
    ( 0.0,  1.0),
    ( 1.0,  3.0),
    ( 2.0,  5.0),
]

w = Value(0.0)
b = Value(0.0)

steps = 100

for step in range(steps):
    # 1. Clear the parameter's gradient.
    w.grad = 0.0
    b.grad = 0.0

    # 2. Build a fresh forward graph.
    total_loss = Value(0.0)
    for x, y in data:
        prediction = w * Value(x) + b
        error = prediction - Value(y)
        total_loss += error * error

    mean_error = total_loss * Value(1.0 / len(data))
    mean_error.backward()

    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad

    print(step, mean_error.data, w.data, b.data)

assert abs(w.data - 2.0) < 1e-6
assert abs(b.data - 1.0) < 1e-6