# Scalar autograd

## Core idea

- Each **`Value`** holds a scalar, its gradient, and links to its inputs.
- **Backward** propagates loss gradients using the chain rule.
- A value can affect the loss through several paths: **add every contribution**.

## Local gradient rules

For output `z`, let the incoming gradient be `g = dL/dz`.

| Operation | Contributions to input gradients |
| --- | --- |
| `z = a + b` | `a.grad += g`, `b.grad += g` |
| `z = a * b` | `a.grad += b.data * g`, `b.grad += a.data * g` |
| `z = relu(a)` | `a.grad += g` if `a > 0`, otherwise zero |

## Example: why accumulation matters

- Expression: `y = x * x`.
- Both input roles contribute `x`, giving `dy/dx = 2x`.
- At `x = 3`, the gradient is **6**.
- Using `=` instead of `+=` would overwrite a contribution.

## Backward order

1. Seed the scalar output gradient with **1**.
2. Traverse nodes in **reverse topological order**.
3. Run each node's backward rule after all downstream contributions have arrived.

- Visit each node once, preserving every input-role contribution.
- **Revisit:** topological sorting and graph traversal.

## Pitfalls

- Mutating forward values before backward can invalidate derivatives.
- Our implementation chooses a ReLU gradient of **zero at zero**.
- Repeated backward on the same graph is **not a supported, tested contract** here.
