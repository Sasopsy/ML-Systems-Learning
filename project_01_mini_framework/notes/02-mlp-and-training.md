# MLPs and training

## Model structure

| Component | Role |
| --- | --- |
| Neuron | Computes `sum(w_i * x_i) + b`, optionally followed by ReLU |
| Layer | Applies several neurons to the same inputs |
| MLP | Chains layers; ours uses ReLU hidden layers and a linear output |

- **Parameter identity:** `parameters()` must return the actual shared `Value` objects.
- Updating copies would not change weights used by the next forward pass.

## One training iteration

1. **Clear** parameter gradients from the previous iteration.
2. **Build** a fresh forward graph and compute the loss.
3. **Backpropagate** to accumulate loss gradients.
4. **Update** with SGD: `parameter -= learning_rate * parameter.grad`.

## Regression example

- Prediction: `w*x + b`.
- Loss: `MSE = sum((prediction - target)**2) / N`.
- Target rule: `y = 2x + 1`.
- Expected fitted parameters: approximately **`w = 2`, `b = 1`**.

## Pitfalls

- Accidental gradient accumulation changes the update.
- Loss is a **scalar objective**, not a parameter.
- Missing a tight tolerance after a few iterations does not prove backward is wrong.
- Check **gradients and convergence trajectory** before diagnosing the failure.
- Historical result: 20 iterations were insufficient; 100 passed on the earlier
  Mac runtime. This is a recorded result, not a new test.
