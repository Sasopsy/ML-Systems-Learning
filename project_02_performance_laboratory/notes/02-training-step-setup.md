# Training-step setup pitfalls

## Setup order

1. **Seed the RNG** before creating random weights and inputs.
2. **Create the model** and move it to its final device.
3. **Create the optimizer** using that model's parameters.
4. **Prepare inputs and targets**, then execute the training loop.

- `seed = 42` only stores a number; **`torch.manual_seed(seed)`** seeds the RNG.
- Seeding alone does **not guarantee deterministic GPU algorithms**.
- Device placement before optimizer creation keeps parameter references correct.
- ResNet-18 uses **`num_classes`**, not `classes`.

## Training-step order

1. Clear gradients.
2. Compute model predictions.
3. Compute loss.
4. Backpropagate.
5. Update parameters.

## Keep these values distinct

| Value | Meaning | Common mistake |
| --- | --- | --- |
| `loss_fn` / criterion | Callable that computes the loss | Overwriting it with a result breaks later iterations |
| Loss tensor | Result of applying the criterion | Confusing it with the loss function |
| `loss.item()` | Python scalar extracted from the tensor | Can require waiting for a pending CUDA value |

## Checks and pitfalls

- Keep `.item()` outside timing when scalar extraction is excluded from the boundary.
- Defining a function does not execute it: **call the loop**.
- Try **two untimed steps** to expose state/naming errors hidden by one step.
- Keep the chosen optimizer settings fixed when comparing measurements.
