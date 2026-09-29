# From isolated kernels to model performance

## What our isolated result covers

- We measured **dX for one convolution workload**, not its full backward pass.
- **dW was excluded.** Full training also includes forward work, other gradients,
  optimizer updates, and CPU work.
- A local time reduction does not transfer directly to the complete step.
- Enabling **cuDNN search model-wide** can change multiple convolutions; do not
  attribute the full improvement to the one isolated kernel.

## Amdahl's law: a simple estimate

- If a component occupies fraction **f** of step time and becomes **s times
  faster**, then `new_time / old_time = (1 - f) + f / s`.
- Assumes the component contributes directly to elapsed time and other work
  stays unchanged; summed kernel durations are not automatically that fraction.

| Example: 4.5 ms step, 0.45 ms component | New step | Time saved |
| --- | --- | --- |
| Component becomes 2x faster | 4.275 ms | 5% |
| Component takes zero time (ideal limit) | 4.05 ms | 10% |

- **Time reduction** = `(old - new) / old`; **speedup** = `old / new`.

## Control the starting state

- Training changes **weights** and **BatchNorm running statistics**. Optimizers
  may also maintain state; our plain SGD has no momentum buffers.
- Running mode B after mode A on the same model gives B an already-trained model.
- Use **fresh processes**, the same seed and input-generation order, and equal
  warmup/update counts. Apply settings before any convolution.
- Fresh processes also separate in-process algorithm-selection caches.
- Keep **TF32 permission equal** across modes. Same initialization does not
  guarantee identical numerical training trajectories.

## Interpret the evidence

- Repeated blocks show within-process variation; fresh-process repeats also
  expose variation between starts. Vary mode order and preserve each run.
- Our isolated comparison showed **about 19% less helper time**. The first model
  pair showed **about 9.2% less step time**; model repeatability remains pending.
- These are workload-specific estimates, not exact or universal improvements.
- Finite or similar final losses do **not** establish equivalent training quality.
  Our original elementwise numerical criterion remains unresolved.
- [First model comparison evidence](../../progress/sessions/2026-09-29-project-02-model-cudnn-comparison/summary.json)

## Next experiment

- Repeat the model comparison in fresh processes with varied mode order, then
  inspect phase/kernel changes before assigning a cause to the total saving.


## Repeated model comparison

- Three fresh processes per mode gave **3.838 ms off** versus **3.529 ms on**
  using median-of-process-medians: **about 8% lower step time**.
- All on-mode medians were lower than all off-mode medians. The initial pair
  showed about 9%; repeat results reinforce the direction, not an exact percentage.
- Search-on trace kernel counts varied across processes. Count differences
  alone do not establish which algorithms changed or explain the timing gain.
- Finite final losses are only a sanity check, not training-accuracy validation.
- [Repeated-run evidence](../../progress/sessions/2026-09-29-project-02-model-cudnn-repeat-runs/summary.json).
