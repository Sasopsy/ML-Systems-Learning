# GPU timing and synchronization

## Core idea

- CUDA work is typically **asynchronous with respect to the CPU**.
- The CPU submits work and can continue before the GPU finishes.
- Returning from `train_step()` does **not guarantee GPU completion**.
- `time.perf_counter()` measures **elapsed wall-clock time**, including waiting.
- Without a completion wait, the timer can stop while GPU work is still running.

## Timing sketch

```python
import time
import torch

# Current CUDA device; this is not a complete benchmark.
torch.cuda.synchronize()  # Finish earlier GPU work.
start = time.perf_counter()

train_step(model, inputs, targets)

torch.cuda.synchronize()  # Finish this step's GPU work.
elapsed = time.perf_counter() - start
```

| Wait | Purpose |
| --- | --- |
| Before the timer | Prevent earlier unfinished work entering the measurement |
| Before the ending timestamp | Ensure the measured GPU work has finished |

- **Synchronization means waiting**, not copying tensors between CPU and GPU.

## Example: earlier work

- At the starting timestamp, earlier GPU work still needs **20 ms**.
- Our step takes **5 ms** and runs afterward, without overlap.
- Waiting at the end can produce a measurement of roughly **25 ms**.
- If earlier work already finished before timing, it contributes **nothing**.
- These numbers are illustrative, not measured results.

## Pitfalls

- This measures **completed-step wall-clock latency**, not isolated kernel time.
- CPU overhead includes Python execution, dispatch, and framework bookkeeping.
- A complete benchmark also needs **warmup, repetitions, and explicit boundaries**.
- CUDA events and kernel profiling are future lessons.
