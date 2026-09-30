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


## Locate savings by phase

- Attribute actual GPU kernels via **correlation -> CPU launch -> phase**;
  include autograd worker launches. Do not sum nested CPU operations or GPU
  annotation rows as additional kernels.
- **Absolute saving:** off time minus on time. **Relative reduction:** saving
  divided by off time. A phase can rank differently on these two measures.
- Our representative second-step captures showed forward saving **360 us
  (21.5%)**, backward **120 us (6.7%)** in kernel-duration sums; all six
  captures supported the same ordering. The isolated dX result did not predict
  which model phase would save most.
- Profiled kernel-duration sums do not directly equal unprofiled wall-time
  savings. CPU work, overlap and profiler perturbation affect that comparison.
- [Phase attribution evidence](../../progress/sessions/2026-10-01-project-02-phase-kernel-comparison.json).


## Match calls before explaining savings

- Match convolution **execution order, shapes, strides and parameters** across
  modes; repeated kernel names alone do not identify equivalent workloads.
- Four 64-channel 3x3 convolutions saved **164 us**, and three 512-channel
  3x3 convolutions saved **146 us** in our selected step: about **86%** of the
  forward kernel-time saving together. Both groups changed kernel selection.
- **64-channel group:** named tile changed from `256x64x8` to `128x32x8`.
- **512-channel group:** `implicit_gemm_indexed` changed to `implicit_gemm`,
  retaining the named `32x32x8` tile. Tile size alone does not describe a kernel.
- A different tile size or kernel name identifies an implementation change;
  it does not by itself establish why that implementation executes faster.
- Some calls can regress despite overall improvement; include them in totals.
  One 1x1 convolution increased from **6.784 to 10.496 us** in this pair.
- [Forward comparison evidence](../../progress/sessions/2026-10-01-project-02-forward-convolution-comparison.json).


## Tiles, blocks and resource use

- A tile is a chunk of computation, not a thread count. In a simple GEMM block
  tile, M/N describe the output rectangle and K the reduction chunk.
- Smaller output tiles can expose more independent blocks; larger tiles can
  reuse inputs across more outputs. Neither size is universally faster.
- Verified first 64-channel 3x3 forward call, second step:

| Field | Search off | Search on |
| --- | --- | --- |
| Named tile | 256x64x8 | 128x32x8 |
| Grid / total blocks | [1,32,1] / 32 | [2,64,1] / 128 |
| Threads per block | 128 | 128 |
| Registers per thread | 255 | 166 |
| Shared memory per block | 33,280 bytes | 16,896 bytes |
| Profiled duration | 73.440 us | 27.840 us |

- Our H100 has 132 SMs: 32 blocks can occupy at most 32 SMs at once; 128 blocks
  permit broader distribution. Actual simultaneous scheduling was not measured.
- More blocks and lower per-block resource demand support a parallelism/resource
  hypothesis. They do not isolate the cause or measure achieved occupancy.
- This is one invocation's duration, not a fourfold speedup or whole-step result.
