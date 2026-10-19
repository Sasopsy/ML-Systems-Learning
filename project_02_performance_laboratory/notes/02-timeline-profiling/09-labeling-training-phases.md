# Labeling training phases

## Purpose

- **record_function** adds a named CPU scope to the profiler trace.
- Use phase names to connect whole-step structure to operators and GPU kernels.

```python
from torch.profiler import record_function

with record_function("forward"):
    outputs = model(input)
```

## Current exercise

1. Keep the existing timing path; create a diagnostic `profiled_train_step` with
   the same operations and arguments as `train_step`.
2. Add five sibling scopes: **zero_grad**, **forward**, **loss**, **backward**,
   **optimizer_step**. Return the loss tensor as before.
3. Call this function only in the profiler loop, with an outer **train_step**
   scope around each invocation.
4. Export to `resnet18.phases.trace.json` to retain the original capture.

## Interpretation pitfalls

- **Names do not create nesting.** A label like `train_step: forward_pass` is
  just text. An outer `with record_function("train_step"):` around the function
  call creates the parent scope containing the phase scopes.
- These labels are **CPU intervals**, not synchronized GPU stage timers.
- Adding a label does not wait for GPU completion.
- Do not insert synchronization between phases for this exercise; keep the
  existing capture-boundary synchronization outside the step labels.
- Keep operation ordering and training behavior equivalent in the diagnostic
  function; duplicated logic must stay consistent if the workload later changes.
- This is an annotation exercise, not an optimization or speedup measurement.
- Implementation and capture verified; independent interpretation is pending.

## Verified capture

- File: `../resnet18_phases.trace.json` (local, Git-ignored).
- **Three** CPU `train_step` scopes, each containing five ordered sibling phases:
  zero_grad, forward_pass, loss, loss_backward, optimizer_step.
- Original capture retained unchanged; annotation implementation was assisted.
- This capture also contains **gpu_user_annotation** events. These are annotation
  ranges and must be distinguished from events with category **kernel**.
- Summary output includes a forward_pass row with Self CUDA **14.660 ms / 212.55%**
  while the footer reports **6.897 ms**. Do not interpret that percentage as its
  share of actual GPU kernel execution or compare phase totals blindly.
- Start by inspecting CPU scope nesting in the second captured step; GPU work
  attribution needs separate checking, especially for autograd worker threads.

## First capture step versus later steps

| CPU scope duration (ms) | Step 1 | Step 2 | Step 3 |
| --- | --- | --- | --- |
| forward_pass | 13.855 | 2.868 | 2.745 |
| loss_backward | 6.149 | 3.628 | 3.733 |

- Two **Activity Buffer Request** overhead events appear only during the first
  step: **1.763 ms** within forward and **1.748 ms** within backward.
- Their presence supports investigating profiler startup overhead. Their own
  durations do not account for the full difference: forward drops about 10.987 ms
  from step 1 to step 2, much more than the 1.763 ms buffer event.
- Overlap is observed; exact causal slowdown and remaining time are unestablished.
  Do not subtract overhead durations and label the result unprofiled performance.
- **Workload warmup and profiler warmup are different.** Earlier training steps
  prepare the workload, while first use of profiling can introduce extra costs.
- Later steps are useful for this trace comparison; three steps do not establish
  steady-state stability. All values above are CPU intervals, not GPU kernel sums.
- [PyTorch profiler recipe: first-use overhead and profiler warmup](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

## Backward work can launch from another CPU thread

- Verified learner-selected second-step kernel: **dgrad_engine**, **142.015 us**.
- Launch: **cudaLaunchKernel**, correlation **7174**, CPU thread **55549**
  (`pt_autograd_0`); associated operation **aten::convolution_backward**, External
  id **1208**. Launch CPU duration is **5.233 us**.
- The `loss_backward` label is on main thread **55492**, while this backward
  operation and launch are on autograd worker **55549**.
- Phase attribution must account for worker-thread execution; looking only for
  nested launch rectangles on the main thread can miss backward GPU work.
- For `Y = conv(X, W)`, **dgrad** computes the loss gradient with respect to the
  layer input `X`; **wgrad** computes it with respect to weights `W`. Input can
  be an intermediate activation, enabling propagation to earlier layers.
- [cuDNN convolution dgrad and wgrad](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Convolutions.html)

## One backward operation launches several kernels

- Learner found matching **wgrad_alg1_engine**, duration **37.247 us**,
  correlation **7194**, External id **1208**; CPU launch on worker **55549**.
- Saved trace contains three GPU kernels for operation 1208:

| Kernel | Duration (us) | Correlation |
| --- | --- | --- |
| scalePackedTensor_kernel | 1.344 | 7170 |
| dgrad_engine | 142.015 | 7174 |
| wgrad_alg1_engine | 37.247 | 7194 |

- **180.606 us** is the sum of these kernel durations. It excludes inter-kernel
  gaps and is not the CPU operation's duration or whole backward phase duration.
- Same External id links to one CPU operator; distinct correlations link each
  kernel to its own CUDA launch. Layer/phase/operator/kernel are different levels.
- Before choosing an isolated-kernel optimization, assess aggregate contribution
  across the workload; one long invocation alone does not establish priority.

## API reference

- [PyTorch 2.11 record_function](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.profiler.record_function.html)
