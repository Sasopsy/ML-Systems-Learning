# Isolating a convolution input gradient

## Learning target

- Reproduce the **dX computation** from a model trace using a small standalone
  workload, then verify its GPU kernel before measuring or optimizing it.
- Matching an operation does not guarantee matching the original kernel choice.

## Inputs to the isolated computation

| Item | Value |
| --- | --- |
| Desired input-gradient shape | `(32, 512, 2, 2)` |
| Weight W | `(512, 512, 3, 3)`, float32 CUDA |
| Incoming gradient dY | `(32, 512, 2, 2)`, float32 CUDA |
| Stride / padding / dilation / groups | `1 / 1 / 1 / 1` |

- Use seed 42 and contiguous random tensors: **synthetic values**, matching
  captured shapes/settings, not the original tensors.
- `torch.nn.grad.conv2d_input(input_size, weight, grad_output, ...)` computes dX.
- For ordinary linear convolution, dX depends on **W and dY plus shape/settings**;
  actual X values are not needed. X is used in the separate reference check.
- Installed PyTorch 2.11 helper requests only input gradient with output mask
  `(True, False, False)`. Original model requested input and weight gradients.
- The helper constructs a placeholder input with expanded strides; kernel choice
  must be verified, not assumed identical to the model capture.

## First correctness check

1. Implement one input-gradient call in `conv_dgrad.py`.
2. Create X of the matching shape with `requires_grad=True` for a reference.
3. Compute Y with `torch.nn.functional.conv2d`, using the same W/settings.
4. Obtain reference dX with `torch.autograd.grad`, passing dY as grad_outputs.
5. Check shape, dtype/device, finite values, and `torch.testing.assert_close`.

- Autograd is an API consistency check, not an independent cuDNN implementation;
  both paths can use the same backend.
- Keep correctness/reference work separate from later timings.
- Learner implementation passed on CUDA: dX shape **(32,512,2,2)**, float32,
  finite results, default assert_close passes. Maximum absolute error against
  autograd reference: **7.6293945e-06**. This is numerical agreement, not bitwise
  equality or independent backend validation.
- Kernel matching and timing follow below; optimization has not been attempted.

## Next: verify the isolated kernel

1. Keep tensors/reference/checks outside the capture.
2. Warm up **10** calls to the input-gradient helper with the same tensors.
3. Synchronize, then profile **3** calls with CPU and CUDA activities.
4. Put each helper call inside a **dgrad_only** record_function scope.
5. Synchronize after all three calls inside the profiler, outside the labels.
6. Export **conv_dgrad.trace.json** beside the script.

- Record existing cuDNN benchmark, deterministic, and allow_tf32 settings;
  compare with the model configuration before interpreting kernel differences.
- Verify exact kernel name and launch grid/block against the model candidate.
- Additional helper kernels may appear; do not assume one Python call is one
  GPU kernel. Keep reference forward/autograd operations outside this capture.

## Isolated kernel match verified

- Corrected capture runs; correctness still passes with max error **7.6293945e-06**.
- Three calls each launch **scalePackedTensor_kernel + dgrad_engine**; no wgrad.
- Exact dgrad name matches model; **grid [72,1,32]**, **block [8,8,1]**,
  **96 registers/thread**, **3328 bytes shared memory/block** also match.
- Profiled dgrad durations: **143.711, 142.880, 142.176 us**; model example was
  **142.302 us** in the shape capture. These are diagnostic observations, not an
  unprofiled baseline, repeatability study, or speedup claim.
- cuDNN flags: benchmark **False**, deterministic **False**, allow_tf32 **True**.
- [Verified capture evidence](../../progress/sessions/2026-09-27-project-02-isolated-dgrad-capture.json).

## Completed helper-call baseline

- Use **5 blocks of 100 calls** outside the profiler, after warmup/correctness.
- Synchronize before start and before end of perf_counter interval per block;
  do not synchronize each iteration.
- Report **elapsed seconds x 1e6 / 100** as microseconds/call per block.
- Reuse W/dY; keep reference, assertions, printing, and profiler outside timing.
- This boundary measures completed helper calls including Python/launch work,
  allocations and both GPU kernels, not dgrad kernel execution alone.
- Corrected implementation ran successfully: **147.50, 144.53, 144.49, 144.49,
  144.49 us/call** across five blocks. Values are rounded to two decimal places;
  matching printed values do not imply identical raw measurements.
- Separate capture: **142.068 us** mean dgrad and **1.354 us** mean scale helper
  (three of each). These are different measurement boundaries, not a speedup.
- **Boundary pitfall:** synchronizing/timing each inner-loop call measures
  serialized per-call latency with repeated completion waits. Timing the entire
  100-call block allows host submission to overlap earlier GPU work and gives
  an amortized completed-call cost. These answer different questions.
- Use **time.perf_counter()** for elapsed intervals; store one duration per block
  and print after all five blocks.
- Avoid `for i, time in ...` inside a function using the imported `time` module:
  Python treats `time` as local throughout that function, breaking earlier
  `time.perf_counter()` calls. Use a name such as **elapsed**.
- Stored block duration must be divided by **100** to report a per-call average.

## Synchronization and CPU/GPU overlap

- **Block-boundary synchronization:** CPU can submit later calls while the GPU
  executes earlier ones. Block time / call count measures amortized completed-call
  cost; CPU submission and GPU execution times need not simply add together.
- **Synchronize after every call:** CPU waits for completion before submitting
  the next call. This removes overlap between calls and adds repeated waits;
  some overlap within a call can remain.
- **Prediction:** per-call synchronization will increase average time here.
  We discussed the mechanism but skipped the comparison experiment.
- **Pitfall:** subtracting separately profiled kernel time from block-average
  wall time does not isolate CPU overhead. Caching was suggested, not established.

## References

- [PyTorch 2.11 convolution gradient helper](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/grad.py)
- [torch.autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html)


## Algorithm search and numerical precision

- **cuDNN benchmarking** tries candidate implementations and reuses its selection
  for matching calls; search cost belongs outside steady-state timing.
- **Float32 storage does not guarantee full-precision multiplication:** allowing
  TF32 permits reduced-precision internal multiplication for eligible operations.
- Different selected implementations can disagree beyond a strict tolerance.
  Preserve the assertion and investigate precision before claiming a speedup.
- Our search-enabled diagnostic failed with TF32 permitted and passed with it
  disabled. This supports a precision-related explanation, not a proven kernel cause.
- Compare search on/off with **the same precision settings**; changing both
  search and TF32 requires a new matched baseline.
- A worst-element relative error is not an overall gradient error metric.


## Checking accuracy across algorithm choices

- Save final timed outputs **after timing**, alongside inputs. Verify W/dY are
  exactly equal across modes before comparing gradients.
- Agreement with a backend-sharing autograd reference can hide numerical error:
  both modes passed their own reference, but failed direct cross-mode agreement.
- Build a **higher-precision CPU reference** by promoting the original saved
  float32 inputs to float64; do not regenerate random inputs.
- Preserve explicit acceptance tolerances after promotion:
  `abs(result - reference) <= 1e-5 + 1.3e-6 * abs(reference)`.
- Both modes failed this criterion against our float64 reference. The baseline
  had lower maximum and mean error; neither was established as ground truth.
- A float64 result is not exact arithmetic. Default tolerances are not a
  universal application accuracy requirement; do not loosen them just to pass.


## Tolerance defaults

- **Pass rule:** `abs(error) <= atol + rtol * abs(reference)`.
- Float32 `assert_close` defaults: **atol=1e-5**, **rtol=1.3e-6**; inherited
  from PyTorch, not derived for our convolution. Near zero, allowance approaches
  **0.00001**. Different addition orders can change rounded float32 results.
- Failing this default does **not** establish unacceptable training accuracy;
  application tolerances need justification, not adjustment just to pass.
- [PyTorch tolerance defaults](https://docs.pytorch.org/docs/2.11/testing.html#torch.testing.assert_close).


## Overall versus elementwise error

- **Relative L2 error:** `norm(result - reference) / norm(reference)` measures
  aggregate error relative to the reference, assuming its norm is nonzero.
- A small overall ratio can coexist with many elementwise tolerance failures;
  neither metric replaces the other or establishes downstream training quality.


## Failure counts versus rates

- **Rate = failures / elements in the bin.** More failures can simply reflect
  more elements: search-on had 63/120 failures below 0.1 (52.5%), versus
  5075/54137 at 10+ (9.37%).
- Small reference values failed more often proportionally; most search-on
  failures were nevertheless in the 10+ bin. The magnitude-dependent tolerance
  influences these rates; they do not measure accuracy alone.
- Isolated result: about 19% lower helper time, small relative L2 error, but both
  modes fail the original elementwise criterion. Training impact remains unknown.


## Comparing against a float64 reference

- Convert the existing float32 inputs to float64; do not regenerate different values.
- Keep the reference in float64 and promote the tested output before subtracting. Casting the reference back to float32 adds rounding to the comparison.
- **Maximum absolute error** uses `.abs().max()`; **mean absolute error** uses `.abs().mean()`.
- Maximum error alone cannot determine an elementwise tolerance result: the allowance also depends on the corresponding reference magnitude.
- Explicitly retain the chosen float32 error criterion when doing comparison arithmetic in float64.
