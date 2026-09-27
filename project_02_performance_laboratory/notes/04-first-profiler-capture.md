# First profiler capture

## The three levels

| Level | What we inspect |
| --- | --- |
| Training step | Completed wall-clock latency, including host overhead and waiting |
| PyTorch operation | Framework work such as convolution, normalization, or addition |
| GPU kernel | A device execution launched to carry out part of the work |

- An operation is **not necessarily one kernel**.
- Use an **operator summary** to find expensive operation families.
- Use a **timeline** to inspect order, launches, and device execution events.
- The first summary does not yet diagnose a kernel's memory or compute limits.

## First capture plan

1. Finish the existing benchmark and its reporting.
2. Wait for earlier GPU work to finish.
3. Enter a **`torch.profiler.profile`** context with CPU and CUDA activities.
4. Run **three additional training steps**; finish GPU work before leaving.
5. Print the operator summary **after** the context exits.

- Reuse the current model, optimizer, and synthetic batch; training state continues.
- Keep diagnostic capture **separate from the benchmark timing**.
- Start without shape, stack, or memory tracing to keep the first capture simple.
- Profiler instrumentation can perturb execution; do not use capture latency as
  the final benchmark result. Shape/stack tracing adds further overhead.

## APIs to recognize

| API | Purpose |
| --- | --- |
| `torch.profiler.ProfilerActivity.CPU` | Record host-side activities |
| `torch.profiler.ProfilerActivity.CUDA` | Request GPU activity recording |
| `prof.key_averages()` | Aggregate events by operator name by default |
| `.table(sort_by="self_cuda_time_total", row_limit=15)` | Show top rows by self CUDA time |

- **Self** attribution excludes child-operation contributions; it is not the
  end-to-end duration of the model step.
- Aggregation combines repeated calls across the captured steps.
- First local capture successfully returned both operation and GPU kernel events.
- **Reference:** [PyTorch 2.11 profiler docs](https://docs.pytorch.org/docs/2.11/profiler.html).

## Reading the first table

- **`aten::...` rows:** framework operations with device work attributed to them.
- **Kernel-name rows:** device execution events, also aggregated by name.
- Operator and kernel rows can describe **the same GPU work**; do not add them
  together as independent costs.
- **Self CUDA time:** accumulated device time attributed directly to that row,
  excluding child-operation contributions for an operator row.
- **Calls:** number of grouped invocations across the whole capture.
- A reported total over three steps is **not one step's wall-clock latency**.
- A large attributed time identifies work to investigate; it does **not prove**
  a compute or memory-bandwidth bottleneck.
- With inputs already on GPU, distinguish **device memory traffic** from excluded
  CPU-to-GPU input transfers.

## Per call versus per training step

Example: convolution backward reports **4.163 ms across 60 calls in 3 steps**.

| Question | Calculation | Approximate result |
| --- | --- | --- |
| Average attributed GPU time per operator call? | `4.163 / 60` | 0.0694 ms = 69.4 us |
| Attributed GPU time per training step for this row? | `4.163 / 3` | 1.388 ms |

- Choose the denominator to match the **unit of the question**.
- The 60 calls are **operator invocations**, not necessarily 60 GPU kernels.
- This row covers only convolution backward; the full step includes other work
  and host-side overhead/waiting. It is not full-step wall-clock latency.
- Results above are calculations from rounded profiler output, not new timings.

## Why CUDA time differs from wall-clock time

| Measurement | What it counts | Our recorded result |
| --- | --- | --- |
| Profiler self CUDA footer | Sum of captured device-event durations | 6.910 ms over 3 steps ≈ 2.303 ms/step |
| Synchronized wall-clock benchmark | Elapsed time until all measured steps finish | Roughly 4.5–4.6 ms/step |

- **GPU time does not include gaps** between device events.
- A GPU may wait while the CPU runs Python, dispatches operations, or prepares
  more work. This is a possible source of gaps, not a diagnosis from the table.
- Wall-clock timing includes such intervals as well as GPU execution.
- CPU and GPU can work **at the same time**: do not simply add their totals.
- Concurrent GPU events can also overlap; summed durations are not necessarily
  the elapsed span of their execution.
- The benchmark and profiler captured **different steps**, and profiling can
  perturb execution. Subtracting their averages does not isolate CPU overhead.
- **Next evidence:** a timeline showing launches, kernels, and gaps.
