# Exporting and reading a timeline

## Summary versus timeline

| View | Question it helps answer |
| --- | --- |
| Operator table | Which operation families accumulate device time? |
| Timeline | When do operations, launches, and GPU executions happen? |

- A summary hides event ordering, overlap, and intervals between executions.
- A timeline places CPU and GPU activity on a shared time axis.

## Export and open

1. Finish the profiler context before exporting.
2. Call **`prof.export_chrome_trace(path)`** to save Chrome trace JSON.
3. Keep the output beside the script as **`resnet18.trace.json`** for this exercise.
4. Open [Perfetto](https://ui.perfetto.dev/) and select **Open trace file**.

- Use a script-relative path so output location does not depend on shell directory.
- Existing Git rules ignore `*.trace.json`; the trace will not travel with a normal
  code push. Save small observations in progress notes for handoff.
- Exported trace now exists locally and passed JSON/event validation. Historical
  captures were not exported; this file comes from a new bounded run.

## First things to inspect

- **Tracks:** distinguish CPU operations/runtime calls from GPU stream activity.
- **Event width:** duration on the time axis; select an event for its details.
- **Gaps:** intervals with no event on that track.
- **Relationships:** use available correlation/flow information to connect launches
  to device work; timestamp alignment alone is not proof of causality.

## Pitfalls

- A gap on one GPU stream does **not establish device-wide idleness**; check other
  relevant streams and copy activity.
- A gap alone does **not prove CPU launch overhead** caused it.
- The trace is instrumented execution; retain the separate wall-clock benchmark.
- Current trace contains CPU operations, CUDA runtime calls, GPU kernels, and
  GPU memset events. Captured kernels are on **GPU 0 / stream 7**.
- File validation is complete; learner has opened the timeline and observed gaps.
  Track interpretation is underway; gap causes remain unverified.

## Track labels in our trace

| Label | Meaning |
| --- | --- |
| `python 99112` | Python process; 99112 is its process ID |
| `thread 99112 (python)` | Main CPU thread |
| `thread 99148 (pt_autograd_0)` | CPU worker used by the autograd engine |
| `python 0` / `stream 7` | Synthetic GPU grouping and CUDA stream 7 on device 0 |
| `Activity Buffer Request` | Profiler overhead event, not a model kernel |

- CPU thread IDs are **identifiers**, not thread counts or CPU core numbers.
- A CUDA stream is an **execution queue**, not one CUDA thread or seven threads.
- Chrome trace uses process/thread fields to arrange GPU tracks too; generic
  viewer labels do not necessarily describe actual OS processes/threads.

## Why a CPU selection can cover GPU gaps

- `aten::conv2d` is a **CPU-side operation scope**, from host entry to return.
- Nested rows show calls inside it: convolution → _convolution → cudnn_convolution.
- Their nesting does not mean four separate convolutions or four CPU threads.
- CPU scopes can include setup, nested calls, waits, and instrumentation overhead.
- Associated GPU kernels have their own start/end times on GPU stream tracks.
- A highlighted time interval can include blank areas on other tracks; a highlight
  is **not evidence that a GPU kernel executed throughout that interval**.
- Hover behavior alone is ambiguous: click the event and inspect **Current Selection**.

## Concrete example from our trace

- First CPU conv2d scope: **2127.788 us ≈ 2.128 ms**.
- Four associated GPU kernels: **31.839 us total**.
- First associated kernel starts about **2054.113 us after CPU scope entry**.
- The long CPU interval is not 2.128 ms of GPU execution, nor proof of continuous
  CPU computation. Do not infer the cause of its delay from its width alone.
- Click an event, press **F** to fit it, and compare CPU scope and kernel durations.

## References

- [PyTorch 2.11 trace export](https://docs.pytorch.org/docs/2.11/profiler.html#torch.profiler._KinetoProfile.export_chrome_trace)
- [Perfetto UI guide](https://perfetto.dev/docs/visualization/perfetto-ui)
