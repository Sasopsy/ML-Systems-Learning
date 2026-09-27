# Timeline reading checklist

## Identify the event before interpreting its width

| Category | What its duration represents |
| --- | --- |
| `cpu_op` | Elapsed host operation scope, including nested work and possible waiting |
| `cuda_runtime` | Time inside a CUDA runtime API call; a launch call is not the kernel |
| `kernel` | A recorded GPU kernel execution |
| `gpu_memset` | GPU memory initialization activity |
| `overhead` | Profiler bookkeeping, such as Activity Buffer Request |
| `gpu_user_annotation` | A labeled region; not necessarily one kernel or continuous execution |

## Six interpretation rules

1. **Use category and track, not color.** Similar-looking bars may represent
   entirely different kinds of activity.
2. **Do not double-count nesting.** A 2 ms parent containing a 1.8 ms child spans
   2 ms, not 3.8 ms. CPU and GPU overlap also prevents simply adding their times.
3. **Launch is not execution.** A CPU launch can return before the GPU kernel
   starts or finishes. Use correlation IDs/flow links, not vertical alignment alone.
4. **Blank means no recorded event on that track.** A CPU gap may contain
   uninstrumented work or waiting. A GPU stream gap does not prove the whole GPU
   is idle or identify why that stream has no execution.
5. **Profiling affects execution.** Our trace includes profiler buffer-request
   events. Compare recurring patterns across steps; do not treat the first long
   CPU scope as representative without checking later ones.
6. **Activity is not efficiency.** A visible kernel may use only part of the GPU
   or wait on memory internally. Timeline width alone cannot establish occupancy,
   bandwidth saturation, or a compute-versus-memory limit.

## A small inspection procedure

1. Pick a kernel in the **second profiled repetition**.
2. Record its **name, category, stream, and duration**.
3. Note the gap after the preceding GPU event on that stream.
4. Inspect the associated CPU launch using available correlation information.
5. Write an **observation**, then a separate **hypothesis** about its cause.

- Click a slice for details; **F** fits it to the viewport, **Q** toggles the
  details drawer. [Perfetto controls](https://perfetto.dev/docs/visualization/perfetto-ui)
- Shape/stack recording adds profiling overhead; keep performance measurements
  separate. [PyTorch profiler](https://docs.pytorch.org/docs/2.11/profiler.html)
