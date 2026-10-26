# CUDA Graph replay timing

## Core idea

- **Capture:** record a sequence of GPU operations and their dependencies once.
- **Replay:** submit that recorded sequence without issuing every operation individually from Python.
- Direct launches can leave GPU idle gaps while the CPU submits work. Replay reduces per-operation host submission overhead.
- **No automatic kernel fusion:** capturing 500 softmax calls does not turn them into one softmax kernel.

| Direct batch | Graph batch |
| --- | --- |
| Python repeatedly submits each operation | Python submits a replay of the recorded sequence |
| Event interval can include submission gaps | Event interval includes graph execution and its scheduling overhead |
| Measures the direct launch path | Measures the captured replay path |

## Measurement procedure

1. Allocate inputs/output buffers, check correctness and warm up before capture.
2. Capture **500 calls** into a graph, using the same persistent buffers.
3. Replay once and synchronize outside the timed region to warm graph execution.
4. Record a start event, replay the graph, record a stop event and synchronize the stop event.
5. Convert milliseconds to microseconds and divide by 500.
6. Repeat for multiple rounds; report the distribution of **batch averages**.

## Minimal timing example

Assumes CUDA tensors `x` and `out` already exist, correctness was checked, and warmup completed. Keep these buffers alive for replay.

```python
repeats = 500
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for _ in range(repeats):
        torch.softmax(x, dim=-1, out=out)

graph.replay()
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
# Initialize the events outside the measured rounds.
start.record()
stop.record()
stop.synchronize()

samples_us = []
for _ in range(9):
    start.record()
    graph.replay()
    stop.record()
    stop.synchronize()
    samples_us.append(start.elapsed_time(stop) * 1000 / repeats)
```

## Interpretation and pitfalls

- **Excluded:** allocation, capture, warmup, transfers and correctness checks.
- **Included:** execution of all captured operations, graph scheduling and any gaps inside the event interval.
- GPU work remains asynchronous: wait for the stop event before reading elapsed time.
- Kernel arithmetic is unchanged by capture, but launch scheduling differs. Compare all implementations under the same timing mode.
- Our softmax experiment repeatedly uses identical inputs and overwrites the same output. This favors warm-buffer/cache behavior; it does not represent every application workload.
- Dividing a replay duration by 500 gives **average time per operation in that batch**, not individual-call latency or per-call p99.
- Keep direct and graph results separate. A faster replay does not guarantee the same speedup in eager execution or a complete model.
- Custom CUDA launches must use the capture stream; see the actual [Project 3 sweep harness](../../../project_03_gpu_operator_library/benchmarks/sweep_softmax.py) for stream handling and matched validation.

- [Back to timing notes](README.md)
