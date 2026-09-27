# Kernel launch arguments

## Keep event identities together

- One kernel name can appear in many launches; **correlation ID identifies the
  launch relationship** in this trace.
- Selected example: **11.808 us**, correlation **5036**, External id **392**.
- Pasted args: correlation **132**, External id **24**; earlier invocation with
  duration **11.680 us**. Resource configuration is the same.
- Both execute on **device 0, stream 7**. Device number is not stream number.

## Kernel duration, launch delay, and preceding gap

For correlation 5036, take the preceding stream-7 GPU event's end as time zero:

| Event | Relative start (us) | Relative end (us) |
| --- | --- | --- |
| CPU cudaLaunchKernelExC | 11.933 | 18.826 |
| GPU kernel | 20.256 | 32.063 |

- **Execution duration:** 11.808 us; the GPU is executing this kernel then.
- **Flow delay:** 1.430 us from CPU launch-call end to GPU kernel start.
- **Preceding stream gap:** about 20.256 us from previous GPU event end to start.
- The full gap may include host work, submission/dispatch delays, or other causes.
- Check other device activity before inferring device-wide idleness; a launch-flow
  arrow alone does not prove the whole gap is CPU computation.

## Worked gap calculation from Perfetto timestamps

- Previous kernel: start **13.330792 ms**, duration **0.002560 ms**;
  end = **13.333352 ms**.
- CPU launch: start **13.345284 ms**, duration **0.006894 ms**;
  end = **13.352178 ms**.
- Next GPU kernel starts at **13.353608 ms**.
- Gap = **20.256 us** = **11.932 us before launch starts** + **6.894 us
  inside the CPU launch call** + **1.430 us after launch returns**.
- These values use the displayed timestamps; earlier raw-trace relative values
  can differ slightly through rounding.
- **Compare positions on the timeline**, not just CPU/GPU durations: a long
  launch call can overlap earlier GPU work in other cases.
- Here, the next kernel's launch had not started when the previous kernel ended.
  This supports a host-submission delay in this example, but does not establish
  what the CPU was doing, device-wide idleness, or a whole-model bottleneck.
- Next inspect the launching CPU thread between **13.333352 and 13.345284 ms**;
  any blank portion is unaccounted time, not automatically computation or idleness.

### Attribute-query follow-up

- Learner identified two **cudaFuncGetAttributes** CPU API calls in this interval.
- This API queries a compiled kernel's properties, such as register use, static
  shared-memory requirement, and maximum threads per block. It does not launch
  that kernel or measure its performance.
- Saved trace: correlation **5034**, **5.097 us**; correlation **5035**, **2.374 us**.
- Both fall entirely inside the interval before launch begins: **7.471 us total**
  of approximately **11.932 us**. About **4.46 us** remains outside these calls.
- These are instrumented CPU API durations; neither their internal time breakdown
  nor why two queries occur is established. Their names do not prove the same
  kernel was queried twice or that either call is redundant.
- **Queries need not immediately precede launches.** They return metadata to
  host code; a caller can query A and B, then launch A and B, or query the same
  function from two helper routines. These are possible patterns, not a diagnosis.
- Two queries followed by two launches do **not** establish one query per kernel.
  The trace's query args contain External id, cbid, and correlation, but no `func`
  argument identifying the queried kernel. Correlation identifies each API call;
  it does not pair an attribute query with a future launch.
- Supported: host-side attribute queries occupy part of this stream's gap before
  its next launch. Not established: complete gap cause, removable overhead, or a
  whole-model CPU bottleneck.
- [CUDA 13 runtime API: cudaFuncGetAttributes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__EXECUTION.html)

## Argument meanings

| Argument | Meaning |
| --- | --- |
| `External id` | Kineto link to the associated host operation; one operation may launch several kernels |
| `queued` | CUPTI command-buffer queue timestamp in ns; normally not collected by default. Zero is not a queue-length or zero-wait measurement |
| `device` | GPU device identifier |
| `context` | CUDA context identifier; the execution/resource environment |
| `stream` | CUDA stream identifier; a queue of work |
| `correlation` | Links this device activity to the runtime/driver launch activity |
| `registers per thread` | Reported register requirement per thread; registers are fast per-thread storage |
| `shared memory` | Static + dynamic shared-memory bytes per block; on-chip storage shared within the block |
| `grid` | Number of blocks along x, y, z |
| `block` | Threads per block along x, y, z |
| `blocks per SM` | Total launched blocks / device SM count; not measured resident blocks |
| `warps per SM` | Total launched warps / device SM count; not measured active warps |
| `est. achieved occupancy %` | Kineto model estimate, rounded to an integer; not a hardware-counter measurement |

## Decode our launch

- **SM:** streaming multiprocessor, a GPU execution unit that hosts thread blocks.
- **Warp:** a group of 32 CUDA threads.
- Grid `[8, 4, 1]` → **32 blocks**.
- Block `[384, 1, 1]` → **384 threads = 12 warps per block**.
- Whole grid → **12,288 threads / 384 warps**.
- H100 in this trace → **132 SMs**.
- `32 / 132 = 0.242424` blocks/SM; `384 / 132 = 2.909091` warps/SM.
- Fractional ratios do not mean a fraction of a block is scheduled onto an SM.
- 32 blocks cannot occupy all 132 SMs simultaneously for this kernel alone.
- Shared memory: **231424 bytes = 226 KiB per block**.
- Registers: **168/thread**; nominal product is **64512/block**, before considering
  hardware allocation rules. High resource requirements can limit resident blocks.

## Occupancy caution

- Occupancy concerns **resident warps relative to the SM's maximum**, not the
  percentage of time a kernel bar is visible or the percentage of peak FLOP/s.
- Kineto estimates from launch/resource/device metadata; it does not measure
  achieved occupancy with hardware counters in this capture.
- **Displayed 0% does not mean zero GPU activity.** This kernel demonstrably ran.
- The estimator uses assumed shared-memory/function settings; exact reason for
  zero here is not established. Use a suitable kernel profiler to investigate.
- Resource-heavy and small-grid observations suggest questions to test; they do
  not by themselves prove a compute/memory bottleneck or an available speedup.

## Sources

- [CUPTI kernel activity fields](https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel9.html)
- [Kineto launch metrics at PyTorch 2.11's pinned revision](https://github.com/pytorch/kineto/blob/7a731b6ae01cfc2b1fc75d83a91f84e682e43fd7/libkineto/src/DeviceProperties.cpp)
- [Kineto metadata writer at the same revision](https://github.com/pytorch/kineto/blob/7a731b6ae01cfc2b1fc75d83a91f84e682e43fd7/libkineto/src/CuptiActivity.cpp)
- [Nsight Compute occupancy and launch statistics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
