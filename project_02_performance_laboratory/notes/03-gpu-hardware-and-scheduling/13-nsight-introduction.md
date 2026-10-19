# Nsight: timeline versus kernel investigation

- **Nsight** is NVIDIA's family of developer tools.

| Tool | Main question | Example |
| --- | --- | --- |
| Nsight Systems (`nsys`) | How does work fit together over time? | CPU launches, GPU gaps, copies and overlap |
| Nsight Compute (`ncu`) | What happens while a selected GPU kernel executes? | Compute activity, memory traffic and active warps |

## Connection to our experiment

- PyTorch Profiler collected the trace we viewed in **Perfetto**.
- We observed one forward convolution kernel change from **73.44 us to 27.84 us**.
- Its launch metadata changed from **32 to 128 blocks**, with lower registers per thread and shared memory per block.
- These observations suggest explanations; they do **not prove** why the kernel became faster.
- Nsight Compute supplies hardware measurements to investigate those explanations.
- A **hardware counter** records GPU activity, such as transferred bytes or executed instructions. Reports also contain derived metrics.

## Pitfalls

- A running kernel does not imply that all GPU resources are busy.
- More blocks do not automatically mean proportionally greater speed.
- Detailed profiling can perturb execution; keep ordinary timing separate.

## Sources

- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [Profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)


## Verify isolation before collecting hardware counters

- Match actual kernel names and launch resources against the original model trace; matching tensor shapes alone does not guarantee matching kernel selection.
- Our isolated forward captures reproduced both named tiles, grids, register counts and shared-memory sizes from ResNet.
- Count only `kernel` events for kernel duration; CPU and GPU annotations can share the same label.
- Keep PyTorch Profiler captures separate from unprofiled timing and Nsight Compute runs.


## Mark one call for Nsight Compute

- **NVTX range:** a named region around CPU code; Nsight can select GPU kernels launched while that region is active.
- `torch.cuda.nvtx.range("conv_fprop_target")` provides a push/pop range. CLI filter: `--nvtx --nvtx-include "conv_fprop_target/"`; trailing slash indicates push/pop syntax.
- Warm up before the range; synchronize before and after the marked call. The label itself does not synchronize.
- Use a separate ncu execution mode without torch.profiler; retain output checks after capture.
- `--launch-count 1` selects one matching launch; counter collection may replay it multiple times.
- First sections: LaunchStats (launch configuration), Occupancy (warp residency), SpeedOfLight (compute/memory throughput relative to hardware limits).
- [NVIDIA NVTX filtering reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering).


## First hardware-counter capture

- One selected kernel can require **multiple replay passes**: our first capture used10 passes for the requested counters.
- Searchoff: **32 blocks on132SMs**; Nsight flags the grid as too small to fill the GPU.
- **Occupancy is not whole-GPU utilization.** Interpret achieved warp residency separately from how broadly work covers the SMs.
- Profiler estimated speedups are suggestions, not measured improvements.
- Capture used cache flushing and unfixed clocks; treat its duration as diagnostic, separate from unprofiled block timing.


## Occupancy recap

- **Occupancy:** resident warps as a fraction of the SM's supported maximum. Our H100 supports **64 warps per SM**.
- **Resident** means resources are allocated on the SM; a warp may be executing or waiting. One warp contains **32 threads**.
- **Theoretical occupancy:** resource-based ceiling for this kernel, assuming enough blocks are available.
- **Achieved occupancy:** measured average warp residency during active cycles; it is not whole-GPU utilization or the fraction of time doing useful work.
- Our kernel: **128 threads/block = 4 warps/block**. Registers permit **2 blocks/SM**, shared memory permits **3**; the tighter register limit gives **8 warps/SM → 12.5% theoretical occupancy**.
- **4 resident warps / 64 = 6.25% occupancy.**
- **Grid coverage is separate:** 32 blocks cannot populate all 132 SMs simultaneously. Higher occupancy alone does not guarantee higher performance.


## Same achieved occupancy, different performance

| Metric | Search off | Search on |
| --- | ---: | ---: |
| Grid blocks | 32 | 128 |
| Theoretical occupancy | 12.5% | 18.75% |
| Achieved occupancy | 6.25% | 6.25% |
| Compute (SM) throughput | 17.05% | 57.77% |
| Profiled duration | 73.98 us | 30.27 us |

- One diagnostic capture per mode, with cache flushing and unfixed clocks; durations are not final benchmark estimates.
- **Achieved occupancy alone cannot explain performance.** Interpret it alongside grid coverage, compute activity and memory behavior.


## Compute throughput and resource accounting

- **SMs are hardware; blocks are launched work.** This H100 has132SMs. A128-block launch can distribute work more broadly than32blocks, but grid size alone does not prove the exact placement.
- **Compute (SM) Throughput** is `sm__throughput.avg.pct_of_peak_sustained_elapsed`: the highest normalized percentage among its constituent activity counters, using elapsed cycles and averaging hardware instances. It is not convolution FLOPs divided by advertised peak FLOP/s.
- Conceptual constituent percentage: `100 * observed activity / (elapsed cycles * peak sustained activity per cycle)`.
- In these saved reports, `sm__issue_active` matches the SM-throughput aggregate: off17.045637%, on57.768060%. This is instruction-issue activity, not occupancy or a direct count of useful FLOPs.
- **Registers** hold thread-local working values; compiler allocation depends on implementation, live temporaries and accumulators. Hardware rounds allocation in chunks.

| Register accounting | Off | On |
| --- | ---: | ---: |
| Reported registers/thread | 255 | 166 |
| Allocated slots/thread (including rounding) | 256 | 168 |
| Allocated slots/block,128threads | 32768 | 21504 |
| Register-limited blocks/SM,65536slots | 2 | 3 |

- Rounded slots are resource accounting, not extra usable registers beyond the per-thread limit. Architecture partitioning/allocation rules matter generally.
- **Shared memory** is block-shared scratch space; per-block demand comes from static declarations and dynamic launch allocation, plus driver reservation. Buffer shapes, dtype, pipeline stages and padding affect demand; do not infer exact cuDNN buffers from the tile name alone.
- Our reports: static0B; dynamic33280/16896B; driver1024B per block. Total34304/17920B.
- Configured shared memory is135168B per SM for these launches (not the full hardware maximum): `floor(135168/34304)=3`, `floor(135168/17920)=7` blocks.
- [Hopper resource limits](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy).


## Driver reservation and throughput from first principles

- **Driver reservation:** CUDA sets aside1024B of shared memory per thread block on H100 for its own use; include it in residency accounting. NVIDIA documents the reservation, not the exact internal contents.
- Searchoff example:33280B kernel dynamic shared memory +0B static +1024B driver =34304B per block.
- **Instruction:** a machine-level step, such as arithmetic, address calculation or loading data. A warp scheduler dispatches ready warp instructions to execution pipelines.
- **Cycle:** one tick of a hardware clock. A pipeline has a limited sustained processing rate per cycle; instructions can take several cycles to finish while later independent instructions enter the pipeline.
- **Throughput:** activity/work divided by time. Toy arithmetic unit:40adds/cycle at1GHz =40billion adds/s; if capacity100adds/cycle, it reaches40% of that capacity.
- Nsight normalizes relevant hardware counters against their own peak rates, then SM throughput selects the maximum constituent percentage. It does not average every pipeline percentage or directly measure useful convolution FLOP/s.
- **Occupancy counts residents; issue activity counts dispatch.** Resident warps may wait for data or dependencies.
- Toy4-SM example: one resident4-warp block on each working SM gives6.25% occupancy with a64-warp limit. If only1SM works, broad-device activity can be much lower than when4SMs work, despite the same active-cycle occupancy.
- Our17.05%/57.77% SM throughput values match instruction-issue activity; do not interpret57.77% as the fraction of convolution completed or as all arithmetic units each running at57.77%.
- [CUDA reservation on Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#unified-shared-memory-l1-texture-cache).


## Hardware counters and pipeline activity

- **Hardware counters** record events or active cycles inside GPU hardware (for example instructions issued, cache transactions or pipeline-active cycles). Nsight derives rates and percentages from counter values; launch resource metadata is not the same as an execution counter.
- Saved normalized pipeline readings, off/on: FMA12.86/35.19%, ALU5.29/31.18%, LSU3.53/17.34%. Instruction issue17.05/57.77%. Collected tensor-MMA activity metric is0 in both reports.
- Percentages use each metric's own peak and can overlap; do not add them or interpret them as instruction-mix shares. FMA also supports some integer arithmetic; LSU includes more than loads/stores.
- Memory instructions contribute to SM-side activity. **Issuing a load is different from completing it**; L1/L2/HBM bytes, throughput, hit rates and latency need their own measurements. A load need not reach HBM if served by cache.
- Kernels can approach100% on a particular throughput metric. Common limits include insufficient blocks, dependent instructions, memory waits, barriers, instruction mix, and startup/tail effects. Occupancy can help hide waits but does not guarantee peak throughput.
- In our pair the small grid is established; exact contributions from dependencies/memory waits need further evidence. Optimize elapsed time/correctness, not a utilization percentage alone.


## Preserve reports for offline study

- Nsight Compute can open saved `.ncu-rep` files on a Mac without a GPU connection; Perfetto opens saved timeline JSON.
- Keep binary reports, text/CSV exports, workload settings, source and numerical checks together. Git ignores our binary reports/tensors/traces, so transfer them separately.
- Search on selected both128x32x8 and32x32x8 tiles in expanded captures. **Identify the kernel before combining measurements.**
- Repeated captures support consistency checks; they do not prove exact causality or independent understanding.
- [Offline report index and reading order](../../../OFFLINE_PROJECT02.md).

## Counters, controllers, and what a percentage measures

- **Controller:** hardware that coordinates actions, such as a memory controller managing memory requests.
- **Counter:** measurement hardware that records events or cycles. Nsight combines readings into rates and percentages; counters do not schedule the workload.
- **Raw count:** e.g. 100 arithmetic operations. **Rate:** operations divided by elapsed cycles or seconds. **Activity:** cycles satisfying a condition, such as a pipeline being active.
- **Toy example:** a unit can process 4 operations/cycle. Across 100 cycles it processes 100 operations during 50 active cycles. Operation throughput is `100 / (4 * 100) = 25%` of capacity; active-cycle fraction is `50 / 100 = 50%`. These answer different questions.
- A **cycle** is a clock tick. A **pipeline** processes instructions in stages, allowing work to overlap.
- **FLOP/s:** floating-point operations per second. The report's **Compute (SM) Throughput** percentage summarizes selected normalized hardware metrics; it is not automatically a FLOP/s measurement.
- In our basic reports, FMA/ALU rows measure pipeline-active cycles; LSU measures executed instructions normalized to its peak. The tensor-MMA-specific counter is zero. Always read the exact metric name.
- **Pitfalls:** percentages need not sum to 100%; issuing a memory instruction does not imply an HBM access completed; a high aggregate does not mean every execution unit is equally busy.
- Sources: [Nsight metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure), [pipeline definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#pipelines).

## Why below 100%, and why summarize with a maximum?

- **Issue capacity can go unused:** resident warps may wait for a load, a prior instruction's result, or a barrier; an instruction may also lack an available execution resource. Another ready warp can sometimes fill that issue opportunity.
- **Coverage matters:** too few blocks can leave SMs without work. Startup and tail periods can also reduce elapsed-cycle averages.
- These are possible mechanisms, not an established breakdown of our 57.77% result. Grid size is known; exact scheduling and waiting contributions require further report analysis.
- **Toy scheduler:** if it can issue once/cycle but issues in only 58 of 100 cycles, issue activity is 58%. This illustrates missing issue opportunities; it is not the exact whole-GPU metric calculation.
- **Aggregate means summary, not necessarily average.** Nsight SM throughput selects the maximum of selected normalized constituent metrics; their own rollups summarize hardware instances.
- **Why maximum:** one pipeline can reach capacity while others are unused. Averaging a hypothetical 100% arithmetic pipe and 0% tensor pipe would give 50% and conceal the saturated arithmetic resource.
- The summary helps locate a potentially limiting resource; it is not proof of a bottleneck, useful FLOP/s, or the sum of pipeline percentages.

## Calculating the warps-per-scheduler graph

- Screenshot bars are consistent with expanded on035: **16, 3, approximately 1.00, 0.65, 0.65**. Screenshot itself does not identify the report.
- **GPU maximum:** recorded H100 supports 64 warps/SM and 4 schedulers/SM, giving 16 resident warps/scheduler. This is residency capacity, not instructions issued each cycle.
- **Theoretical:** 128 threads/block gives 4 warps/block. Recorded block limits: registers 3, shared memory 7, warp capacity 16, blocks 32, barriers 32. Minimum is 3 resident blocks/SM; `3 * 4 / 4 = 3` warps/scheduler, or `3 / 16 = 18.75%` theoretical occupancy.
- Resource arithmetic: 168 allocated register slots/thread times 128 threads = 21,504 slots/block; `floor(65,536 / 21,504) = 3`. Shared-memory allocation 17,920 B/block with 135,168 B configured per SM gives 7 blocks. Allocation granularity and hardware partitioning matter generally.
- **Active:** counter accumulates resident-warp counts over active cycles. `smsp__warps_active.avg.per_cycle_active = 0.997023` in on035.
- **Eligible:** counter accumulates ready-warp counts over active cycles. `smsp__warps_eligible.avg.per_cycle_active = 0.651120`.
- **Issued:** saved issue-activity metric `smsp__issue_active.avg.per_cycle_active = 0.65`, with percentage 65.112002%. Counts cycles with issue; does not count completed warps. The graph is not a count of instructions in flight.
- **Toy one-scheduler calculation:** resident counts `[2,2,2,2,2]` average 2; eligible counts `[0,1,2,0,1]` average 0.8; issue indicators `[0,1,1,0,1]` average 0.6. Two eligible warps do not imply two issued warps that cycle.
- Actual measurements combine hardware instances and use active-cycle normalization. Do not reconstruct them by dividing total launch warps by all hardware schedulers, or assume SM and SMSP active-cycle denominators are identical.
- Theoretical residency assumes enough work. The 128-block grid cannot supply 3 blocks to every one of 132 SMs simultaneously; exact placement is not established by this graph.
