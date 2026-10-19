# GPU memory: throughput, caches, and wavefronts

## Quick revision

- **Bandwidth:** rate of data transfer. **Latency:** time until a requested result becomes available.
- **Low DRAM throughput is not inherently bad:** compute and cache/shared-memory reuse can reduce DRAM demand; dependency waits can also leave bandwidth unused.
- **Cache hit rate counts accesses, not time.** L1 and L2 have different denominators.
- **One instruction can generate several wavefronts.** More service steps do not mean more CUDA warps or necessarily proportional end-to-end latency.
- **Available capacity needs ready work:** a kernel can stall while hardware bandwidth remains available.

## Measurement and normalization

- Hardware counters record DRAM traffic; this is not the sum of tensor sizes or every program load. Cache hits can avoid DRAM reads.
- **Absolute rate:** measured DRAM bytes / measurement duration.
- **Percentage of peak:** 100 × measured bytes / peak bytes transferable over elapsed memory-unit cycles, with matching hardware-unit aggregation. Peak capacity uses the hardware sustained bytes/cycle rate.
- **Elapsed** includes cycles with no traffic. Memory-unit cycles are not SM cycles. Peak per-second bandwidth depends on the clock rate.

## Different hardware clocks: match the denominator

- **Different hardware units can have different clock rates.** SM cycles and DRAM-unit cycles need not represent the same duration.
- **Clock domain:** hardware whose cycles are counted using a particular clock. Match the counter’s clock domain, measurement interval, and hardware scope when normalizing.
- **Toy example, not measured H100 frequencies:** over the same 1 μs interval:

| Hardware clock | Frequency | Elapsed cycles |
| --- | ---: | ---: |
| SM | 1 GHz | 1,000 |
| DRAM unit | 2 GHz | 2,000 |

- A scheduler issuing during **300 SM cycles** has elapsed issue activity `300 / 1,000 = 30%`. Dividing by 2,000 DRAM cycles gives 15%, but **not** the scheduler’s issue-active percentage.
- If the scheduler had resident warps during only **400 SM cycles**, its active-cycle issue percentage is `300 / 400 = 75%`. Active versus elapsed is a separate denominator choice from clock domain.
- **DRAM normalization:** `100 × DRAM bytes / (elapsed DRAM-unit cycles × peak bytes per DRAM-unit cycle)`, with matching hardware aggregation.
- **Common-time alternative:** convert activity to work/second and compare with the matching peak work/second. Do not directly substitute another unit’s cycle count.
- **Pitfall:** bytes/cycle is incomplete without specifying the hardware unit/interface, clock domain, and whether the scope is per unit or whole device.
- Source: [NVIDIA cycle-metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure).

## Saved on035 evidence

| Counter | Value |
| --- | ---: |
| `dram__bytes_read.sum` | 2.299392 MB |
| `gpu__time_duration.sum` | 30.144 µs |
| `dram__bytes_read.sum.per_second` | 76.280255 GB/s |
| `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` | 2.279788% |
| Recorded DRAM writes | 0 bytes |

- Screenshot Details calls the absolute rate **Memory Throughput**. Find the percentage in **Raw** using the exact counter above.
- Dividing the observed rate by 0.02279788 implies approximately 3.346 TB/s of peak bandwidth for this normalization; this is inferred from the counters, not a separately measured bandwidth benchmark.
- Low average DRAM bandwidth does not exclude memory latency or cache/shared-memory bottlenecks. Do not equate Details **Max Bandwidth** with this DRAM percentage.

## Source and evidence

- [NVIDIA normalization explanation](https://forums.developer.nvidia.com/t/how-are-pct-of-peak-metrics-calculated/330187).
- Saved evidence: `progress/sessions/2026-10-01-project-02-offline-batch/ncu_on_035-raw.csv` (repository-relative).

## Shared SRAM: L1 cache versus shared memory

- **SRAM** is the storage technology; **cache** and **shared memory** describe how storage is managed. L1 caching is automatic; shared-memory contents are controlled by the program.
- H100 has a 256 KiB combined L1/texture/shared-memory resource per SM. A per-kernel **carveout** reserves a supported capacity for shared memory. This differs from the program's shared-memory allocation **per block**.
- CUDA selects a compatible configuration; the application can request a preferred carveout. Unused bytes inside the reserved shared-memory portion do not automatically become L1 capacity.
- Shared addresses select program-managed storage; global addresses can be looked up in the L1 cache using hardware tags (depending on cache policy). The cache cannot evict a live shared-memory allocation.
- **Dense-layer example:** Y=XW, with 32×32 FP32 X and W tiles, needs 8 KiB of user shared memory per block for the two arrays (ignoring other allocations and runtime reservations). The block cooperatively loads them, synchronizes, and reuses them. Ordinary global loads during staging can use L1; reads of the shared tiles use shared memory directly. This is an illustrative SIMT implementation, not the measured cuDNN kernel.
- **Convolution example:** a direct implementation's repeated global input reads may hit L1 automatically; a tiled implementation explicitly stages an input patch in shared memory and controls its reuse. These are different management mechanisms using the same underlying SRAM technology.
- Source: [Hopper unified L1/shared-memory resource](https://docs.nvidia.com/cuda/hopper-tuning-guide/#unified-shared-memory-l1-texture-cache).

## Cache hit rates and reuse

- **Hit rate:** `100 × sector accesses that hit / total sector accesses at that cache`. A sector is 32 bytes.
- Saved screenshot: **L1/TEX 65.93%**, **L2 87.94%**. L2's denominator is traffic arriving at L2, not all original loads.
- **Simplified read-only example:** 100 sector accesses → 66 L1 hits → 34 reach L2 → about 30 L2 hits → about 4 continue beyond L2. This assumes every read checks L1 first; it is not an exact reconstruction of aggregate report traffic.
- **Why L1 hits coexist with shared memory:** blocks on the same SM can load overlapping convolution input regions into their separate shared tiles. Later global loads may hit L1; subsequent reads of the shared arrays do not count as L1 hits.
- One miss followed by two hits gives a 66.7% hit rate. This illustrates reuse, not the measured kernel's exact access sequence.
- Cache entries may be evicted; reuse benefits depend on the access path and cache policy. A hit rate alone does not identify which tensor supplied the hits.
- [NVIDIA cache-table definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-tables).

## Three memory utilization summaries

| Screenshot field | Value | What it checks |
| --- | ---: | --- |
| Mem Busy | 41.91% | Utilization of memory hardware units |
| Max Bandwidth | 30.28% | Utilization of data-transfer bandwidth between memory units |
| Mem Pipes Busy | 17.28% | How heavily the SM’s memory-instruction processing pipelines are used relative to their capacity |

- These are distinct utilization summaries, not percentages of allocated memory, and are not additive portions of runtime.
- Max Bandwidth is not the DRAM-only percentage. Aggregate utilization can conceal temporal bursts or imbalance.
- [NVIDIA section definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules).

## Hardware names and sharing

| Name | Meaning |
| --- | --- |
| **LTC** | Level 2 cache |
| **LTS** | One L2 slice/sub-partition; includes storage and request-handling logic |
| **TEX** | Texture fetching/filtering hardware, e.g. sampling image data |
| **L1TEX** | Combined L1/texture subsystem; also handles ordinary CUDA memory traffic |

- **Profiler prefixes:** `lts__t`, `lts__m`, and `lts__d` refer to L2 tag, miss, and data stages.
- **Our H100:** saved on035 attributes report **132 SMs** sharing **50 MiB L2** (52,428,800 bytes). SM counts depend on the H100 variant.
- **Sharing:** each SM has its own L1; the overall L2 is shared across the GPU and physically divided into partitions/slices.
- **DL example:** blocks on different SMs reading the same weights can benefit from shared L2 contents. They cannot directly reuse each other's ordinary L1 entries.
- **Pitfalls:** TEX is not a Tensor Core; L1/TEX hits do not prove texture instructions were used. Shared L2 does not mean one serial controller or a fixed equal number of SMs per slice.
- [Profiler hardware names](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder), [Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).

## How the three summaries are measured

- Hardware counters count instructions, internal work packages, transfer activity, and elapsed cycles. A component utilization is `100 × activity / (elapsed cycles × sustained peak activity per cycle)`, with consistent unit-instance aggregation.
- Each throughput summary takes the **maximum normalized constituent**, not the sum or average across different resources. `.avg` refers to aggregation across hardware instances; this is distinct from selecting the maximum constituent.
- **Mem Pipes Busy:** instruction-handling capacity in memory pipelines. A warp instruction can generate multiple pieces of downstream work.
- **Mem Busy:** internal service capacity of memory hardware. L1TEX divides requests into **wavefronts** (work packages processed by its pipeline); these are not CUDA warps.
- **Max Bandwidth:** utilization of transfer paths, possibly measured through transfer-active cycles rather than literal byte counters. It is neither DRAM-only bandwidth nor the maximum instantaneous bandwidth observed.

| Summary metric (all `.avg.pct_of_peak_sustained_elapsed`) | Saved on035 value | Matching component in saved CSV |
| --- | ---: | --- |
| `gpu__compute_memory_access_throughput` | 41.909052% | `l1tex__data_pipe_lsu_wavefronts` |
| `gpu__compute_memory_request_throughput` | 30.280275% | `l1tex__lsu_writeback_active` |
| `sm__memory_throughput` | 17.281627% | `sm__inst_executed_pipe_lsu` |

- Matches were rechecked from raw CSV; full architecture-specific constituent membership was not independently queried from PerfWorks. Do not treat these matching components as the sole possible inputs on every GPU.
- Toy example: over 100 cycles, 20 instructions at 1/cycle peak = 20%; 40 work packages at 1/cycle peak = 40%; 30 transfer-active cycles at 1/cycle peak = 30%. These hypothetical capacities and counts illustrate distinct denominators, not H100 specifications.
- [NVIDIA normalization and throughput aggregation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure).

## Q&A: Mem Busy, Mem Pipes Busy, and Max Bandwidth

**Q: What is the difference from first principles?**
- **Mem Pipes Busy:** how heavily the SM’s memory-instruction processing pipelines are being used relative to their capacity.
- **Mem Busy:** capacity of memory hardware to service their requests internally.
- **Max Bandwidth:** capacity of paths that transfer data between units.
- One warp instruction may generate multiple internal work packages and transfers, so these rates can differ.

**Q: What does “pipeline” mean here?**
- A **hardware processing path** inside the SM that handles a class of instructions; for memory instructions, think of the load/store path associated with the **LSU (Load/Store Unit)**.
- The **warp scheduler issues** an instruction; the memory pipeline **handles** it and generates requests to the appropriate memory system.
- Successive instructions can be processed while earlier requests await data. Accepting an instruction is not the same as completing its memory access that cycle.
- Example: a load uses this instruction path but hits in L1, requiring no DRAM access. **Mem Pipes Busy is not DRAM utilization.**

**Q: Does Mem Busy measure a fixed 100-cycle interval?**
- No. **100 cycles was a teaching example.** These report summaries normalize activity over elapsed unit cycles during the profiled interval, including idle cycles.

**Q: Does “one work package per cycle” mean the load completes in one cycle?**
- No. It describes a stage's **service throughput**, not full load latency. A hypothetical pipeline can accept one package each cycle while each package takes four cycles to traverse it.

**Q: What is the wavefront size on H100 and B200?**
- There is no universal byte size for every path/stage. A wavefront groups accesses serviced together; it is not a CUDA warp.
- For ordinary shared-memory bank service, the documented model is **32 banks × 4 bytes = 128 bytes per service cycle**. Conflicts and broadcasts change the distinct bytes handled per step.
- Do not apply this number to every L1, tensor, or TMA path. A complete H100/B200-specific wavefront-capacity table was not verified; no B200 measurement was made.

**Q: Is bytes/cycle different for different memory units?**
- Yes. Specify **which interface, direction, clock domain, and scope** (one unit or whole GPU). The saved H100 DRAM normalization implies **1,280 aggregate bytes per DRAM-unit cycle**, not per SM or per controller.
- Exact counter peaks can be obtained using `.peak_sustained` on the target GPU; preserve matching `.avg`/`.sum` scope.

**Q: Are these three percentages based on different hardware counters?**
- Yes: groups of relevant counters, not necessarily one counter per label. Count activity → normalize each constituent against its own sustained peak over elapsed cycles → select the **largest constituent percentage** in the group.
- `.avg` aggregates hardware instances; it does not mean averaging the percentages of unrelated resources. The table above lists observed on035 component matches, not exhaustive membership.

**Q: Does 41.91% Mem Busy mean memory consumed or stalled 41.91% of runtime?**
- No. It is a normalized throughput summary, not allocated capacity or time lost to stalls. The three percentages can overlap and must not be added.

## Service rate, wavefront size, and clock domains

- The 100-cycle window was hypothetical. Saved on035 has about 59,590 average elapsed L1TEX cycles and 78,797 average elapsed DRAM cycles. Different unit clocks mean different cycle counts over the same measurement interval.
- One wavefront/cycle is a pipeline service-rate statement, not a one-cycle end-to-end load latency.
- A wavefront is a stage-specific service grouping, not a fixed-byte packet across all L1TEX paths or architectures. Keep warp (32 threads), sector (32 bytes), and wavefront distinct.
- CUDA's ordinary shared-memory bank model is 32 banks × 4 bytes/bank/cycle = 128 bytes/cycle of bank service. A conflict-free warp of scalar FP32 shared loads can use all banks; conflicts or broadcast change how many distinct bytes a service step handles. This does not specify every tensor/TMA or L1 interface rate, nor a universal H100/B200 wavefront byte size.
- For the saved H100 report, 2,299,392 DRAM read bytes / (78,796.8 average elapsed DRAM cycles × 0.02279788) implies **1,280 aggregate peak bytes per DRAM-unit cycle**. This is inferred from reported normalization, not a newly collected peak metric.
- Query/collect the relevant counter's `.peak_sustained` on the target GPU for its exact peak; match `.avg`/`.sum`, units, direction, and clock domain. No B200 measurement was performed.
- [Shared-memory bank model](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-access-patterns).

### Extra service work versus total completion latency

- A request can spend time in **address processing, queues, bank service, and returning its result**. A bank conflict increases service work at one stage; the other portions need not double.
- **Separate simplified example, not measured GPU timing:** assume other stages remain unchanged and these portions add without overlap.

| Portion of latency | Without conflict | With conflict |
| --- | ---: | ---: |
| Other stages | 6 cycles | 6 cycles |
| Bank service | 1 cycle | 2 cycles |
| **Total** | **7 cycles** | **8 cycles** |

- **Two service steps instead of one does not necessarily mean twice the end-to-end latency.** Actual timing can also involve overlap and queueing, so this table is an illustration, not a universal timing formula.

## Worked example: one warp, scalar shared loads

Assume 32 active lanes, initialized FP32 shared array `s`, and one scalar shared-load instruction per warp. For clarity, `s[0]` maps to bank 0; `bank(s[i]) = i % 32`.

| Access | Bank mapping | Intrinsic service work |
| --- | --- | --- |
| `s[lane]` | 0, 1, …, 31 | 1 wavefront: one word from every bank |
| `s[2*lane]` | 0, 2, …, 30, 0, 2, …, 30 | 2 wavefronts: two distinct words per used bank |

- Stride-2 bank 0 receives `s[0]` (lane 0) and `s[32]` (lane 16); bank 2 receives `s[2]` and `s[34]`, and so on. Different addresses in each bank require separate service steps.
- Illustrative grouping: wavefront 1 serves lanes 0–15; wavefront 2 serves lanes 16–31. This grouping explains the count, not a guaranteed arbitration order. Each step serves 16 distinct floats (64 bytes), totalling the same 128 requested bytes.
- The warp is not split into new CUDA warps and the program still executes one scalar load instruction. Extra internal service work does not imply that end-to-end load latency exactly doubles. Other contention is excluded.
- Same-address reads are broadcasts, not conflicts: distinguish `s[0]` requested twice from `s[0]` and `s[32]` sharing bank 0.
- This is a worked-through illustration, not measured kernel evidence.

## Current case: supported conclusion versus hypothesis

| Evidence | What it supports |
| --- | --- |
| DRAM reads at 2.28% of peak over elapsed cycles | No evidence of sustained DRAM bandwidth saturation |
| Roughly one active warp per scheduler | Few alternative resident warps to cover waits |
| Wait: 382 all samples, 376 no-issue samples | Sampled fixed-latency dependency waits often coincide with no scheduler issue |
| Dispatch Stall: 223 all samples, 5 no-issue samples | Far fewer sampled dispatch stalls coincide with no scheduler issue |

- **Working hypothesis:** execution dependencies plus limited independent work restrict instruction issue. This does not identify the exact instruction sequence or predict recoverable runtime.
- **Example dependency:** `acc = x0*w0 + acc;` then `acc = x1*w1 + acc;`. The second update needs the first result. Spare DRAM bandwidth cannot produce that arithmetic result sooner.
- Another eligible warp or an independent instruction may cover the wait; occupancy alone does not guarantee readiness.
- Low DRAM throughput also does not rule out individual memory-load latency, local bursts, imbalance, or other memory-path limits. **Wait** specifically concerns fixed-latency execution dependencies; memory dependencies have scoreboard categories.
- **Avoid:** saying most runtime was spent in on-chip transfers without evidence; treating sample counts as percentages of lost runtime; calling the kernel definitively compute-bound solely because DRAM utilization is low.
- See [latency hiding and stall interpretation](16-latency-hiding.md).
