# cuDNN search off/on: performance synthesis

## Observed comparison

Same synthetic forward convolution; FP32, TF32 disabled. Search on/off refers to cuDNN algorithm benchmarking, not precision. Expanded captures off030/on035:

| Measurement | Off | On |
| --- | ---: | ---: |
| Tile | 256×64×8 | 128×32×8 |
| Blocks × threads/block | 32 × 128 | 128 × 128 |
| Registers/thread | 255 | 166 |
| Achieved active-cycle occupancy | 6.25% | 6.25% |
| Active warps/scheduler | ~1 | ~1 |
| Issue-active percentage of active cycles | 72.40% | 65.11% |
| Elapsed SM throughput | 17.06% | 57.58% |
| Profiled duration | 73.984 µs | 30.144 µs |

## Explanation supported by these observations

- Smaller output tile → more blocks for this fixed workload → opportunity to engage more of the GPU's 132 SMs. Grid ceilings: 32 occupied SMs versus 128; exact placement is not established by grid size alone.
- One 128-thread block supplies four warps; with four schedulers/SM, both configurations can have roughly one resident warp/scheduler on occupied SMs. Active-cycle occupancy does not count idle units like elapsed utilization does.
- On is faster despite lower issue rate during scheduler-active cycles. The evidence supports better device coverage as a major contributor, not improved per-active-scheduler issue efficiency.
- Lower per-block resource demands increase theoretical residency, but measured achieved occupancy remains similar. Do not claim the speedup comes from higher achieved occupancy.
- On035's low DRAM utilization and Wait/no-issue samples support spare DRAM capacity plus remaining latency-hiding limitations; they do not establish the entire off/on causal breakdown.

## Performance result and boundaries

- Separate unprofiled campaign: six fresh processes, three per mode; median of process medians **67.76275 µs off → 27.65771 µs on**. About **2.45× speedup / 59.18% less operation-call time**.
- Profiled pair: about 2.45× as well, but profiler timings remain diagnostic and are not substituted for benchmarks.
- Kernel choice changed in additional on captures (512-block variant); captured identities do not prove which kernel ran in every separate benchmark process.
- Different kernels change more than grid size. Coverage is evidence-backed interpretation, not a controlled experiment assigning an exact percentage of the gain to one cause.
- Numerical criterion still fails **4,736 / 524,288** elements. One saved off/on pair has bitwise-equal outputs; this does not establish acceptance against the reference.
- This is synthetic forward-operation performance, not a 2.45× full-training speedup.

## Evidence and next step

- [Offline handoff and benchmark record](../../../progress/sessions/2026-10-01-project-02-offline-handoff.md).
- Raw expanded CSVs: `progress/sessions/2026-10-01-project-02-offline-batch/ncu_off_030-raw.csv` and `ncu_on_035-raw.csv` (repository-relative).
- [Memory findings](../03-gpu-hardware-and-scheduling/17-dram-throughput.md), [latency hiding](../03-gpu-hardware-and-scheduling/16-latency-hiding.md).
- Next: transfer this reasoning to the saved alternate on031/on032 512-block variant. Explanation above is assisted; independent transfer remains unassessed.

## Transfer comparison: 128 versus 512 blocks

Both are **search-on** variants for the same workload; comparisons below use on035 and on031, with repeat timings checked separately.

| Measurement | 128-block on035 | 512-block on031 |
| --- | ---: | ---: |
| Tile | 128×32×8 | 32×32×8 |
| Threads/block | 128 | 64 |
| Warps/block | 4 | 2 |
| Total launched warps | 512 | 1,024 |
| Registers/thread | 166 | 92 |
| Dynamic shared bytes/block | 16,896 | 6,528 |
| Register-limited blocks/SM | 3 | 10 |
| Theoretical occupancy | 18.75% | 31.25% |
| Achieved occupancy | 6.25% | 12.04% |
| Active warps/scheduler | 0.997 | 1.947 |
| Eligible warps/scheduler | 0.651 | 1.197 |
| Issue-active % of active cycles | 65.11% | 75.21% |
| Elapsed SM throughput | 57.58% | 67.57% |
| Profiled duration | 30.144 µs | 29.920 µs |

- Repeat timings: 128-block on035/036/037 = 30.144/29.920/29.920 µs; 512-block on031/032 = 29.920/29.920 µs. No clear runtime advantage is established by these captures.
- Four times as many blocks means only twice as many launched warps here, because threads/block halves. Total launched warps are not necessarily simultaneously resident.
- Transfer question discussed: higher utilization does not guarantee proportional speedup. The checked instruction-work comparison follows below.

## Checked prediction: more instructions

Learner predicted more memory-transfer and accumulation instructions. Saved on035/on031 counters show:

| Count | 128 blocks | 512 blocks | Change |
| --- | ---: | ---: | ---: |
| Executed warp instructions | 18,097,152 | 21,179,392 | +17.03% |
| Shared-load instructions | 902,656 | 1,198,080 | +32.73% |
| Global-to-shared LDGSTS instructions | 378,880 | 606,208 | +60.00% |
| Branch instructions | 52,736 | 107,520 | +103.88% |
| DRAM bytes read | 2,299,392 | 2,282,240 | Slight decrease |

- Total executed instructions increase about 17.03%; shared loads about 32.73%; LDGSTS 60%. More memory instructions are observed, but DRAM read bytes do not increase. Instruction counts are not transferred-byte counts.
- The inspected raw exports/logs do not include a direct FFMA instruction count; more accumulation instructions is **unverified**, not established by more blocks or total instructions.
- More executed work alongside higher processing rate is consistent with similar runtimes. This is an aggregate observation, not a complete causal or instruction-cost model.
- **Why loads can increase without HBM bytes increasing:** global requests can hit caches; shared loads access on-chip shared memory directly. Different instruction widths can also change instruction counts without proportional byte changes.
- **Why accumulation need not increase:** smaller output tiles can distribute the same mathematical work among more blocks. More total machine instructions do not establish more multiply-accumulates.
- **Revision takeaway:** higher instruction throughput is not automatically higher useful-work throughput; compare total instruction work, transferred bytes, and measured runtime separately.

## Finding instruction counters in Nsight Compute

1. Open the expanded **on035** (128-block) or **on031** (512-block) report.
2. In **Details → Instruction Statistics**, locate **Executed Instructions** for the total.
3. For exact counters and categories, switch to **Raw** and search using the names below. If an exact-name filter finds nothing, start with the prefix `smsp__inst_executed` or `smsp__sass_inst_executed`.
4. Compare matching counters between the two captures; keep **executed** and **issued** totals separate.

| Quantity | Raw metric |
| --- | --- |
| Total executed warp instructions | `smsp__inst_executed.sum` |
| Total issued warp instructions | `smsp__inst_issued.sum` |
| Shared-memory load instructions | `smsp__sass_inst_executed_op_shared_ld.sum` |
| Global-to-shared copies (LDGSTS) | `smsp__inst_executed_op_ldgsts.sum` |
| Branch instructions | `smsp__inst_executed_op_branch.sum` |

- **Name breakdown:** `smsp` = SM sub-partition; `inst_executed` = executed warp instructions; `.sum` = sum across hardware instances.
- **Units:** these are warp-instruction counts, not per-thread instructions, FLOPs, or bytes transferred. A warp instruction can act on multiple participating threads; multiplying by 32 is not generally valid when lanes are inactive or predicated off.
- **How we checked:** parsed the saved raw CSVs with Python and cross-checked the total against **Instruction Statistics** in `ncu_on_035.log` and `ncu_on_031.log`. No instruction counts were inferred from grid size; no new GPU run was performed.
- **Availability:** use the expanded captures for these counts. If a metric was not collected in a report, switching tabs cannot recover it. Direct FFMA counts were not present in the exports inspected here.

## Nsight lookup: activity, instruction work, and traffic

| Question | Details location | Raw counter / interpretation |
| --- | --- | --- |
| How often does a scheduler issue while active? | Scheduler Statistics | `smsp__issue_active.avg.pct_of_peak_sustained_active`; on035 = 65.112002% |
| How much issue activity over elapsed time? | Compute Workload Analysis / throughput breakdown | Saved on035 `sm__issue_active.avg.pct_of_peak_sustained_elapsed` = 57.580204%; different scope/denominator from scheduler-active metric |
| How many instructions executed? | Instruction Statistics → Executed Instructions | `smsp__inst_executed.sum`; on035 = 18,097,152 |
| Which instructions? | Instruction Statistics, Instructions/Opcode when collected; Source/SASS for locations | Use exact shared-load/LDGSTS/branch counters listed above; pipe utilization is not opcode count |
| How many bytes moved? | Memory Workload Analysis → Memory Tables → Device Memory, L2, L1/TEX, Shared | `dram__bytes_read.sum`, `dram__bytes_write.sum` for DRAM; distinguish other levels |

- On035 DRAM reads = 2.299392 MB, writes = 0 bytes in this capture. **Bytes** measures traffic volume; **Throughput** is bytes/second; **% Peak** measures normalized rate.
- In the screenshot the section dropdown is **Memory Tables**. Use per-unit tables to avoid equating all memory traffic with DRAM traffic. Hover a cell for its underlying metric/description where supported.
- Raw filters can use prefixes (`smsp__issue_active`, `smsp__inst_executed`, `dram__bytes`) when exact names are difficult to find.
- No current live UI inspection; navigation is based on the documentation and supplied screenshot. Opcode breakdown availability depends on collected data; the inspected raw exports lack a direct FFMA count.

## Integer instructions versus address arithmetic

- **Integer counter:** search Raw for `smsp__sass_thread_inst_executed_op_integer_pred_on.sum` (availability depends on GPU/version and collection).
- **Unit:** predicated-on **thread instructions**, unlike the warp-instruction counts above. One integer warp instruction with 32 participating, predicate-true threads contributes 32 thread instructions; with 8, it contributes 8. Do not assume a fixed factor of 32.
- **Address arithmetic is a purpose, not a separate clean opcode category.** Integer instructions may calculate tensor addresses, update loop counters, or compute integer results.
- Example: `address = base + index * element_size`. Inspect **Source → SASS** and follow the computed value into a memory instruction to identify address calculation.

| Breakdown (newer Nsight versions, if collected) | Metric |
| --- | --- |
| Executed warp instructions by opcode category | `sass__inst_executed_per_opcode_category` |
| Executed warp instructions by opcode | `sass__inst_executed_per_opcode` |

- **Executed counts versus code listing:** counting SASS lines does not count dynamic executions; loops can execute a line repeatedly. Source-correlated execution counts require collected profiling data.
- **Pitfalls:** throughput/utilization does not identify which instruction counts increased. Missing metrics are not zero; a tab change cannot recover uncollected data. No integer/address count has been established for our saved comparison.
- Sources: [published integer-counter mapping](https://pmc.ncbi.nlm.nih.gov/articles/PMC8633859/), [NVIDIA opcode metrics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#instructions-per-opcode-metrics), [NVIDIA Source view](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#source-page).
