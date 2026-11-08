# 16 — Why the winner flips: 37×2048 versus 37×4096

## Finding

**Best-supported explanation: work distribution and dependency hiding.** At 2048,
our kernel offers more independently scheduled blocks and fewer inputs per thread.
At 4096, PyTorch switches to 32 warps per row, greatly reducing each thread's local
work and providing more eligible warps. Our kernel stays at four warps per row.

This is an explanation supported by source, SASS and profiles, not an isolated
measurement of each mechanism's contribution. The same-H100, same-environment,
repeated benchmarks are sufficient to proceed; a rerun is not a prerequisite.

## Timing and mapping

Timing rows below are **unprofiled benchmark medians**, in µs/call. Profiled
replay durations are diagnostic and are not substituted into the benchmark.

| Quantity | Torch2048 | Ours2048 | Torch4096 | Ours4096 |
|---|---:|---:|---:|---:|
| CUDA Graph median, µs | 6.633 | 3.988 | 3.017 | 6.114 |
| Direct median, µs | 7.633 | 4.873 | 5.065 | 6.982 |
| Kernel family | Warp | Retained Online | Reg | Retained Online |
| Blocks | 10 | 37 | 37 | 37 |
| Threads/block | 128 | 128 | 1024 | 128 |
| Warps cooperating on one row | 1 | 4 | 32 | 4 |
| Input values/thread | 64 | 16 | 4 | 32 |

- Warp packs four independent rows into each 128-thread block; `ceil(37/4)=10`
  blocks. Each row is reduced within its own warp.
- Our kernel always uses one 128-thread block per row. Reg uses one 1024-thread
  block per row here. The H100 has 132 SMs; a block resides on one SM.
- A 10-block grid can occupy at most 10 SMs concurrently; a 37-block grid at most 37.
  These are launch-based bounds, not directly reported per-SM instance counts.

## Measured counters

| Profile metric | Torch2048 | Ours2048 | Torch4096 | Ours4096 |
|---|---:|---:|---:|---:|
| Active warps/scheduler | 1.01 | 1.00 | 7.84 | 1.00 |
| Eligible warps/scheduler | 0.21 | 0.16 | 2.10 | 0.17 |
| Issue active, active-cycle % | 20.69 | 15.77 | 59.72 | 16.66 |
| Achieved occupancy % | 5.92 | 6.24 | 48.63 | 6.24 |
| Executed warp instructions | 90,068 | 140,870 | 357,457 | 250,662 |
| Global-load warp instructions | 2,368 | 2,368 | 4,736 | 4,736 |
| Registers/thread | 80 | 40 | 30 | 56 |
| Global load sectors | 9,472 | 9,472 | 18,944 | 18,944 |
| Global store sectors | 9,472 | 9,472 | 18,944 | 18,944 |
| Local spilling requests | 0 | 0 | 0 | 0 |
| DRAM throughput, elapsed-cycle % | 0.044 | 0.029 | 0.304 | 0.061 |
| L2 read hit rate % | 99.75 | 99.79 | 99.89 | 99.85 |

Active-cycle averages describe the units while active; they do not express
whole-GPU coverage. The occupancy values must not be read as fractions of all 132
SMs kept busy. Executed instructions are warp-level counts, not FLOPs.

## At 2048: ours spreads the work more widely

- PyTorch gives a thread 64 logits; ours gives it 16. Our four warps cooperate on
  a row, while PyTorch uses one warp per row. Our 37 blocks also offer broader SM
  distribution than PyTorch's 10 blocks.
- Both show about one active warp per scheduler. **Ours actually has lower issue
  activity** (15.77% versus 20.69%) and executes **more** warp instructions
  (140870 versus 90068). Better issue efficiency or less total instruction work
  therefore does not explain this win.
- Source/SASS show the tradeoff: ours has a shorter local input sequence, but
  online branching/rescaling, pair merging and two block barriers add work.
  Distributing that work more widely can finish sooner even with more total work.
- **Interpretation:** broader block distribution and more cooperation per row
  outweigh the custom algorithm's overhead for this small-row-count workload.

## At 4096: PyTorch exposes much more parallelism within each row

- Both launch 37 blocks, so the block-count advantage disappears. PyTorch assigns
  1024 threads to each row; ours assigns 128. Inputs per thread are 4 versus 32.
- PyTorch has 7.84 active warps/scheduler and 2.10 eligible; ours has approximately 1
  active and 0.17 eligible. Its issue-active percentage rises to 59.72%, versus
  our 16.66%. This supports better hiding of instruction and memory dependencies:
  the scheduler often has another ready warp when one waits.
- Ours doubles its per-thread values 16→32 and online recurrence length as width
  doubles. Executed instructions rise 140870→250662. PyTorch instead changes the
  mapping from 64 values/thread at 2048 to 4 at 4096, despite the wider row.
- PyTorch executes **more total instructions** than ours here (357457 versus
  250662), yet finishes sooner. Its additional warps and smaller per-thread work
  make those instructions easier to overlap. Higher occupancy is evidence of
  this mapping, not a standalone rule that higher occupancy is always faster.
- **Interpretation:** PyTorch's Reg mapping and short local reductions overcome
  its extra block-reduction work. Our conditional online recurrence remains a
  longer dependency chain on only four warps per row.

## Connect the counters directly to the algorithms

- **Width 2048:** PyTorch's warp kernel loads 64 logits per thread, computes a
  local maximum, reduces within the warp, computes/sums exponentials, reduces
  again, then divides retained exponentials. Our four-warps-per-row mapping gives
  each thread 16 logits; its online recurrence updates the maximum and shifted
  sum together, then merges pairs across the four warps. Extra rescaling/shuffles
  and barriers cost work, but each lane handles fewer inputs and the grid has
  more independently schedulable blocks.
- **Width 4096:** Reg assigns only 4 logits per thread. Its conventional maximum
  and exponential-sum loops are short; 32 warps share each row. Our mapping is
  unchanged, so each thread now carries 32 logits through a conditional online
  recurrence. Each update consumes the previous maximum/sum. With about one
  resident warp per scheduler, there is little alternative warp work when that
  recurrence or its loads/shuffles wait. Reg's ready-warp counts support better
  overlap despite its additional reductions and barriers.

The global-load instruction counts make the redistribution concrete:

| Width | PyTorch: row-serving warps × loads/warp | Ours: warps × loads/warp |
|---:|---:|---:|
| 2048 | `37 × 64 = 2368` | `148 × 16 = 2368` |
| 4096 | `1184 × 4 = 4736` | `148 × 32 = 4736` |

These products match the measured global-load warp-instruction counts. The 2048
PyTorch launch contains 40 warps, but only 37 have valid rows. The products describe
load distribution, not cycles or a predicted speedup. Lower retained values per
thread also do not imply fewer total instructions: pair merging and branching
change the computation.

## Alternative explanations checked

- **Less input traffic?** No: equal global-load and store sectors within each
  shape. At 32 bytes/sector, each variant loads and stores 303104 bytes at 2048,
  and 606208 bytes at 4096—one FP32 input/output pass at the L1/TEX sector level.
  This is not a DRAM-byte count.
- **Spilling?** No local spilling requests or local load/store sectors in any of
  these four profiles. Register counts differ, but spills do not explain this flip.
- **DRAM saturation?** These captures show DRAM throughput below 0.31% and L2
  read hits above 99.75%. They do not support a DRAM-bandwidth limit. Memory latency
  and memory-instruction dependencies may still matter; cache behavior in a
  diagnostic replay is not automatically identical to benchmark replay.
- **Fewer stalls/barriers?** The faster Reg kernel has more barriers and more
  warp cycles per issued instruction. These per-warp quantities cannot substitute
  for scheduler readiness, instruction work and total runtime.

## Source/SASS corroboration

These are static instruction sites in the exported SASS, not dynamic counts:

| Structure | Torch2048 | Ours2048 | Torch4096 | Ours4096 |
|---|---:|---:|---:|---:|
| Input `LDG.E` sites | 64 | 16 | 4 | 32 |
| Output `STG.E` sites | 64 | 16 | 4 | 32 |
| `BAR.SYNC` sites | 0 | 2 | 6 | 2 |

- Reg has four local-max `FMNMX` sites and eight `MUFU.EX2` sites: four for the
  sum and four for output. Its duplicate source exponentials remain visible here.
- Our code contains repeated comparison/branch/exponential online-update steps.
  Branch-target uniformity is 79.31% at 2048 and 77.06% at 4096, versus 100% for PyTorch.
  This supports divergence as additional overhead, without quantifying its cost.
- Algorithm changes and launch changes occur together. The evidence supports
  mapping + local dependency work + readiness as the explanation, not a precise
  percentage attribution to each or a universal claim about online softmax.

## Where to inspect in Nsight

| Question | Section / raw metric |
|---|---|
| Grid/block size, registers | Launch Statistics: `launch__grid_size`, `launch__block_size`, `launch__registers_per_thread` |
| Resident/ready warps | Scheduler Statistics: `smsp__warps_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active` |
| How often schedulers issue | Scheduler Statistics: `smsp__issue_active.avg.pct_of_peak_sustained_active` |
| Achieved occupancy | Occupancy: `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Total executed instructions | Instruction Statistics: `smsp__inst_executed.sum` |
| Global memory sectors | Memory Workload Analysis: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, corresponding `_st.sum` |
| Spills | Memory Workload Analysis: `derived__local_spilling_requests`; local load/store sectors |
| DRAM/L2 reads | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `lts__t_sector_op_read_hit_rate.pct` |

Raw CSV exports contain a units row before the data row; some byte values are
scaled. Details display and raw throughput metrics can use different aggregates,
so do not interchange differently named L1/TEX percentages. Stall ratios use
cycles per issued instruction, not percentages of kernel wall time.

## Evidence and next step

- **PyTorch 37x2048:** [Details](../../results/2026-10-27-offline-profiles/37x2048/pytorch_details.txt), [raw counters](../../results/2026-10-27-offline-profiles/37x2048/pytorch_raw.csv), [SASS](../../results/2026-10-27-offline-profiles/37x2048/pytorch_sass.txt), [report](../../results/2026-10-27-offline-profiles/37x2048/pytorch.ncu-rep).
- **Ours 37x2048:** [Details](../../results/2026-10-27-offline-profiles/37x2048/online_registers_details.txt), [raw counters](../../results/2026-10-27-offline-profiles/37x2048/online_registers_raw.csv), [SASS](../../results/2026-10-27-offline-profiles/37x2048/online_registers_sass.txt), [report](../../results/2026-10-27-offline-profiles/37x2048/online_registers.ncu-rep).
- **PyTorch 37x4096:** [Details](../../results/2026-10-27-offline-profiles/37x4096/pytorch_details.txt), [raw counters](../../results/2026-10-27-offline-profiles/37x4096/pytorch_raw.csv), [SASS](../../results/2026-10-27-offline-profiles/37x4096/pytorch_sass.txt), [report](../../results/2026-10-27-offline-profiles/37x4096/pytorch.ncu-rep).
- **Ours 37x4096:** [Details](../../results/2026-10-27-offline-profiles/37x4096/online_registers_details.txt), [raw counters](../../results/2026-10-27-offline-profiles/37x4096/online_registers_raw.csv), [SASS](../../results/2026-10-27-offline-profiles/37x4096/online_registers_sass.txt), [report](../../results/2026-10-27-offline-profiles/37x4096/online_registers.ncu-rep).
- [Recorded PyTorch source](../../results/2026-10-27-offline-profiles/source/pytorch/SoftMax.cu),
  [warp source](../../results/2026-10-27-offline-profiles/source/pytorch/PersistentSoftmax.cuh),
  [custom source](../../results/2026-10-27-offline-profiles/source/softmax.cu).
- [Timing inventory](09-results-and-kernel-inventory.md), [selected cases](15-blog-case-selection.md).
- [NVIDIA metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure)
  and [scheduler state model](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#hardware-model).
- Next: discuss this explanation, then compare 37×4096 with 4096×4096 to see why
  changing only the row count reverses the Reg-versus-retained result.
- [Section index](README.md). Source/results are separate downloaded artifacts;
  no new GPU run was performed for this analysis.
