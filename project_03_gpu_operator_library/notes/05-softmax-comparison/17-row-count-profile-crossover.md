# 17 — Why more rows favor ours: 37×4096 versus 4096×4096

**Best-supported explanation:** many independent rows let our small blocks hide
latency using other rows while doing less instruction/reduction work per row.
PyTorch's large blocks favor short row latency when rows are scarce. The larger
input also changes cache behavior, so this is not a pure block-count experiment.

## Timing and unchanged algorithms

Unprofiled benchmark medians, in microseconds/call:

| Shape | PyTorch graph | Ours graph | PyTorch direct | Ours direct | Graph winner |
|---|---:|---:|---:|---:|---|
| 37 × 4096 | 3.017 | 6.114 | 5.065 | 6.982 | PyTorch, 2.03× |
| 4096 × 4096 | 58.885 | 49.372 | 60.683 | 50.681 | Ours, 1.19× |

Both implementations take longer for the larger input; ours becomes faster
**relative to PyTorch**. Each uses the same specialization at both row counts.

| Per-row property | PyTorch Reg | Our retained Online |
|---|---:|---:|
| Threads / warps | 1024 / 32 | 128 / 4 |
| Input values/thread | 4 | 32 |
| Registers/thread | 30 | 56 |
| Resident block limit per SM | 2 | 9 |
| Local work | Separate max and exponential-sum loops | Conditional coupled max/sum recurrence |
| Block-reduction work | Two scalar reductions across 32 warps | One pair reduction across 4 warps |
| Block barriers in source/SASS | 6 | 2 |

Both retain original logits and read each input once at source level. Ours still
has the longer per-thread recurrence; increasing rows does not shorten it.

## What additional rows change

- **37 rows:** both launch 37 blocks, at most 37 concurrently occupied SMs on the
  132-SM device. There are too few blocks to fill either resident-block capacity.
  Reg exposes 32 warps from each row; ours exposes four. Reg therefore has far
  more alternative warps when one waits, and its small local loops finish sooner.
- **4096 rows:** both have ample queued row-blocks. Resource limits allow up to
  **two Reg rows or nine custom rows resident per SM**. Our scheduler can now issue
  work from another row while a warp in the current row waits at a dependency or
  block barrier. Each block's barrier concerns its own row, not all resident rows.
- Reg's limit is two blocks from warp/register limits; ours is nine from registers.
  These are resource ceilings, not claims that every SM holds that count at every
  instant. The observed rise in resident/eligible warps confirms increased supply.

## Measured scheduler and work counters

| Metric | Torch, 37 rows | Ours, 37 rows | Torch, 4096 rows | Ours, 4096 rows |
|---|---:|---:|---:|---:|
| Active warps/scheduler | 7.84 | 1.00 | 15.19 | 8.11 |
| Eligible warps/scheduler | 2.10 | 0.17 | 3.87 | 1.26 |
| Issue-active cycles, % | 59.72 | 16.66 | 68.36 | 58.57 |
| Achieved occupancy, % | 48.63 | 6.24 | 93.78 | 50.52 |
| Executed warp instructions | 357,457 | 250,662 | 39,571,456 | 27,748,704 |
| Shared-load warp instructions | 2,442 | 185 | 270,336 | 20,480 |
| Shared-store warp instructions | 2,442 | 185 | 270,336 | 20,480 |
| L2 read hit rate, % | 99.8854 | 99.8544 | 0.0069 | 0.0066 |
| DRAM throughput / peak, % | 0.30 | 0.06 | 66.53 | 79.54 |

- Our issue activity rises **16.66% → 58.57%**, with active warps/scheduler
  **1.00 → 8.11**. Additional independent rows hide waits that previously left
  our scheduler idle. Reg improves too, but its issue rate rises only
  **59.72% → 68.36%**.
- At 4096 rows Reg still has **higher** occupancy, readiness and issue activity.
  Our win therefore is not “higher occupancy” or a higher issue rate than Reg.
  Active-cycle percentages also do not measure the fraction of the whole GPU busy.

## Link to algorithm work: fewer instructions per completed row

| Work per row | PyTorch Reg | Ours |
|---|---:|---:|
| Executed warp instructions, 37-row capture | 9,661 | 6,774.65 |
| Executed warp instructions, 4096-row capture | 9,661 | 6,774.59 |
| Shared-load warp instructions | 66 | 5 |
| Shared-store warp instructions | 66 | 5 |
| Global-load / global-store warp instructions | 128 / 128 | 128 / 128 |

- Our **~30% smaller warp-instruction workload per row already existed at 37 rows**.
  It becomes useful once there are enough independent rows to keep execution
  moving. Reg's issue-rate advantage shrinks, while its larger instruction
  workload remains. Total work plus execution rate explain more than occupancy
  or total thread count alone.
- Reg launches 32 warps per row and combines their summaries twice: maximum and
  sum. Each block reduction writes 32 warp partials, then broadcasts the result.
  Ours writes four warp pairs and combines those pairs once. This explains the
  shared-instruction disparity and fewer block barriers. Online rescaling and
  branching still add work; the entire instruction difference is not all barriers.
- SASS sites for each kernel are unchanged between row counts. Ours' branch
  uniformity is essentially unchanged (~77.06%); there is no newly favorable
  branch pattern or new kernel algorithm in the large case.

## Memory also becomes a larger part of this workload

- A 37×4096 FP32 input is 606,208 bytes (~0.578MiB); a 4096×4096 input is 64MiB.
  The device reports 50MiB L2. The larger input alone exceeds that capacity.
- Captured L2 **read** hits fall from ~99.85–99.89% to ~0.0066–0.0069%. DRAM reads
  rise from essentially zero to ~67.1MB for both kernels. These are profile/cache
  observations, not a claim of identical cache state in every benchmark replay.
- At 4096 rows, L1/TEX input/output sectors are exactly equal: 2,097,152 sectors
  each direction, or 64MiB read and 64MiB written per kernel. DRAM traffic is nearly
  equal (133.83MB Reg versus 132.27MB ours); all four profiles show zero spilling.
- Large-case profile throughput is **2.23TB/s Reg versus 2.67TB/s ours**, or
  **66.53% versus 79.54%** of the reported DRAM peak. Our implementation sustains
  faster completion of nearly the same traffic. Higher bandwidth is an observed
  outcome, not independent proof of the cause or proof of complete saturation.

**Interpretation:** with many rows, our 128-thread mapping can exploit parallelism
between rows. Its lower aggregate warp/reduction work helps sustain throughput
once it has sufficient independent work to hide dependencies. Cache capacity and
memory concurrency also change; these profiles do not isolate an exact speedup
contribution for each mechanism.

## Inspection guide and limits

- **Launch Statistics:** `launch__occupancy_limit_registers`,
  `launch__occupancy_limit_warps`, block/grid size, registers/thread.
- **Scheduler Statistics:** `smsp__warps_active.avg.per_cycle_active`,
  `smsp__warps_eligible.avg.per_cycle_active`,
  `smsp__issue_active.avg.pct_of_peak_sustained_active`.
- **Instruction Statistics:** `smsp__inst_executed.sum`, shared/global
  `smsp__sass_inst_executed_op_*` counts. Per-row values divide by grid size.
- **Memory:** `lts__t_sector_op_read_hit_rate.pct`, `dram__bytes_read.sum`,
  `dram__bytes_write.sum`, `dram__bytes.sum.per_second`,
  `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, global/local sectors
  and `derived__local_spilling_requests`. Read the CSV units row before values.
- Reported waves/SM are 15.52 Reg versus 3.45 ours at 4096 rows, calculated from
  grid/(SM count × resident-block limit). They are not measured sequential rounds
  or a stand-alone prediction of speedup.
- Source/SASS barriers count instruction sites; stall ratios are cycles per
  issued instruction, not percentages of kernel wall time. Profile durations
  (60.000/49.632µs at 4096 rows) remain separate from benchmark medians.

## Evidence and next step

- **PyTorch, 37×4096:** [Details](../../results/2026-10-27-offline-profiles/37x4096/pytorch_details.txt), [raw metrics](../../results/2026-10-27-offline-profiles/37x4096/pytorch_raw.csv), [SASS](../../results/2026-10-27-offline-profiles/37x4096/pytorch_sass.txt), [report](../../results/2026-10-27-offline-profiles/37x4096/pytorch.ncu-rep).
- **Ours, 37×4096:** [Details](../../results/2026-10-27-offline-profiles/37x4096/online_registers_details.txt), [raw metrics](../../results/2026-10-27-offline-profiles/37x4096/online_registers_raw.csv), [SASS](../../results/2026-10-27-offline-profiles/37x4096/online_registers_sass.txt), [report](../../results/2026-10-27-offline-profiles/37x4096/online_registers.ncu-rep).
- **PyTorch, 4096×4096:** [Details](../../results/2026-10-27-offline-profiles/4096x4096/pytorch_details.txt), [raw metrics](../../results/2026-10-27-offline-profiles/4096x4096/pytorch_raw.csv), [SASS](../../results/2026-10-27-offline-profiles/4096x4096/pytorch_sass.txt), [report](../../results/2026-10-27-offline-profiles/4096x4096/pytorch.ncu-rep).
- **Ours, 4096×4096:** [Details](../../results/2026-10-27-offline-profiles/4096x4096/online_registers_details.txt), [raw metrics](../../results/2026-10-27-offline-profiles/4096x4096/online_registers_raw.csv), [SASS](../../results/2026-10-27-offline-profiles/4096x4096/online_registers_sass.txt), [report](../../results/2026-10-27-offline-profiles/4096x4096/online_registers.ncu-rep).
- [Reg algorithm](12-pytorch-register-softmax.md), [retained Online algorithm](14-retained-online-softmax.md), [prior crossover](16-small-row-count-profile-crossover.md).
- [Benchmark medians](../../results/2026-10-31-softmax-128k/combined_summary.csv) · [Section index](README.md).
- Next: discuss this row-count crossover, then compare 4096×8192 with4096×16384.
  Existing downloaded evidence only; no new GPU experiments or kernel changes.
