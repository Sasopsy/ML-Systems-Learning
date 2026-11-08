# 18 — Why PyTorch wins again: 4096×8192 versus 4096×16384

**Best-supported explanation:** our retained buffer and online recurrence grow,
reducing residency and readiness; PyTorch switches to a small-buffer, vectorized
General kernel. Caches serve most of its extra input passes. **No spilling occurs
at 16384** in these profiles.

## Timing and algorithm mapping

Unprofiled benchmark medians in microseconds/call:

| Shape | PyTorch graph | Ours graph | PyTorch direct | Ours direct | Graph winner |
|---|---:|---:|---:|---:|---|
| 4096 × 8192 | 145.427 | 100.415 | 146.931 | 102.817 | Ours, 1.45× |
| 4096 × 16384 | 214.255 | 261.213 | 215.990 | 262.812 | PyTorch, 1.22× |

| Property | Torch8192 | Ours8192 | Torch16384 | Ours16384 |
|---|---:|---:|---:|---:|
| Kernel family | Reg | Retained Online | General | Retained Online |
| Threads per row | 1024 | 128 | 1024 | 128 |
| Original values retained per thread | 8 | 64 | Reused4-float chunks | 128 |
| Source input passes | 1 | 1 | 3 | 1 |
| Main global load/store width | Scalar | Scalar | 4 floats | Scalar |

All four launch 4096 blocks, one per row. There is ample queued work; the key
question is how much independent work can reside and make progress together.

## 1. Retaining the larger row reduces our residency before spilling

- Ours keeps `vals[ncols/128]`: **64→128 FP32 values/thread**. Each remains live
  through the online update and pair reductions until output. The local recurrence
  also grows from 63 to 127 dependent updates after initializing from the first value.
- Register allocation rises **80→168 per thread**. The profiler's register limit
  falls **six→three resident 128-thread blocks/SM**, or 24→12 resident warps.
- Achieved occupancy follows that ceiling: **35.51%→18.39%**. Our eligible warps
  and issue activity fall too; there is less other work to cover the longer
  per-thread dependency sequence.
- Local spilling requests **and local load/store sectors are zero** in all four
  captures. Register pressure can hurt through reduced residency without spills.

## 2. General buys residency by rereading instead of retaining

- Reg at 8192 holds eight inputs/thread and uses 37 registers/thread. Its 1024-thread
  block is register-limited to **one block/SM**.
- General at 16384 reuses small four-float input/output chunks. It needs only
  **32 registers/thread**, allowing **two 1024-thread blocks/SM**, or 64 warps.
- At 16384, the contrast is **up to 64 resident warps for General versus 12 for ours**.
  Comparing block counts alone (two versus three) would hide their different sizes.
- General still performs local max/sum loops and two block reductions. At16384,
  each thread processes16 values per pass in four-value chunks, versus128 retained
  values and online updates per thread in ours (127 updates after initialization).
  Three input passes avoid the large live buffer and long online recurrence.

| Metric | Torch8192 (Reg) | Ours8192 | Torch16384 (General) | Ours16384 |
|---|---:|---:|---:|---:|
| Registers/thread | 37 | 80 | 32 | 168 |
| Register-limited blocks/SM | 1 | 6 | 2 | 3 |
| Achieved occupancy, % | 49.19 | 35.51 | 93.79 | 18.39 |
| Active warps/scheduler | 7.91 | 5.75 | 15.15 | 2.96 |
| Eligible warps/scheduler | 2.06 | 0.78 | 2.28 | 0.41 |
| Issue-active cycles, % | 52.87 | 48.52 | 46.44 | 32.72 |
| Executed warp instructions | 76,926,976 | 48,528,404 | 101,306,368 | 87,342,088 |
| Local spilling requests | 0 | 0 | 0 | 0 |

At 16384, General has 2.28 eligible warps/scheduler versus 0.41 for ours, and
issue activity 46.44% versus 32.72%. These support better dependency hiding.
They are active-cycle statistics, not percentages of the whole GPU occupied.
General's issue rate does **not** improve over Reg8192 (52.87%); its advantage
here is relative to our 16384 kernel.

## 3. Four-float accesses reduce memory-instruction work

SASS shows General's `LDG.E.128` and `STG.E.128` vector instructions. Its dynamic
counts match the full vector main path:

| 16384-width metric | PyTorch General | Ours |
|---|---:|---:|
| Global-load warp instructions | 1,572,864 | 2,097,152 |
| Global-store warp instructions | 524,288 | 2,097,152 |
| Global-load sectors | 25,165,824 | 8,388,608 |
| Global-store sectors | 8,388,608 | 8,388,608 |

- General performs **three passes × one quarter as many load instructions/pass**:
  `3/4` as many loads as ours, despite requesting three times the input bytes.
- Its one vectorized output pass needs **one quarter as many store instructions**.
- This reduces instruction overhead per byte. It does not mean three passes move
  fewer input bytes. At 8192 both kernels use scalar accesses and their global
  load/store instruction counts match.

## 4. Three input passes do not become three DRAM passes

Each input tensor at 16384 occupies 256MiB. General requests 768MiB of global loads
at L1/TEX versus our 256MiB, but measured DRAM reads are:

| 16384-width memory metric | PyTorch General | Ours |
|---|---:|---:|
| DRAM reads, decimal MB | 274.981 | 268.563 |
| DRAM writes, decimal MB | 267.311 | 266.754 |
| Global-load L1 hit rate | 43.59% | 0% |
| DRAM throughput / peak | 75.30% | 57.81% |
| Profiled DRAM bandwidth | 2.52TB/s | 1.94TB/s |

- General reads only **~2.4% more from DRAM**, not 3×. Its extra passes get substantial
  L1/L2 reuse. Retaining the entire input saves cache accesses, but saves much less
  DRAM traffic here than a three-pass-versus-one-pass comparison might suggest.
- Relevant General counters: L1 load-hit sectors 10,970,528; L2 TEX-read-hit
  sectors 5,554,732. L2 TEX-read total 14,329,836. Do not apply aggregate L2 read-hit
  percentages to a differently scoped L1 count; replay counters may not reconcile
  exactly across passes. Direct DRAM-byte counters establish the traffic contrast.
- Higher achieved bandwidth is an outcome of completing similar DRAM traffic
  faster, not independent proof of its cause or proof of full bandwidth saturation.

## Why fewer total instructions no longer wins

- At 8192 ours executes **36.9% fewer warp instructions** than Reg, with relatively
  close issue rates: 48.52% versus 52.87%. Ours wins despite lower occupancy.
- At 16384 ours executes only **13.8% fewer** than General, while its issue rate
  falls further behind: 32.72% versus 46.44%. The reduced work advantage no longer
  compensates for reduced execution rate and the longer local recurrence.
- Our branch uniformity improves 84.22%→91.77%; worsening divergence is not supported
  as the cause. Neither spilling nor reduced PyTorch DRAM traffic explains the win.

**Interpretation:** retaining more data has a resource cost. Here PyTorch's small
buffers, vector memory operations, many resident warps and cache reuse make its
extra input passes affordable. Our exact-width 128-thread design retains too much
per-thread state to sustain its previous concurrency advantage. These observations
support a combined explanation; they do not isolate a speedup fraction per factor.

## Evidence and scope

- Resource/readiness: `launch__registers_per_thread`,
  `launch__occupancy_limit_registers`, `sm__warps_active.avg.pct_of_peak_sustained_active`,
  `smsp__warps_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`,
  `smsp__issue_active.avg.pct_of_peak_sustained_active`.
- Instruction work: `smsp__inst_executed.sum`,
  `smsp__sass_inst_executed_op_global_ld.sum`, corresponding `_st.sum`.
- Traffic: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` and `_st.sum`,
  `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct`,
  `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum`, `dram__bytes_read.sum`,
  `dram__bytes_write.sum`, `dram__bytes.sum.per_second`,
  `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`. Read units: DRAM bytes
  are exported in decimal Mbyte; a sector is 32 bytes.
- Spills: `derived__local_spilling_requests` plus local load/store sectors.
- Benchmark results are separate from profiler durations 148.160/103.328µs at 8192
  and 214.848/276.256µs at 16384 (PyTorch/ours). Evidence comes from the two existing
  benchmark packages on the same H100/environment; no new run is required to
  discuss the observed comparison. Profile cache state need not match every
  benchmark replay exactly.

- **PyTorch,4096×8192:** [Details](../../results/2026-10-27-offline-profiles/4096x8192/pytorch_details.txt), [raw counters](../../results/2026-10-27-offline-profiles/4096x8192/pytorch_raw.csv), [SASS](../../results/2026-10-27-offline-profiles/4096x8192/pytorch_sass.txt), [report](../../results/2026-10-27-offline-profiles/4096x8192/pytorch.ncu-rep).
- **Ours,4096×8192:** [Details](../../results/2026-10-27-offline-profiles/4096x8192/online_registers_details.txt), [raw counters](../../results/2026-10-27-offline-profiles/4096x8192/online_registers_raw.csv), [SASS](../../results/2026-10-27-offline-profiles/4096x8192/online_registers_sass.txt), [report](../../results/2026-10-27-offline-profiles/4096x8192/online_registers.ncu-rep).
- **PyTorch,4096×16384:** [Details](../../results/2026-10-31-offline-profiles-128k/4096x16384/pytorch_details.txt), [raw counters](../../results/2026-10-31-offline-profiles-128k/4096x16384/pytorch_raw.csv), [SASS](../../results/2026-10-31-offline-profiles-128k/4096x16384/pytorch_sass.txt), [report](../../results/2026-10-31-offline-profiles-128k/4096x16384/pytorch.ncu-rep).
- **Ours,4096×16384:** [Details](../../results/2026-10-31-offline-profiles-128k/4096x16384/online_registers_details.txt), [raw counters](../../results/2026-10-31-offline-profiles-128k/4096x16384/online_registers_raw.csv), [SASS](../../results/2026-10-31-offline-profiles-128k/4096x16384/online_registers_sass.txt), [report](../../results/2026-10-31-offline-profiles-128k/4096x16384/online_registers.ncu-rep).
- Algorithms: [Reg](12-pytorch-register-softmax.md), [General](13-pytorch-general-softmax.md), [retained Online](14-retained-online-softmax.md).
- [Timing source](../../results/2026-10-31-softmax-128k/combined_summary.csv),
  [recorded PyTorch source](../../results/2026-10-31-offline-profiles-128k/source/pytorch/SoftMax.cu),
  [recorded custom source](../../results/2026-10-31-offline-profiles-128k/source/softmax.cu).
- [Section index](README.md). Next: discuss this crossover, then synthesize the
  five selected cases into the blog narrative. No new GPU run or kernel edits.
