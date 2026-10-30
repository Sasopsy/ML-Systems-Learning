# 2026-10-31 — Extend softmax comparisons through 128K dims

## Context

- Project 3, BUILD explicitly delegated. User prefers **dims** in labels and requested extending the existing comparison to about 128K dims.
- Starting committed checkpoint: `b664fcb`; configurable-N work remains local and uncommitted.
- Same Linux H100 80GB HBM3 host, CUDA 13.0.88, Nsight Compute 2025.3.1.0, kvforge Python 3.12 and PyTorch 2.11.0+cu130.

## Work and evidence — in progress

- Added CUDA bridge specializations for 16384, 32768, 65536 and 131072 dims; kernel algorithm unchanged. Still N = 128 for timing/profiling, rows 37/256/4096, all three variants.
- Benchmark CLI now accepts selected `--dims`, `--output` and `--library`; new CSVs/metadata use dims. Profile CLI uses `--dims` with legacy `--cols` compatibility. Capture/index tools accept separate output folders and normalize old cols-based evidence.
- Built the new bridge `softmax_sweep_128k.so` with O3/C++17/lineinfo/sm_90/PIC. Initial independent FP64 checks passed at all four added dims values, rows 37.
- Compiler observations: 16384 dims uses 168 registers/thread and zero spills. At 32768/65536/131072 dims, 255 registers/thread, with stack bytes 64/1088/3136, spill-store bytes 64/1088/3136, spill-load bytes 172/3228/9372. These are compiler resource reports, not dynamic whole-kernel traffic.
- Timing extension uses identical fixture, tolerances, 50 warmups, nine rounds × 500 calls, rotating order, direct and CUDA Graph modes, reference checks before timing and after every batch. No tolerance changes or new block-size timing sweep.
- The extension writes to `project_03_gpu_operator_library/results/2026-10-31-softmax-128k/`; paired profile destination is `project_03_gpu_operator_library/results/2026-10-31-offline-profiles-128k/`. Earlier experiment folders are preserved.
- At this checkpoint, both modes passed for all added dims at rows 37 and 256, plus 4096×16384 and 4096×32768. Remaining 4096-row cases and paired captures are in progress; do not infer completion from this entry.

## Learning and handoff

- At N = 128, 131072 dims requires retaining 1024 original values per thread. The variant name does not guarantee register residency; inspect spills and local-memory traffic.
- At 4096×16384, graph medians already favor PyTorch despite no compiler spills. Do not attribute the whole crossover to spills without further evidence.
- Next: finish the extension, save complete tables and paired profiles, validate the offline manifest, then inspect the 16K-to-32K transition. Assisted work; learner conclusion not yet provided, no independent mastery claim.
- Code/docs are uncommitted and unpushed. Results ignored by Git; download separately. No shutdown action.

## Completion — extended timing and offline reports

- All 12 added shapes completed: 36 initial checks plus validation after every batch, 648 samples, 72 summaries. Existing 18-shape results were preserved; combined_summary.csv labels which experiment produced each value.
- Captured all 24 paired full reports with matching shape, FP32 fixture, 50 warmups and preallocated output; independent NumPy FP64 pre/post checks passed. Used full replay with cache/clock control none, source import, one selected launch and a bounded 600-second timeout per capture. Exact commands/warnings are preserved.
- Additional `compute-sanitizer --tool memcheck --error-exitcode 1` runs passed the retained kernel at three rows for each new dims value. Logs in the package's validation_evidence. Other sanitizer tools were not rerun for these new large dims.
- Regenerated text/raw/SASS exports, per-shape comparisons, source snapshots and checksum manifest. All manifest entries and local README links verified. New bundle includes original sweep evidence for small-dims provenance; its profile reports cover only the added large-dims cases.
- Runtime evidence at 37 rows agrees with compiler spilling: derived local spilling requests were 0 / 8456 / 124712 / 352041 at dims 16384 / 32768 / 65536 / 131072 respectively. These request counts are not bytes or a causal runtime percentage.
- No algorithm change or launch-size tuning; the name online_registers is retained even when the compiler spills. Static compiler byte counts are not measured whole-kernel memory traffic.

### New graph medians at 4096 rows (microseconds per call)

| Dims | PyTorch | Online | Online registers |
|---:|---:|---:|---:|
| 16384 | 214.255 | 317.428 | 261.213 |
| 32768 | 654.465 | 624.104 | 1646.811 |
| 65536 | 1400.800 | 1241.854 | 4481.342 |
| 131072 | 2805.656 | 2534.497 | 12867.443 |

- Means/min/max and direct timings are saved alongside all raw batches. No per-call p99, confidence intervals or causal runtime attribution from a single counter. Cache state, occupancy, instruction work, spilling and shape-dependent PyTorch dispatch may all matter.
- Added revision-note observations and updated the offline handoff/index. Next: inspect the 16K-to-32K transition. Learner conclusion remains not yet provided; assisted work is not independent mastery. Everything from this extension remains local/uncommitted; no Git push or shutdown performed.

## Git handoff — configurable softmax and 128K dims

- User authorized committing and pushing the kernel, benchmark/profile tools, validation harness, notes and progress records.
- Results, profiler reports and generated binaries remain ignored and must be downloaded separately. Earlier local/uncommitted entries describe historical checkpoints.
- This entry prepares the authorized handoff; remote synchronization is verified after pushing. No new GPU runs, download verification or shutdown are part of this step.
- Next: inspect the saved 4096×16384 and 4096×32768 profile pairs offline.
