# 2026-10-31 — Retained softmax template cleanup and experiment refresh

## Context

- Project 3, BUILD explicitly delegated by the user: simplify the template API and replace this experiment's benchmark/profiler outputs.
- Starting commit: `b664fcbf97ed34ac7c0d60b629af505c82b1d872`.
- Environment: Linux GPU host, NVIDIA H100 80GB HBM3, driver 580.173.02, CUDA 13.0.88, Nsight Compute 2025.3.1.0; kvforge Python 3.12, PyTorch 2.11.0+cu130.

## Work and evidence

- Changed `softmax_online_registers<N, Cols>` to `softmax_online_registers<dims>`, default 1024. Internal block size remains 128; updated standalone tests, timing/profile harnesses and sweep bridge. Other softmax kernels retain their existing APIs.
- Four builds with `nvcc -std=c++17 -O3 -lineinfo -arch=sm_90 --ptxas-options=-v`; bridge additionally uses `--shared -Xcompiler -fPIC`.
- Ran the expanded softmax suite: 62 variant/case runs passed. Ran `compute-sanitizer --tool` memcheck/racecheck/synccheck/initcheck, each with `--error-exitcode 1`: all passed. Retained variant sanitizer fixtures cover width 1024 only.
- Reran `benchmarks/time_cuda.cu` and `benchmarks/sweep_softmax.py`; independent reference checks passed. Refreshed 18 shapes × three variants × two modes × nine rounds = 972 samples.
- Ran `benchmarks/capture_winning_cases.py --all-cases`: 36 full reports, both kernels at all 18 shapes; independent NumPy FP64 pre/post checks passed. Separately reran two LaunchStats/Occupancy captures at 4096×1024, also with independent checks.
- Refreshed full 4096×1024 comparison files are copies of the new sweep pair, not additional measurements. Regenerated details/raw/SASS, comparison pages, source snapshots and checksum manifest with `benchmarks/build_offline_profile_index.py`.
- Replaced outputs in `project_03_gpu_operator_library/results/2026-10-27-register-online/`, `2026-10-27-softmax-sweep/` and `2026-10-27-offline-profiles/`. Folder names are retained for existing links; contents identify the 2026-10-31 refresh. Old versions are outside results in ignored `artifacts/softmax-before-dims-2026-10-31/`.

## Experiment

- Same deterministic FP32 contiguous fixture, rows 37/256/4096 × widths 128/512/1024/2048/4096/8192. PyTorch preallocated output, current conditional online, retained online. Fixed tolerances unchanged.
- Fifty warmups, nine rounds × 500 calls, rotating order, CUDA-event intervals with stop synchronization. Graph mode captures 500 operations and warms one replay. Setup, transfers and validation excluded; direct host submission paths differ. No per-call p99.
- Hypothesis: removing a template argument restricted to 128 should preserve intended computation/layout. Timing variation is an alternative explanation for small separate-run differences; no speedup expected or claimed from API cleanup.
- At 4096×1024, retained kernel name now has one template argument; resources/instructions remain 30 registers/thread and 9,516,912 executed warp instructions. Width specializations retain 22/24/30/40/56/80 registers with zero compiler spills.
- At 4096 rows, graph medians (PyTorch/retained, µs): 1024 = 9.248/12.780; 2048 = 28.695/25.502; 4096 = 58.893/49.412; 8192 = 146.120/100.706. Nine shapes favor retained in both modes, seven favor PyTorch, two change winner by mode. These are observed batch medians, not confidence intervals or universal superiority.
- Profiler replay is diagnostic; caches/clocks uncontrolled. PyTorch CUDA line mapping remains unavailable. Counter warnings remain in capture logs; unavailable counters are not zeros.

## Learning and handoff

- Clarified `N` means block threads, not row width; 128 threads make four warps, each retaining `dims / 128` inputs.
- Learner asked whether other values could work. Explained complete-warp and divisibility constraints, exact launch requirement, resource limits and block-size performance tradeoffs. The old `N == 128` restriction was experimental, not mathematical. No alternate block-size implementation or benchmark was requested or executed.
- Added the explanation to the register-buffering revision note. Learner conclusion not yet provided; assisted results are not independent mastery.
- Next task: compare refreshed 4096×1024 and 4096×2048 paired profiles offline; a configurable-block-size experiment is a separate possible future step.
- Results are ignored by Git and require separate download. Code/docs remain local/uncommitted. No new commit/push, download verification on Mac or shutdown performed.

## Follow-up — configurable N requested, intermediate outputs superseded

- User clarified that N should remain configurable under valid conditions and explicitly requested another full benchmark/report refresh. Restored N with complete-warp/range conditions and positive dims divisible by N. Default dims stays 1024. There is no `N == 128` assertion.
- Reran the same build, original 62-case correctness suite and all four sanitizers, standalone timing, 18-shape direct/graph sweep and all 36 full paired captures after this final edit. Separately recaptured both occupancy reports. Every command exited zero and all independent checks passed.
- Added `benchmarks/validate_register_block_sizes.cu`: checks every multiple of 32 from 32 through 1024, including non-powers of two. For each N, three rows at dims = N and dims = 8*N, independent FP64 reference plus invariants and NaN sentinel. All 64 checks and four sanitizer tools passed; logs `block_sizes_*` are included in register-online results and offline validation evidence. This is bounded correctness coverage, not timing coverage for all N or exhaustive shapes.
- Final timing/profiling uses N = 128, explicitly preserving the prior workload/mapping. Kernel name in reports is `softmax_online_registers<128, dims>`. Differences from earlier N = 128 runs should not be attributed to this source-level rename/restriction removal without controlled evidence.
- Intermediate fixed-block results moved outside results to ignored `artifacts/softmax-fixed128-before-configurable-2026-10-31/`. Original pre-session results remain in the separate prior backup. The three active experiment folders contain final-version outputs, including regenerated README tables, source/metric exports and manifest; every checksum verified.

### Final graph medians at 4096 rows (microseconds per call)

| Width | PyTorch | Current online | Online registers |
|---:|---:|---:|---:|
| 128 | 2.549 | 6.638 | 5.591 |
| 512 | 5.055 | 10.645 | 8.642 |
| 1024 | 9.207 | 15.817 | 12.679 |
| 2048 | 28.460 | 32.341 | 25.487 |
| 4096 | 58.885 | 81.342 | 49.372 |
| 8192 | 145.427 | 160.694 | 100.415 |

- Full direct/graph tables for all 18 shapes are in `project_03_gpu_operator_library/results/2026-10-27-softmax-sweep/README.md`. Profile replay durations remain separate from these batch medians; no per-call p99 or optimal-block-size claim.
- Updated revision notes to reflect configurable N. Next task is offline profile comparison; a block-size performance sweep would be a separate experiment. Code/docs remain uncommitted and unpushed; results ignored, no rental shutdown or verified Mac download.
